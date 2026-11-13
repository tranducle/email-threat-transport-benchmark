#!/usr/bin/env python3
"""Prior-shift counterfactuals and 9-seed score-ceiling audit (writes the exp1_*.csv tables).

Executes:
1. GPU scoring of all 9 saved A2 mDistilBERT seed checkpoints
   (seeds 20260503, 20260504, 20260505 across fit_trec5, fit_trec6, final_models)
   on held-out TREC6 (14,685), held-out TREC5 (46,640), SpaPhish v5 (1,395),
   and clean CEAS-08 (38,920). Verifies 3-seed mean parity (< 1e-5 max abs diff)
   against the saved ensemble score parquets.
2. Per-seed vs 3-seed ensemble upper score ceiling and exhaustive reciprocal
   cutoff feasibility audit across all 9 cross-route seed pairs (3 x 3) and the
   3-seed ensemble at alpha in {0.005, 0.01, 0.02}.
3. Full prior-shift & recalibration suite (P0, P3, P3b_EM, P3b_ORACLE, P4,
   P5_EM, P6_BBSE_soft, P6_BBSE_hard, P5_ORACLE, plus full-stream reference
   P1_FULL and P2_FULL) across all routes, models (A1, A2, and individual A2 seeds),
   corpora, and alpha in {0.005, 0.01, 0.02}, with exact Clopper-Pearson 95% CIs
   and 2,000-replicate paired bootstrap intervals on key policy contrasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("THRESHOLD_TRANSPORT_DATA", REPO_ROOT / "data"))

ANCHOR_RUN = DATA_ROOT / "models" / "anchors"
DEV_DATA = DATA_ROOT / "processed" / "meajor_trec56_reconstructed.parquet"
SPA_DATA = DATA_ROOT / "processed" / "spaphish_confirmation_urlfixed.parquet"
SPA_PRED = DATA_ROOT / "scores" / "spaphish_frozen_scores.parquet"
CEAS_DATA = DATA_ROOT / "processed" / "ceas08_confirmation.parquet"
CEAS_PRED = DATA_ROOT / "scores" / "ceas08_frozen_scores.parquet"

PRED_DIR = DATA_ROOT / "scores"
RESULTS_DIR = DATA_ROOT / "results"
RUNS_DIR = DATA_ROOT / "results" / "receipts"

SEEDS = (20260503, 20260504, 20260505)
ALPHAS = (0.005, 0.01, 0.02)
MAX_LENGTH = 192
LOGIT_EPS = 1e-6
PROB_EPS = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CleanTextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer) -> None:
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def score_checkpoint(model_dir: Path, texts: list[str], batch_size: int = 256) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i : i + batch_size],
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            prob = torch.softmax(logits, dim=-1)[:, 1]
            probs.append(prob.detach().cpu().numpy().astype(np.float64))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(probs, axis=0)


def threshold_at_max_fpr(y: np.ndarray, scores: np.ndarray, alpha: float) -> float:
    neg = np.sort(np.asarray(scores, dtype=np.float64)[np.asarray(y, dtype=int) == 0])[::-1]
    allowed = int(math.floor(alpha * len(neg)))
    if allowed >= len(neg):
        return float(np.min(neg))
    return float(np.nextafter(neg[allowed], np.inf))


def cp_cutoff(neg_scores: np.ndarray, alpha: float, conf: float = 0.95) -> tuple[bool, float]:
    neg = np.sort(np.asarray(neg_scores, dtype=np.float64))[::-1]
    n0 = len(neg)
    if n0 == 0:
        return False, float("nan")
    x = np.arange(0, n0)
    u = beta.ppf(conf, x + 1, n0 - x)
    ok = np.flatnonzero(u <= alpha)
    if not ok.size:
        return False, float("nan")
    allowed = int(ok.max())
    return True, float(np.nextafter(neg[allowed], np.inf))


def clopper_pearson_two_sided(k: int, n: int, alpha_ci: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha_ci / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha_ci / 2.0, k + 1, n - k))
    return lo, hi


def eval_at_cutoff(y: np.ndarray, scores: np.ndarray, cutoff: float) -> dict[str, float | int]:
    y_arr = np.asarray(y, dtype=int)
    s_arr = np.asarray(scores, dtype=np.float64)
    if not math.isfinite(cutoff):
        return {
            "fpr": float("nan"),
            "fpr_ci_lo": float("nan"),
            "fpr_ci_hi": float("nan"),
            "recall": float("nan"),
            "fp": 0,
            "tp": 0,
            "flagged": 0,
            "n_neg": int((y_arr == 0).sum()),
            "n_pos": int((y_arr == 1).sum()),
        }
    flag = s_arr >= cutoff
    fp = int(flag[y_arr == 0].sum())
    tp = int(flag[y_arr == 1].sum())
    n0 = int((y_arr == 0).sum())
    n1 = int((y_arr == 1).sum())
    lo, hi = clopper_pearson_two_sided(fp, n0)
    return {
        "fpr": fp / max(1, n0),
        "fpr_ci_lo": lo,
        "fpr_ci_hi": hi,
        "recall": tp / max(1, n1),
        "fp": fp,
        "tp": tp,
        "flagged": int(flag.sum()),
        "n_neg": n0,
        "n_pos": n1,
    }


def ece15(y: np.ndarray, p: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, 16)
    val = 0.0
    for i in range(15):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < 14 else (p <= edges[i + 1]))
        if m.any():
            val += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return val


def saerens_adjust_scores(s: np.ndarray, pi_src: float, pi_tgt: float) -> np.ndarray:
    s_clip = np.clip(np.asarray(s, dtype=np.float64), PROB_EPS, 1.0 - PROB_EPS)
    pi_s = float(np.clip(pi_src, 1e-6, 1.0 - 1e-6))
    pi_t = float(np.clip(pi_tgt, 1e-6, 1.0 - 1e-6))
    num = (pi_t / pi_s) * s_clip
    den = num + ((1.0 - pi_t) / (1.0 - pi_s)) * (1.0 - s_clip)
    return num / den


def saerens_em_prior(s_target: np.ndarray, pi_src: float, max_iter: int = 500, tol: float = 1e-9) -> tuple[float, int]:
    pi_t = float(pi_src)
    for it in range(1, max_iter + 1):
        post = saerens_adjust_scores(s_target, pi_src, pi_t)
        pi_next = float(np.clip(post.mean(), 1e-6, 1.0 - 1e-6))
        if abs(pi_next - pi_t) < tol:
            return pi_next, it
        pi_t = pi_next
    return pi_t, max_iter


def bbse_priors(y_src: np.ndarray, s_src: np.ndarray, s_tgt: np.ndarray, cut: float = 0.5) -> dict[str, float]:
    pred_src = (s_src >= cut).astype(np.float64)
    tpr = float(pred_src[y_src == 1].mean())
    fpr = float(pred_src[y_src == 0].mean())
    q_tgt = float((s_tgt >= cut).mean())
    denom = tpr - fpr
    pi_hard = float(np.clip((q_tgt - fpr) / denom, 1e-6, 1.0 - 1e-6)) if abs(denom) > 1e-6 else float(y_src.mean())

    mu1 = float(s_src[y_src == 1].mean())
    mu0 = float(s_src[y_src == 0].mean())
    mu_tgt = float(s_tgt.mean())
    denom_soft = mu1 - mu0
    pi_soft = float(np.clip((mu_tgt - mu0) / denom_soft, 1e-6, 1.0 - 1e-6)) if abs(denom_soft) > 1e-6 else float(y_src.mean())
    return {"pi_bbse_soft": pi_soft, "pi_bbse_hard": pi_hard}


def p3_rank_map(target_scores: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    st = np.sort(target_scores)
    u = np.searchsorted(st, target_scores, side="right") / len(st)
    return np.quantile(ref_sorted, u, method="inverted_cdf")


def p3b_prior_quantile_cutoff(s_target: np.ndarray, alpha: float, tpr_src_at_alpha: float, pi_hat_tgt: float) -> float:
    q_flag = float(np.clip((1.0 - pi_hat_tgt) * alpha + pi_hat_tgt * tpr_src_at_alpha, 0.0, 1.0))
    n = len(s_target)
    allowed_flag = int(math.floor(q_flag * n))
    desc = np.sort(np.asarray(s_target, dtype=np.float64))[::-1]
    if allowed_flag <= 0:
        return float(np.nextafter(desc[0], np.inf))
    if allowed_flag >= n:
        return float(desc[-1])
    return float(np.nextafter(desc[allowed_flag], np.inf))


def exhaustive_reciprocal_feasibility(
    y_t6: np.ndarray, s_t6: np.ndarray, y_t5: np.ndarray, s_t5: np.ndarray, alpha: float
) -> dict[str, float | int]:
    """Exhaustive check over all unique observed scores across TREC6 and TREC5.

    Matches CSI_20261107 Eq. (feasible-set): counts candidate cutoffs in the union of
    observed scores (plus nextafter boundaries) where FPR <= alpha AND TP > 0 in BOTH sources.
    """
    cand = np.unique(np.concatenate([s_t6, s_t5]))
    # Sort ascending
    s6_neg = np.sort(s_t6[y_t6 == 0])
    s6_pos = np.sort(s_t6[y_t6 == 1])
    s5_neg = np.sort(s_t5[y_t5 == 0])
    s5_pos = np.sort(s_t5[y_t5 == 1])

    n6_neg, n6_pos = len(s6_neg), len(s6_pos)
    n5_neg, n5_pos = len(s5_neg), len(s5_pos)

    fp6 = n6_neg - np.searchsorted(s6_neg, cand, side="left")
    tp6 = n6_pos - np.searchsorted(s6_pos, cand, side="left")
    fp5 = n5_neg - np.searchsorted(s5_neg, cand, side="left")
    tp5 = n5_pos - np.searchsorted(s5_pos, cand, side="left")

    fpr6 = fp6 / n6_neg
    fpr5 = fp5 / n5_neg
    feasible = (fpr6 <= alpha) & (fpr5 <= alpha) & (tp6 > 0) & (tp5 > 0)

    tau_t6 = threshold_at_max_fpr(y_t6, s_t6, alpha)
    tau_t5 = threshold_at_max_fpr(y_t5, s_t5, alpha)
    tau_common = max(tau_t6, tau_t5)

    # Highest cutoff with nonzero recall in both sources
    both_pos_max = float(min(s6_pos[-1], s5_pos[-1]))
    r6_at_pos_max = eval_at_cutoff(y_t6, s_t6, both_pos_max)
    r5_at_pos_max = eval_at_cutoff(y_t5, s_t5, both_pos_max)

    return {
        "n_candidates": int(len(cand)),
        "n_feasible": int(feasible.sum()),
        "feasible_fraction": float(feasible.mean()),
        "tau_own_val_trec6": float(tau_t6),
        "tau_own_val_trec5": float(tau_t5),
        "tau_common": float(tau_common),
        "max_score_val_trec6": float(s_t6.max()),
        "max_score_val_trec5": float(s_t5.max()),
        "max_neg_val_trec6": float(s6_neg[-1]),
        "max_neg_val_trec5": float(s5_neg[-1]),
        "highest_nonzero_both_cutoff": both_pos_max,
        "fpr_trec6_at_highest_nonzero_both": float(r6_at_pos_max["fpr"]),
        "fpr_trec5_at_highest_nonzero_both": float(r5_at_pos_max["fpr"]),
        "recall_trec6_at_common": float(eval_at_cutoff(y_t6, s_t6, tau_common)["recall"]),
        "recall_trec5_at_common": float(eval_at_cutoff(y_t5, s_t5, tau_common)["recall"]),
        "fpr_trec6_at_common": float(eval_at_cutoff(y_t6, s_t6, tau_common)["fpr"]),
        "fpr_trec5_at_common": float(eval_at_cutoff(y_t5, s_t5, tau_common)["fpr"]),
    }


def paired_bootstrap_fpr_diff(
    y: np.ndarray, s_a: np.ndarray, t_a: float, s_b: np.ndarray, t_b: float, reps: int = 2000, seed: int = 20261112
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    neg_idx = np.flatnonzero(y == 0)
    flag_a = (s_a[neg_idx] >= t_a).astype(np.float64)
    flag_b = (s_b[neg_idx] >= t_b).astype(np.float64)
    diff = flag_b - flag_a
    n0 = len(neg_idx)
    # Vectorized multinomial/bootstrap over negative indices
    draws = rng.integers(0, n0, size=(reps, n0))
    means = diff[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_corpora() -> dict[str, pd.DataFrame]:
    dev = pd.read_parquet(DEV_DATA, columns=["row_id", "source", "label", "text_clean"])
    trec5 = dev[dev["source"] == "trec5"].copy().reset_index(drop=True)
    trec6 = dev[dev["source"] == "trec6"].copy().reset_index(drop=True)

    # Align order strictly with the anchor validation parquets.
    v6 = pd.read_parquet(ANCHOR_RUN / "fit_trec5__val_trec6" / "A2_MDISTILBERT" / "ensemble_validation.parquet")
    v5 = pd.read_parquet(ANCHOR_RUN / "fit_trec6__val_trec5" / "A2_MDISTILBERT" / "ensemble_validation.parquet")
    trec6 = v6[["row_id", "label"]].merge(trec6[["row_id", "text_clean"]], on="row_id", validate="one_to_one")
    trec5 = v5[["row_id", "label"]].merge(trec5[["row_id", "text_clean"]], on="row_id", validate="one_to_one")

    spa = pd.read_parquet(SPA_DATA, columns=["row_id", "label", "text_clean", "temporal_group"])
    ceas_full = pd.read_parquet(CEAS_DATA, columns=["row_id", "label", "text_clean", "contamination_removed"])
    ceas = ceas_full.loc[~ceas_full["contamination_removed"].astype(bool)].drop(columns=["contamination_removed"]).reset_index(drop=True)
    return {"trec5": trec5, "trec6": trec6, "spaphish": spa, "ceas08": ceas}


def run_smoke(corpora: dict[str, pd.DataFrame]) -> None:
    print("[prior-shift smoke] Checking GPU availability and 3-seed parity on 32 rows per corpus...", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is expected for checkpoint scoring")
    v6 = pd.read_parquet(ANCHOR_RUN / "fit_trec5__val_trec6" / "A2_MDISTILBERT" / "ensemble_validation.parquet")
    spa_p = pd.read_parquet(SPA_PRED)
    ceas_p = pd.read_parquet(CEAS_PRED)

    texts_t6 = corpora["trec6"]["text_clean"].iloc[:32].tolist()
    texts_spa = corpora["spaphish"]["text_clean"].iloc[:32].tolist()
    texts_ceas = corpora["ceas08"]["text_clean"].iloc[:32].tolist()

    s6_list, spa_list, ceas_list = [], [], []
    for seed in SEEDS:
        mdir = ANCHOR_RUN / "fit_trec5__val_trec6" / "A2_MDISTILBERT" / f"seed_{seed}"
        s6_list.append(score_checkpoint(mdir, texts_t6, batch_size=32))
        spa_list.append(score_checkpoint(mdir, texts_spa, batch_size=32))
        ceas_list.append(score_checkpoint(mdir, texts_ceas, batch_size=32))

    diff_t6 = float(np.max(np.abs(np.mean(s6_list, axis=0) - v6["score"].iloc[:32].to_numpy(np.float64))))
    diff_spa = float(np.max(np.abs(np.mean(spa_list, axis=0) - spa_p["fit_trec5_a2"].iloc[:32].to_numpy(np.float64))))
    diff_ceas = float(np.max(np.abs(np.mean(ceas_list, axis=0) - ceas_p["fit_trec5_a2"].iloc[:32].to_numpy(np.float64))))
    print(
        json.dumps(
            {
                "smoke_status": "PASS",
                "gpu": torch.cuda.get_device_name(0),
                "max_abs_diff_trec6_32": diff_t6,
                "max_abs_diff_spaphish_32": diff_spa,
                "max_abs_diff_ceas08_32": diff_ceas,
            },
            indent=2,
        ),
        flush=True,
    )
    if max(diff_t6, diff_spa, diff_ceas) > 1e-5:
        raise RuntimeError(f"Parity check failed: diffs=({diff_t6}, {diff_spa}, {diff_ceas})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Run 32-row parity smoke check only")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    t0 = time.time()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    corpora = load_corpora()
    if args.smoke:
        run_smoke(corpora)
        return

    # 1. Score or load cached per-seed predictions for all 9 checkpoints
    v6_ens = pd.read_parquet(ANCHOR_RUN / "fit_trec5__val_trec6" / "A2_MDISTILBERT" / "ensemble_validation.parquet")
    v5_ens = pd.read_parquet(ANCHOR_RUN / "fit_trec6__val_trec5" / "A2_MDISTILBERT" / "ensemble_validation.parquet")
    v6_a1 = pd.read_parquet(ANCHOR_RUN / "fit_trec5__val_trec6" / "A1_TFIDF_LR" / "tfidf_lr_validation.parquet")
    v5_a1 = pd.read_parquet(ANCHOR_RUN / "fit_trec6__val_trec5" / "A1_TFIDF_LR" / "tfidf_lr_validation.parquet")
    spa_frozen = pd.read_parquet(SPA_PRED)
    ceas_frozen = pd.read_parquet(CEAS_PRED)

    seed_trec6_path = PRED_DIR / "a2_per_seed_trec6.parquet"
    seed_trec5_path = PRED_DIR / "a2_per_seed_trec5.parquet"
    seed_spa_path = PRED_DIR / "a2_per_seed_spaphish.parquet"
    seed_ceas_path = PRED_DIR / "a2_per_seed_ceas08.parquet"

    if all(p.exists() for p in (seed_trec6_path, seed_trec5_path, seed_spa_path, seed_ceas_path)):
        print("[prior-shift] Reusing verified per-seed prediction parquets from scores/...", flush=True)
        df_s_t6 = pd.read_parquet(seed_trec6_path)
        df_s_t5 = pd.read_parquet(seed_trec5_path)
        df_s_spa = pd.read_parquet(seed_spa_path)
        df_s_ceas = pd.read_parquet(seed_ceas_path)
    else:
        print("[prior-shift] Scoring all 9 saved mDistilBERT checkpoints on GPU...", flush=True)
        df_s_t6 = pd.DataFrame({"row_id": corpora["trec6"]["row_id"], "label": corpora["trec6"]["label"].astype(int)})
        df_s_t5 = pd.DataFrame({"row_id": corpora["trec5"]["row_id"], "label": corpora["trec5"]["label"].astype(int)})
        df_s_spa = pd.DataFrame({"row_id": corpora["spaphish"]["row_id"], "label": corpora["spaphish"]["label"].astype(int)})
        df_s_ceas = pd.DataFrame({"row_id": corpora["ceas08"]["row_id"], "label": corpora["ceas08"]["label"].astype(int)})

        texts_t6 = corpora["trec6"]["text_clean"].tolist()
        texts_t5 = corpora["trec5"]["text_clean"].tolist()
        texts_spa = corpora["spaphish"]["text_clean"].tolist()
        texts_ceas = corpora["ceas08"]["text_clean"].tolist()

        for seed in SEEDS:
            t_s = time.time()
            m_t5 = ANCHOR_RUN / "fit_trec5__val_trec6" / "A2_MDISTILBERT" / f"seed_{seed}"
            df_s_t6[f"fit_trec5_seed_{seed}"] = score_checkpoint(m_t5, texts_t6, batch_size=args.batch_size)
            df_s_spa[f"fit_trec5_seed_{seed}"] = score_checkpoint(m_t5, texts_spa, batch_size=args.batch_size)
            df_s_ceas[f"fit_trec5_seed_{seed}"] = score_checkpoint(m_t5, texts_ceas, batch_size=args.batch_size)

            m_t6 = ANCHOR_RUN / "fit_trec6__val_trec5" / "A2_MDISTILBERT" / f"seed_{seed}"
            df_s_t5[f"fit_trec6_seed_{seed}"] = score_checkpoint(m_t6, texts_t5, batch_size=args.batch_size)
            df_s_spa[f"fit_trec6_seed_{seed}"] = score_checkpoint(m_t6, texts_spa, batch_size=args.batch_size)
            df_s_ceas[f"fit_trec6_seed_{seed}"] = score_checkpoint(m_t6, texts_ceas, batch_size=args.batch_size)

            m_fin = ANCHOR_RUN / "final_models" / "A2_MDISTILBERT" / f"seed_{seed}"
            df_s_spa[f"final_seed_{seed}"] = score_checkpoint(m_fin, texts_spa, batch_size=args.batch_size)
            df_s_ceas[f"final_seed_{seed}"] = score_checkpoint(m_fin, texts_ceas, batch_size=args.batch_size)
            print(f"  Scored seed {seed} across all routes in {time.time() - t_s:.1f}s", flush=True)

        df_s_t6.to_parquet(seed_trec6_path, index=False)
        df_s_t5.to_parquet(seed_trec5_path, index=False)
        df_s_spa.to_parquet(seed_spa_path, index=False)
        df_s_ceas.to_parquet(seed_ceas_path, index=False)

    # Parity verification across all 8 ensemble vectors
    parity_diffs = {
        "val_trec6_fit_trec5": float(
            np.max(np.abs(df_s_t6[[f"fit_trec5_seed_{s}" for s in SEEDS]].mean(axis=1).to_numpy() - v6_ens["score"].to_numpy()))
        ),
        "val_trec5_fit_trec6": float(
            np.max(np.abs(df_s_t5[[f"fit_trec6_seed_{s}" for s in SEEDS]].mean(axis=1).to_numpy() - v5_ens["score"].to_numpy()))
        ),
    }
    for route in ("fit_trec5", "fit_trec6", "final"):
        parity_diffs[f"spaphish_{route}"] = float(
            np.max(np.abs(df_s_spa[[f"{route}_seed_{s}" for s in SEEDS]].mean(axis=1).to_numpy() - spa_frozen[f"{route}_a2"].to_numpy()))
        )
        parity_diffs[f"ceas08_{route}"] = float(
            np.max(np.abs(df_s_ceas[[f"{route}_seed_{s}" for s in SEEDS]].mean(axis=1).to_numpy() - ceas_frozen[f"{route}_a2"].to_numpy()))
        )
    max_parity_diff = max(parity_diffs.values())
    if max_parity_diff > 1e-5:
        raise RuntimeError(f"Ensemble parity verification failed: {parity_diffs}")
    print(f"[prior-shift] 9-checkpoint ensemble parity PASS (max_abs_diff = {max_parity_diff:.3e})", flush=True)

    # 2. Seed ceiling and reciprocal cutoff feasibility audit
    y_t6 = df_s_t6["label"].to_numpy(int)
    y_t5 = df_s_t5["label"].to_numpy(int)
    y_spa = df_s_spa["label"].to_numpy(int)
    y_ceas = df_s_ceas["label"].to_numpy(int)

    seed_rows: list[dict] = []

    # Per-seed ceiling & external metrics table + reciprocal pairs
    # First: individual checkpoint ceiling characteristics across routes & corpora
    checkpoint_specs = [
        ("A1_ensemble", "fit_trec5", "trec6", y_t6, v6_a1["score"].to_numpy(np.float64)),
        ("A1_ensemble", "fit_trec6", "trec5", y_t5, v5_a1["score"].to_numpy(np.float64)),
        ("A2_ensemble", "fit_trec5", "trec6", y_t6, v6_ens["score"].to_numpy(np.float64)),
        ("A2_ensemble", "fit_trec6", "trec5", y_t5, v5_ens["score"].to_numpy(np.float64)),
    ]
    for route in ("fit_trec5", "fit_trec6", "final"):
        checkpoint_specs.append(("A1_ensemble", route, "spaphish", y_spa, spa_frozen[f"{route}_a1"].to_numpy(np.float64)))
        checkpoint_specs.append(("A1_ensemble", route, "ceas08", y_ceas, ceas_frozen[f"{route}_a1"].to_numpy(np.float64)))
        checkpoint_specs.append(("A2_ensemble", route, "spaphish", y_spa, spa_frozen[f"{route}_a2"].to_numpy(np.float64)))
        checkpoint_specs.append(("A2_ensemble", route, "ceas08", y_ceas, ceas_frozen[f"{route}_a2"].to_numpy(np.float64)))

    for seed in SEEDS:
        checkpoint_specs.append((f"A2_seed_{seed}", "fit_trec5", "trec6", y_t6, df_s_t6[f"fit_trec5_seed_{seed}"].to_numpy(np.float64)))
        checkpoint_specs.append((f"A2_seed_{seed}", "fit_trec6", "trec5", y_t5, df_s_t5[f"fit_trec6_seed_{seed}"].to_numpy(np.float64)))
        for route in ("fit_trec5", "fit_trec6", "final"):
            checkpoint_specs.append((f"A2_seed_{seed}", route, "spaphish", y_spa, df_s_spa[f"{route}_seed_{seed}"].to_numpy(np.float64)))
            checkpoint_specs.append((f"A2_seed_{seed}", route, "ceas08", y_ceas, df_s_ceas[f"{route}_seed_{seed}"].to_numpy(np.float64)))

    # Compute own dev cutoffs per seed for A2 and A1
    own_dev_cutoffs: dict[tuple[str, str, float], float] = {}
    for alpha in ALPHAS:
        own_dev_cutoffs[("A1_ensemble", "fit_trec5", alpha)] = threshold_at_max_fpr(y_t6, v6_a1["score"].to_numpy(np.float64), alpha)
        own_dev_cutoffs[("A1_ensemble", "fit_trec6", alpha)] = threshold_at_max_fpr(y_t5, v5_a1["score"].to_numpy(np.float64), alpha)
        own_dev_cutoffs[("A1_ensemble", "final", alpha)] = max(
            own_dev_cutoffs[("A1_ensemble", "fit_trec5", alpha)],
            own_dev_cutoffs[("A1_ensemble", "fit_trec6", alpha)],
        )
        own_dev_cutoffs[("A2_ensemble", "fit_trec5", alpha)] = threshold_at_max_fpr(y_t6, v6_ens["score"].to_numpy(np.float64), alpha)
        own_dev_cutoffs[("A2_ensemble", "fit_trec6", alpha)] = threshold_at_max_fpr(y_t5, v5_ens["score"].to_numpy(np.float64), alpha)
        own_dev_cutoffs[("A2_ensemble", "final", alpha)] = max(
            own_dev_cutoffs[("A2_ensemble", "fit_trec5", alpha)],
            own_dev_cutoffs[("A2_ensemble", "fit_trec6", alpha)],
        )
        for seed in SEEDS:
            t_5 = threshold_at_max_fpr(y_t6, df_s_t6[f"fit_trec5_seed_{seed}"].to_numpy(np.float64), alpha)
            t_6 = threshold_at_max_fpr(y_t5, df_s_t5[f"fit_trec6_seed_{seed}"].to_numpy(np.float64), alpha)
            own_dev_cutoffs[(f"A2_seed_{seed}", "fit_trec5", alpha)] = t_5
            own_dev_cutoffs[(f"A2_seed_{seed}", "fit_trec6", alpha)] = t_6
            own_dev_cutoffs[(f"A2_seed_{seed}", "final", alpha)] = max(t_5, t_6)

    for variant, route, corpus, y_arr, s_arr in checkpoint_specs:
        neg_s = s_arr[y_arr == 0]
        pos_s = s_arr[y_arr == 1]
        tau_own_01 = own_dev_cutoffs[(variant, route, 0.01)]
        tau_common_01 = own_dev_cutoffs[(variant, "final", 0.01)]
        r_own = eval_at_cutoff(y_arr, s_arr, tau_own_01)
        r_com = eval_at_cutoff(y_arr, s_arr, tau_common_01)
        seed_rows.append(
            {
                "record_type": "checkpoint_ceiling",
                "variant": variant,
                "route": route,
                "corpus": corpus,
                "n": int(len(y_arr)),
                "n_neg": int(len(neg_s)),
                "n_pos": int(len(pos_s)),
                "auroc": float(roc_auc_score(y_arr, s_arr)),
                "auprc": float(average_precision_score(y_arr, s_arr)),
                "brier": float(brier_score_loss(y_arr, s_arr)),
                "log_loss": float(log_loss(y_arr, np.clip(s_arr, PROB_EPS, 1.0 - PROB_EPS))),
                "ece15": ece15(y_arr, s_arr),
                "max_score": float(s_arr.max()),
                "max_neg_score": float(neg_s.max()),
                "max_pos_score": float(pos_s.max()),
                "q99_neg": float(np.quantile(neg_s, 0.99)),
                "q995_neg": float(np.quantile(neg_s, 0.995)),
                "share_neg_ge_0.999": float((neg_s >= 0.999).mean()),
                "share_pos_ge_0.999": float((pos_s >= 0.999).mean()),
                "n_distinct_scores": int(np.unique(s_arr).size),
                "tau_own_0.01": float(tau_own_01),
                "tau_common_0.01": float(tau_common_01),
                "fpr_at_own_0.01": float(r_own["fpr"]),
                "recall_at_own_0.01": float(r_own["recall"]),
                "fpr_at_common_0.01": float(r_com["fpr"]),
                "recall_at_common_0.01": float(r_com["recall"]),
                "flagged_at_common_0.01": int(r_com["flagged"]),
            }
        )

    reciprocal_rows: list[dict] = []
    # Ensemble A1 & A2 reciprocal checks + all 9 (seed_t5, seed_t6) pairs
    pair_specs = [
        ("A1_ensemble", "ensemble", "ensemble", v6_a1["score"].to_numpy(np.float64), v5_a1["score"].to_numpy(np.float64)),
        ("A2_ensemble", "ensemble", "ensemble", v6_ens["score"].to_numpy(np.float64), v5_ens["score"].to_numpy(np.float64)),
    ]
    for s_t5 in SEEDS:
        for s_t6 in SEEDS:
            pair_specs.append(
                (
                    f"A2_pair_{s_t5}_vs_{s_t6}",
                    str(s_t5),
                    str(s_t6),
                    df_s_t6[f"fit_trec5_seed_{s_t5}"].to_numpy(np.float64),
                    df_s_t5[f"fit_trec6_seed_{s_t6}"].to_numpy(np.float64),
                )
            )
    for pair_id, seed_fit_t5, seed_fit_t6, sc_val_t6, sc_val_t5 in pair_specs:
        for alpha in ALPHAS:
            rec = exhaustive_reciprocal_feasibility(y_t6, sc_val_t6, y_t5, sc_val_t5, alpha)
            reciprocal_rows.append(
                {
                    "pair_id": pair_id,
                    "seed_fit_trec5": seed_fit_t5,
                    "seed_fit_trec6": seed_fit_t6,
                    "alpha": alpha,
                    **rec,
                }
            )

    df_seed_ceiling = pd.DataFrame(seed_rows)
    df_reciprocal = pd.DataFrame(reciprocal_rows)
    df_seed_ceiling.to_csv(RESULTS_DIR / "exp1_seed_ceiling_audit.csv", index=False)
    df_reciprocal.to_csv(RESULTS_DIR / "exp1_reciprocal_feasibility.csv", index=False)

    # 3. Prior-shift & threshold policy suite across all corpora, routes, models, and alphas
    # Source training positive prevalences:
    pi_train = {
        "fit_trec5": float(y_t5.mean()),  # 19235 / 46640 = 0.412414
        "fit_trec6": float(y_t6.mean()),  # 3503 / 14685 = 0.238543
        "final": float(np.concatenate([y_t5, y_t6]).mean()),  # 22738 / 61325 = 0.370779
    }

    # Build development reference per model (A1, A2, and per-seed A2)
    model_dev_ref: dict[str, dict] = {}
    model_sources = {
        "A1": (v6_a1["score"].to_numpy(np.float64), v5_a1["score"].to_numpy(np.float64)),
        "A2": (v6_ens["score"].to_numpy(np.float64), v5_ens["score"].to_numpy(np.float64)),
    }
    for seed in SEEDS:
        model_sources[f"A2_seed_{seed}"] = (
            df_s_t6[f"fit_trec5_seed_{seed}"].to_numpy(np.float64),
            df_s_t5[f"fit_trec6_seed_{seed}"].to_numpy(np.float64),
        )

    for m_name, (s6_dev, s5_dev) in model_sources.items():
        pooled_s = np.concatenate([s5_dev, s6_dev])
        pooled_y = np.concatenate([y_t5, y_t6])
        pooled_sorted = np.sort(pooled_s)
        z = np.log(np.clip(pooled_s, LOGIT_EPS, 1.0 - LOGIT_EPS) / (1.0 - np.clip(pooled_s, LOGIT_EPS, 1.0 - LOGIT_EPS)))
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
        lr.fit(z.reshape(-1, 1), pooled_y)
        a_platt, b_platt = float(lr.coef_[0, 0]), float(lr.intercept_[0])

        def apply_platt(arr: np.ndarray, a=a_platt, b=b_platt) -> np.ndarray:
            zz = np.log(np.clip(arr, LOGIT_EPS, 1.0 - LOGIT_EPS) / (1.0 - np.clip(arr, LOGIT_EPS, 1.0 - LOGIT_EPS)))
            return 1.0 / (1.0 + np.exp(-(a * zz + b)))

        cuts = {}
        cuts_cal = {}
        tpr_at_alpha = {}
        for alpha in ALPHAS:
            t_fit5 = threshold_at_max_fpr(y_t6, s6_dev, alpha)
            t_fit6 = threshold_at_max_fpr(y_t5, s5_dev, alpha)
            t_com = max(t_fit5, t_fit6)
            cuts[alpha] = {"fit_trec5": t_fit5, "fit_trec6": t_fit6, "final": t_com, "common": t_com}

            t_fit5_cal = threshold_at_max_fpr(y_t6, apply_platt(s6_dev), alpha)
            t_fit6_cal = threshold_at_max_fpr(y_t5, apply_platt(s5_dev), alpha)
            cuts_cal[alpha] = {"fit_trec5": t_fit5_cal, "fit_trec6": t_fit6_cal, "final": max(t_fit5_cal, t_fit6_cal), "common": max(t_fit5_cal, t_fit6_cal)}

            tpr_at_alpha[alpha] = {
                "fit_trec5": float((s6_dev[y_t6 == 1] >= t_fit5).mean()),
                "fit_trec6": float((s5_dev[y_t5 == 1] >= t_fit6).mean()),
                "final": float((pooled_s[pooled_y == 1] >= t_com).mean()),
            }

        model_dev_ref[m_name] = {
            "s_val_trec6": s6_dev,
            "s_val_trec5": s5_dev,
            "pooled_s": pooled_s,
            "pooled_y": pooled_y,
            "pooled_sorted": pooled_sorted,
            "platt_fn": apply_platt,
            "cuts": cuts,
            "cuts_cal": cuts_cal,
            "tpr_at_alpha": tpr_at_alpha,
        }

    # Evaluate policies across all target streams
    eval_streams: list[tuple[str, str, str, np.ndarray, np.ndarray]] = []
    for m_name in model_sources:
        eval_streams.append(("trec6", m_name, "fit_trec5", y_t6, model_dev_ref[m_name]["s_val_trec6"]))
        eval_streams.append(("trec5", m_name, "fit_trec6", y_t5, model_dev_ref[m_name]["s_val_trec5"]))
        for route in ("final", "fit_trec5", "fit_trec6"):
            if m_name == "A1":
                s_sp = spa_frozen[f"{route}_a1"].to_numpy(np.float64)
                s_ce = ceas_frozen[f"{route}_a1"].to_numpy(np.float64)
            elif m_name == "A2":
                s_sp = spa_frozen[f"{route}_a2"].to_numpy(np.float64)
                s_ce = ceas_frozen[f"{route}_a2"].to_numpy(np.float64)
            else:
                seed_num = m_name.split("_")[-1]
                s_sp = df_s_spa[f"{route}_seed_{seed_num}"].to_numpy(np.float64)
                s_ce = df_s_ceas[f"{route}_seed_{seed_num}"].to_numpy(np.float64)
            eval_streams.append(("spaphish", m_name, route, y_spa, s_sp))
            eval_streams.append(("ceas08", m_name, route, y_ceas, s_ce))

    policy_rows: list[dict] = []
    bootstrap_contrasts: list[dict] = []

    for target, m_name, route, y_tgt, s_tgt in eval_streams:
        ref = model_dev_ref[m_name]
        # Reference labeled validation stream for BBSE confusion/mean estimation:
        if route == "fit_trec5":
            y_src_cal, s_src_cal = y_t6, ref["s_val_trec6"]
        elif route == "fit_trec6":
            y_src_cal, s_src_cal = y_t5, ref["s_val_trec5"]
        else:
            y_src_cal, s_src_cal = ref["pooled_y"], ref["pooled_s"]

        pi_src = pi_train[route]
        pi_true = float(y_tgt.mean())
        pi_em, em_iters = saerens_em_prior(s_tgt, pi_src)
        bb = bbse_priors(y_src_cal, s_src_cal, s_tgt, cut=0.5)

        s_p3 = p3_rank_map(s_tgt, ref["pooled_sorted"])
        s_p4 = ref["platt_fn"](s_tgt)
        s_em = saerens_adjust_scores(s_tgt, pi_src, pi_em)
        s_bb_soft = saerens_adjust_scores(s_tgt, pi_src, bb["pi_bbse_soft"])
        s_bb_hard = saerens_adjust_scores(s_tgt, pi_src, bb["pi_bbse_hard"])
        s_oracle = saerens_adjust_scores(s_tgt, pi_src, pi_true)

        ll_raw = float(log_loss(y_tgt, np.clip(s_tgt, PROB_EPS, 1.0 - PROB_EPS)))
        ll_em = float(log_loss(y_tgt, np.clip(s_em, PROB_EPS, 1.0 - PROB_EPS)))
        ll_bb_soft = float(log_loss(y_tgt, np.clip(s_bb_soft, PROB_EPS, 1.0 - PROB_EPS)))
        ll_bb_hard = float(log_loss(y_tgt, np.clip(s_bb_hard, PROB_EPS, 1.0 - PROB_EPS)))
        ll_oracle = float(log_loss(y_tgt, np.clip(s_oracle, PROB_EPS, 1.0 - PROB_EPS)))

        for alpha in ALPHAS:
            tau_common = ref["cuts"][alpha]["common"]
            tau_own = ref["cuts"][alpha][route]
            tau_cal_common = ref["cuts_cal"][alpha]["common"]
            tpr_src = ref["tpr_at_alpha"][alpha][route]

            tau_p3b_em = p3b_prior_quantile_cutoff(s_tgt, alpha, tpr_src, pi_em)
            tau_p3b_oracle = p3b_prior_quantile_cutoff(s_tgt, alpha, tpr_src, pi_true)
            tau_p1_full = threshold_at_max_fpr(y_tgt, s_tgt, alpha)
            cert_full, tau_p2_full = cp_cutoff(s_tgt[y_tgt == 0], alpha, conf=0.95)

            policy_defs = [
                ("P0_COMMON", s_tgt, tau_common, pi_src, ll_raw),
                ("P0_OWN", s_tgt, tau_own, pi_src, ll_raw),
                ("P3_MARGINAL_QUANTILE", s_p3, tau_common, pi_src, ll_raw),
                ("P3b_EM_PRIOR_QUANTILE", s_tgt, tau_p3b_em, pi_em, ll_em),
                ("P3b_ORACLE_PRIOR_QUANTILE", s_tgt, tau_p3b_oracle, pi_true, ll_oracle),
                ("P4_PLATT_DEV", s_p4, tau_cal_common, pi_src, float(log_loss(y_tgt, np.clip(s_p4, PROB_EPS, 1.0 - PROB_EPS)))),
                ("P5_SAERENS_EM", s_em, tau_common, pi_em, ll_em),
                ("P6_BBSE_SOFT", s_bb_soft, tau_common, bb["pi_bbse_soft"], ll_bb_soft),
                ("P6_BBSE_HARD", s_bb_hard, tau_common, bb["pi_bbse_hard"], ll_bb_hard),
                ("P5_ORACLE_PRIOR", s_oracle, tau_common, pi_true, ll_oracle),
                ("P1_FULL_ORACLE_CUTOFF", s_tgt, tau_p1_full, pi_true, ll_raw),
                ("P2_FULL_CP_CUTOFF", s_tgt, tau_p2_full if cert_full else float("nan"), pi_true, ll_raw),
            ]

            for pol_name, sc_vec, cut_val, pi_est, ll_val in policy_defs:
                ev = eval_at_cutoff(y_tgt, sc_vec, cut_val)
                policy_rows.append(
                    {
                        "target": target,
                        "model": m_name,
                        "route": route,
                        "alpha": alpha,
                        "policy": pol_name,
                        "pi_source": pi_src,
                        "pi_target_true": pi_true,
                        "pi_estimated": pi_est,
                        "em_iters": em_iters if "EM" in pol_name else 0,
                        "cutoff": cut_val,
                        "log_loss": ll_val,
                        "brier": float(brier_score_loss(y_tgt, np.clip(sc_vec, 0.0, 1.0))) if math.isfinite(cut_val) else float("nan"),
                        "exceeds_alpha": bool(ev["fpr"] > alpha) if math.isfinite(ev["fpr"]) else False,
                        **ev,
                    }
                )

            # 2,000-rep paired bootstrap CI on primary models A1 and A2 at alpha = 0.01 for external streams
            if m_name in ("A1", "A2") and route == "final" and alpha == 0.01 and target in ("spaphish", "ceas08"):
                for cname, sc_b in (("P5_SAERENS_EM", s_em), ("P6_BBSE_SOFT", s_bb_soft), ("P5_ORACLE_PRIOR", s_oracle)):
                    lo_d, hi_d = paired_bootstrap_fpr_diff(y_tgt, s_tgt, tau_common, sc_b, tau_common, reps=2000, seed=20261112)
                    ev_a = eval_at_cutoff(y_tgt, s_tgt, tau_common)
                    ev_b = eval_at_cutoff(y_tgt, sc_b, tau_common)
                    bootstrap_contrasts.append(
                        {
                            "target": target,
                            "model": m_name,
                            "route": route,
                            "alpha": alpha,
                            "contrast": f"{cname}_minus_P0_COMMON",
                            "fpr_p0": ev_a["fpr"],
                            "fpr_policy": ev_b["fpr"],
                            "delta_fpr": ev_b["fpr"] - ev_a["fpr"],
                            "delta_fpr_ci95_lo": lo_d,
                            "delta_fpr_ci95_hi": hi_d,
                            "recall_p0": ev_a["recall"],
                            "recall_policy": ev_b["recall"],
                        }
                    )

    df_prior = pd.DataFrame(policy_rows)
    df_prior.to_csv(RESULTS_DIR / "exp1_prior_shift_table.csv", index=False)
    df_boot = pd.DataFrame(bootstrap_contrasts)
    df_boot.to_csv(RESULTS_DIR / "exp1_prior_shift_bootstrap_contrasts.csv", index=False)

    # Summary JSON receipt
    a2_pairs_01 = df_reciprocal[(df_reciprocal["pair_id"].str.startswith("A2_pair_")) & (df_reciprocal["alpha"] == 0.01)]
    spa_a2_01 = df_prior[
        (df_prior["target"] == "spaphish")
        & (df_prior["model"] == "A2")
        & (df_prior["route"] == "final")
        & (df_prior["alpha"] == 0.01)
    ].set_index("policy")
    spa_a1_01 = df_prior[
        (df_prior["target"] == "spaphish")
        & (df_prior["model"] == "A1")
        & (df_prior["route"] == "final")
        & (df_prior["alpha"] == 0.01)
    ].set_index("policy")

    receipt = {
        "schema": "prior_shift_and_seed_ceilings_receipt",
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - t0, 2),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "parity_max_abs_diff": max_parity_diff,
        "parity_diffs": parity_diffs,
        "key_findings": {
            "a2_ensemble_reciprocal_0.01": df_reciprocal[
                (df_reciprocal["pair_id"] == "A2_ensemble") & (df_reciprocal["alpha"] == 0.01)
            ].iloc[0].to_dict(),
            "a1_ensemble_reciprocal_0.01": df_reciprocal[
                (df_reciprocal["pair_id"] == "A1_ensemble") & (df_reciprocal["alpha"] == 0.01)
            ].iloc[0].to_dict(),
            "a2_9_seed_pairs_feasible_counts_at_0.01": a2_pairs_01[["pair_id", "n_candidates", "n_feasible", "tau_own_val_trec6", "tau_own_val_trec5", "max_score_val_trec6", "max_score_val_trec5"]].to_dict("records"),
            "spaphish_final_a2_alpha_0.01": {
                pol: {"fpr": float(spa_a2_01.loc[pol, "fpr"]), "recall": float(spa_a2_01.loc[pol, "recall"]), "pi_estimated": float(spa_a2_01.loc[pol, "pi_estimated"])}
                for pol in ("P0_COMMON", "P3_MARGINAL_QUANTILE", "P3b_EM_PRIOR_QUANTILE", "P3b_ORACLE_PRIOR_QUANTILE", "P4_PLATT_DEV", "P5_SAERENS_EM", "P6_BBSE_SOFT", "P6_BBSE_HARD", "P5_ORACLE_PRIOR", "P1_FULL_ORACLE_CUTOFF", "P2_FULL_CP_CUTOFF")
            },
            "spaphish_final_a1_alpha_0.01": {
                pol: {"fpr": float(spa_a1_01.loc[pol, "fpr"]), "recall": float(spa_a1_01.loc[pol, "recall"]), "pi_estimated": float(spa_a1_01.loc[pol, "pi_estimated"])}
                for pol in ("P0_COMMON", "P3_MARGINAL_QUANTILE", "P3b_EM_PRIOR_QUANTILE", "P3b_ORACLE_PRIOR_QUANTILE", "P4_PLATT_DEV", "P5_SAERENS_EM", "P6_BBSE_SOFT", "P6_BBSE_HARD", "P5_ORACLE_PRIOR", "P1_FULL_ORACLE_CUTOFF", "P2_FULL_CP_CUTOFF")
            },
        },
        "outputs": {
            "exp1_prior_shift_table.csv": sha256_file(RESULTS_DIR / "exp1_prior_shift_table.csv"),
            "exp1_seed_ceiling_audit.csv": sha256_file(RESULTS_DIR / "exp1_seed_ceiling_audit.csv"),
            "exp1_reciprocal_feasibility.csv": sha256_file(RESULTS_DIR / "exp1_reciprocal_feasibility.csv"),
            "exp1_prior_shift_bootstrap_contrasts.csv": sha256_file(RESULTS_DIR / "exp1_prior_shift_bootstrap_contrasts.csv"),
        },
    }
    (RUNS_DIR / "prior_shift_and_seed_ceilings.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt["key_findings"], indent=2), flush=True)


if __name__ == "__main__":
    main()
