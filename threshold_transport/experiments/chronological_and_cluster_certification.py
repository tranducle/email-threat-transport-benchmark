#!/usr/bin/env python3
"""Forward Chronological Calibration Transfer (SpaPhish), 10,000-Draw Exchangeable E1 Curves,
and 2,000-Replicate 16,729-Domain Cluster Bootstrap on CEAS-08.

Reads the per-family score parquets written by multifamily_training_and_probes.py and writes
the exp3_*.csv tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, hypergeom
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("THRESHOLD_TRANSPORT_DATA", REPO_ROOT / "data"))

SPA_DATA = DATA_ROOT / "processed" / "spaphish_confirmation_urlfixed.parquet"
CEAS_DATA = DATA_ROOT / "processed" / "ceas08_confirmation.parquet"
CEAS_RAW_CSV = DATA_ROOT / "raw" / "CEAS_08.csv"

PRED_DIR = DATA_ROOT / "scores"
RESULTS_DIR = DATA_ROOT / "results"
RUNS_DIR = DATA_ROOT / "results" / "receipts"

ALPHAS = (0.005, 0.01, 0.02)
K_GRID = (50, 100, 200, 299, 400, 438, 500, 680, 1000, 2000)
MIN_REMAIN = 395
N_DRAWS_E1 = 10000
N_BOOT_CEAS = 2000
RNG_SEED = 20261112
PROB_EPS = 1e-12
FAMILIES = ("A1", "A1u", "A2", "A3a_PhishKAN", "A3b_MLP", "A3c_HistGBDT")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def threshold_at_max_fpr(y: np.ndarray, scores: np.ndarray, alpha: float) -> float:
    neg = np.sort(np.asarray(scores, dtype=np.float64)[np.asarray(y, dtype=int) == 0])[::-1]
    allowed = int(math.floor(alpha * len(neg)))
    if allowed >= len(neg):
        return float(np.min(neg))
    return float(np.nextafter(neg[allowed], np.inf))


def cp_allowance_table(alpha: float, nmax: int, conf: float = 0.95) -> np.ndarray:
    out = np.full(nmax + 1, -1, dtype=np.int64)
    for n0 in range(1, nmax + 1):
        x = np.arange(0, n0)
        u = beta.ppf(conf, x + 1, n0 - x)
        ok = np.flatnonzero(u <= alpha)
        out[n0] = int(ok.max()) if ok.size else -1
    return out


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
    n0 = int((y_arr == 0).sum())
    n1 = int((y_arr == 1).sum())
    if not math.isfinite(cutoff):
        return {
            "fpr": float("nan"),
            "fpr_ci_lo": float("nan"),
            "fpr_ci_hi": float("nan"),
            "recall": float("nan"),
            "recall_ci_lo": float("nan"),
            "recall_ci_hi": float("nan"),
            "fp": 0,
            "tp": 0,
            "n_neg": n0,
            "n_pos": n1,
        }
    flag = s_arr >= cutoff
    fp = int(flag[y_arr == 0].sum())
    tp = int(flag[y_arr == 1].sum())
    fpr_lo, fpr_hi = clopper_pearson_two_sided(fp, n0)
    rec_lo, rec_hi = clopper_pearson_two_sided(tp, n1)
    return {
        "fpr": fp / max(1, n0),
        "fpr_ci_lo": fpr_lo,
        "fpr_ci_hi": fpr_hi,
        "recall": tp / max(1, n1),
        "recall_ci_lo": rec_lo,
        "recall_ci_hi": rec_hi,
        "fp": fp,
        "tp": tp,
        "n_neg": n0,
        "n_pos": n1,
    }


def saerens_adjust_scores(s: np.ndarray, pi_src: float, pi_tgt: float) -> np.ndarray:
    s_clip = np.clip(np.asarray(s, dtype=np.float64), PROB_EPS, 1.0 - PROB_EPS)
    pi_s = float(np.clip(pi_src, 1e-6, 1.0 - 1e-6))
    pi_t = float(np.clip(pi_tgt, 1e-6, 1.0 - 1e-6))
    num = (pi_t / pi_s) * s_clip
    den = num + ((1.0 - pi_t) / (1.0 - pi_s)) * (1.0 - s_clip)
    return num / den


def saerens_em_prior(s_target: np.ndarray, pi_src: float, max_iter: int = 500, tol: float = 1e-9) -> float:
    pi_t = float(pi_src)
    for _ in range(max_iter):
        post = saerens_adjust_scores(s_target, pi_src, pi_t)
        pi_next = float(np.clip(post.mean(), 1e-6, 1.0 - 1e-6))
        if abs(pi_next - pi_t) < tol:
            return pi_next
        pi_t = pi_next
    return pi_t


def bbse_soft_prior(y_src: np.ndarray, s_src: np.ndarray, s_tgt: np.ndarray) -> float:
    mu1 = float(s_src[y_src == 1].mean())
    mu0 = float(s_src[y_src == 0].mean())
    mu_t = float(s_tgt.mean())
    denom = mu1 - mu0
    return float(np.clip((mu_t - mu0) / denom, 1e-6, 1.0 - 1e-6)) if abs(denom) > 1e-6 else float(y_src.mean())


def p3_rank_map(target_scores: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    st = np.sort(target_scores)
    u = np.searchsorted(st, target_scores, side="right") / len(st)
    return np.quantile(ref_sorted, u, method="inverted_cdf")


def holm_bonferroni(p_vals: list[float]) -> list[float]:
    m = len(p_vals)
    order = np.argsort(p_vals)
    adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, p_vals[idx] * (m - rank))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


def extract_ceas08_domains(ceas_clean: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    raw = pd.read_csv(CEAS_RAW_CSV, usecols=["sender"])
    aligned_sender = raw.iloc[ceas_clean["original_index"].to_numpy(int)]["sender"].fillna("").astype(str).to_numpy()
    dom_re = re.compile(r"@([A-Za-z0-9.-]+)")

    # Primary grouping: 16,729 unique clusters (925 missing-domain rows in 1 combined cluster)
    dom_combined = []
    # Sensitivity grouping: 925 missing-domain rows treated as 925 independent singletons
    dom_singleton = []
    missing_count = 0
    for idx_i, val in enumerate(aligned_sender):
        m = dom_re.search(val)
        if m:
            d = m.group(1).lower()
            dom_combined.append(d)
            dom_singleton.append(d)
        else:
            missing_count += 1
            dom_combined.append("__missing_domain__")
            dom_singleton.append(f"__missing_singleton_{idx_i}__")

    arr_comb = np.asarray(dom_combined, dtype=object)
    arr_sing = np.asarray(dom_singleton, dtype=object)
    u_comb, inv_comb = np.unique(arr_comb, return_inverse=True)
    u_sing, inv_sing = np.unique(arr_sing, return_inverse=True)

    top5 = {str(k): int(v) for k, v in pd.Series(arr_comb).value_counts().head(5).items()}
    info = {
        "n_clean_rows": int(len(ceas_clean)),
        "n_missing_sender_domain_rows": missing_count,
        "n_clusters_combined_missing": int(len(u_comb)),
        "n_clusters_singleton_missing": int(len(u_sing)),
        "top5_domain_counts": top5,
    }
    return inv_comb, inv_sing, info


def weighted_metrics_pack(
    y: np.ndarray, scores_map: dict[str, np.ndarray], cutoffs_map: dict[str, float], weights: np.ndarray
) -> dict[str, dict[str, float]]:
    w = weights.astype(np.float64)
    neg_mask = y == 0
    pos_mask = y == 1
    w_neg_sum = float(w[neg_mask].sum())
    w_pos_sum = float(w[pos_mask].sum())
    out: dict[str, dict[str, float]] = {}
    for fam, s in scores_map.items():
        tau = cutoffs_map[fam]
        flag = (s >= tau).astype(np.float64)
        out[fam] = {
            "auroc": float(roc_auc_score(y, s, sample_weight=w)),
            "auprc": float(average_precision_score(y, s, sample_weight=w)),
            "log_loss": float(log_loss(y, np.clip(s, 1e-7, 1.0 - 1e-7), sample_weight=w)),
            "brier": float(brier_score_loss(y, s, sample_weight=w)),
            "fpr": float((w[neg_mask] * flag[neg_mask]).sum() / max(1.0, w_neg_sum)),
            "recall": float((w[pos_mask] * flag[pos_mask]).sum() / max(1.0, w_pos_sum)),
        }
    return out


def run_smoke() -> None:
    # Verify exact mathematical identity between cluster index concatenation and multinomial sample_weight
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=500)
    s = rng.uniform(0.01, 0.99, size=500)
    clusters = rng.integers(0, 60, size=500)
    u_c, inv_c = np.unique(clusters, return_inverse=True)
    g = len(u_c)
    picked = rng.integers(0, g, size=g)
    idx_concat = np.concatenate([np.flatnonzero(inv_c == p) for p in picked])
    counts = np.bincount(picked, minlength=g)
    row_weights = counts[inv_c]
    mask_nz = row_weights > 0

    auroc_concat = float(roc_auc_score(y[idx_concat], s[idx_concat]))
    auroc_weight = float(roc_auc_score(y[mask_nz], s[mask_nz], sample_weight=row_weights[mask_nz]))
    ap_concat = float(average_precision_score(y[idx_concat], s[idx_concat]))
    ap_weight = float(average_precision_score(y[mask_nz], s[mask_nz], sample_weight=row_weights[mask_nz]))
    ll_concat = float(log_loss(y[idx_concat], s[idx_concat]))
    ll_weight = float(log_loss(y[mask_nz], s[mask_nz], sample_weight=row_weights[mask_nz]))
    max_diff = max(abs(auroc_concat - auroc_weight), abs(ap_concat - ap_weight), abs(ll_concat - ll_weight))
    assert max_diff < 1e-13, f"Weighted cluster equivalence failed: {max_diff}"

    ceas_full = pd.read_parquet(CEAS_DATA)
    ceas_clean = ceas_full.loc[~ceas_full["contamination_removed"].astype(bool)].reset_index(drop=True)
    _, _, info = extract_ceas08_domains(ceas_clean)
    print(json.dumps({"smoke_status": "PASS", "weighted_vs_concat_max_diff": max_diff, "ceas08_domain_info": info}, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Verify cluster weight equivalence and CEAS-08 domain counts")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return

    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all 6 model families' frozen predictions produced by multifamily_training_and_probes.py
    p_t6 = pd.read_parquet(PRED_DIR / "exp2_all_families_trec6.parquet")
    p_t5 = pd.read_parquet(PRED_DIR / "exp2_all_families_trec5.parquet")
    p_spa = pd.read_parquet(PRED_DIR / "exp2_all_families_spaphish.parquet")
    p_ceas = pd.read_parquet(PRED_DIR / "exp2_all_families_ceas08.parquet")

    spa_meta = pd.read_parquet(SPA_DATA, columns=["row_id", "label", "temporal_group"])
    ceas_full = pd.read_parquet(CEAS_DATA)
    ceas_clean = ceas_full.loc[~ceas_full["contamination_removed"].astype(bool)].reset_index(drop=True)
    if not p_spa["row_id"].equals(spa_meta["row_id"]) or not p_ceas["row_id"].equals(ceas_clean["row_id"]):
        raise RuntimeError("Row alignment mismatch")

    y_t6 = p_t6["label"].to_numpy(int)
    y_t5 = p_t5["label"].to_numpy(int)
    y_spa = p_spa["label"].to_numpy(int)
    y_ceas = p_ceas["label"].to_numpy(int)

    # Development references & cutoffs per family and alpha
    dev_ref: dict[str, dict] = {}
    for fam in FAMILIES:
        s6 = p_t6[f"fit_trec5_{fam}"].to_numpy(np.float64)
        s5 = p_t5[f"fit_trec6_{fam}"].to_numpy(np.float64)
        pooled_s = np.concatenate([s5, s6])
        pooled_y = np.concatenate([y_t5, y_t6])
        cuts = {}
        for alpha in ALPHAS:
            c_fit5 = threshold_at_max_fpr(y_t6, s6, alpha)
            c_fit6 = threshold_at_max_fpr(y_t5, s5, alpha)
            cuts[alpha] = {"fit_trec5": c_fit5, "fit_trec6": c_fit6, "final": max(c_fit5, c_fit6), "common": max(c_fit5, c_fit6)}
        dev_ref[fam] = {
            "s6": s6,
            "s5": s5,
            "pooled_s": pooled_s,
            "pooled_y": pooled_y,
            "pooled_sorted": np.sort(pooled_s),
            "cuts": cuts,
        }

    # =========================================================================
    # PART 1: Forward Chronological Calibration on SpaPhish (pre_2023 -> 2023_to_2025)
    # + 24 undated sensitivity partitions
    # =========================================================================
    print("[chronological] Running SpaPhish forward chronological calibration and 24-undated sensitivity...", flush=True)
    tg = spa_meta["temporal_group"].astype(str).to_numpy()
    m_pre = tg == "pre_2023"
    m_post = tg == "2023_to_2025"
    m_undated = tg == "date_missing"

    chrono_splits = [
        ("PRIMARY_DATED_ONLY", m_pre, m_post),
        ("SENS_UNDATED_IN_POST", m_pre, m_post | m_undated),
        ("SENS_UNDATED_IN_PRE", m_pre | m_undated, m_post),
    ]

    pi_train_routes = {
        "fit_trec5": float(y_t5.mean()),
        "fit_trec6": float(y_t6.mean()),
        "final": float(np.concatenate([y_t5, y_t6]).mean()),
    }

    chrono_rows = []
    for split_name, cal_mask, eval_mask in chrono_splits:
        y_cal = y_spa[cal_mask]
        y_ev = y_spa[eval_mask]
        pi_cal_true = float(y_cal.mean())
        pi_ev_true = float(y_ev.mean())

        for fam in FAMILIES:
            for route in ("final", "fit_trec5", "fit_trec6"):
                s_all = p_spa[f"{route}_{fam}"].to_numpy(np.float64)
                s_cal = s_all[cal_mask]
                s_ev = s_all[eval_mask]
                pi_src = pi_train_routes[route]

                # Prior-shift adaptations estimated unlabeled on the future stream s_ev (or using pre_2023 prior!)
                pi_em_ev = saerens_em_prior(s_ev, pi_src)
                pi_bb_ev = bbse_soft_prior(dev_ref[fam]["pooled_y"], dev_ref[fam]["pooled_s"], s_ev)
                s_ev_p3 = p3_rank_map(s_ev, dev_ref[fam]["pooled_sorted"])
                s_ev_em = saerens_adjust_scores(s_ev, pi_src, pi_em_ev)
                s_ev_bb = saerens_adjust_scores(s_ev, pi_src, pi_bb_ev)
                s_ev_pre_prior = saerens_adjust_scores(s_ev, pi_src, pi_cal_true)
                s_ev_oracle_post = saerens_adjust_scores(s_ev, pi_src, pi_ev_true)

                for alpha in ALPHAS:
                    tau_com = dev_ref[fam]["cuts"][alpha]["common"]
                    tau_own = dev_ref[fam]["cuts"][alpha][route]
                    tau_p1_pre = threshold_at_max_fpr(y_cal, s_cal, alpha)
                    cert_p2_pre, tau_p2_pre = cp_cutoff(s_cal[y_cal == 0], alpha, conf=0.95)
                    tau_oracle_post_cut = threshold_at_max_fpr(y_ev, s_ev, alpha)

                    pol_specs = [
                        ("P0_COMMON", s_ev, tau_com, True),
                        ("P0_OWN", s_ev, tau_own, True),
                        ("P1_CHRONO_PRE2023", s_ev, tau_p1_pre, True),
                        ("P2_CHRONO_CP_PRE2023", s_ev, tau_p2_pre if cert_p2_pre else float("nan"), cert_p2_pre),
                        ("P3_MARGINAL_QUANTILE", s_ev_p3, tau_com, True),
                        ("P5_SAERENS_EM_POST", s_ev_em, tau_com, True),
                        ("P6_BBSE_SOFT_POST", s_ev_bb, tau_com, True),
                        ("P5_HIST_PRE2023_PRIOR", s_ev_pre_prior, tau_com, True),
                        ("P5_ORACLE_POST_PRIOR", s_ev_oracle_post, tau_com, True),
                        ("ORACLE_IN_SAMPLE_POST_CUTOFF", s_ev, tau_oracle_post_cut, True),
                    ]
                    for pol_name, sc_vec, cut_val, is_cert in pol_specs:
                        ev = eval_at_cutoff(y_ev, sc_vec, cut_val)
                        chrono_rows.append(
                            {
                                "split_regime": split_name,
                                "n_cal": int(len(y_cal)),
                                "n0_cal": int((y_cal == 0).sum()),
                                "n1_cal": int((y_cal == 1).sum()),
                                "n_eval": int(len(y_ev)),
                                "n0_eval": int((y_ev == 0).sum()),
                                "n1_eval": int((y_ev == 1).sum()),
                                "model": fam,
                                "route": route,
                                "alpha": alpha,
                                "policy": pol_name,
                                "certifiable": bool(is_cert),
                                "cutoff": cut_val,
                                **ev,
                            }
                        )
    df_chrono = pd.DataFrame(chrono_rows)
    df_chrono.to_csv(RESULTS_DIR / "exp3_spaphish_chronological.csv", index=False)

    # =========================================================================
    # PART 2: 10,000-Draw Exchangeable E1 Curves across all 6 families & k grid
    # =========================================================================
    print(f"[chronological] Running {N_DRAWS_E1}-draw exchangeable E1 curves across 4 corpora and 6 families...", flush=True)
    cp_tables = {alpha: cp_allowance_table(alpha, max(K_GRID)) for alpha in ALPHAS}
    e1_targets = [
        ("trec6", y_t6, {fam: p_t6[f"fit_trec5_{fam}"].to_numpy(np.float64) for fam in FAMILIES}, "fit_trec5"),
        ("trec5", y_t5, {fam: p_t5[f"fit_trec6_{fam}"].to_numpy(np.float64) for fam in FAMILIES}, "fit_trec6"),
        ("spaphish", y_spa, {fam: p_spa[f"final_{fam}"].to_numpy(np.float64) for fam in FAMILIES}, "final"),
        ("ceas08", y_ceas, {fam: p_ceas[f"final_{fam}"].to_numpy(np.float64) for fam in FAMILIES}, "final"),
    ]

    e1_rows = []
    for t_idx, (tname, y_vec, fam_scores, rname) in enumerate(e1_targets):
        n_tot = len(y_vec)
        n0_tot = int((y_vec == 0).sum())
        n1_tot = int((y_vec == 1).sum())
        valid_ks = [k for k in K_GRID if n_tot - k >= MIN_REMAIN]

        # Pre-sort full negative and positive score arrays per family for O(log N) remainder evaluation
        sorted_neg = {fam: np.sort(sc[y_vec == 0]) for fam, sc in fam_scores.items()}
        sorted_pos = {fam: np.sort(sc[y_vec == 1]) for fam, sc in fam_scores.items()}

        for k in valid_ks:
            rng = np.random.default_rng(np.random.SeedSequence(RNG_SEED, spawn_key=(t_idx, k)))
            # Generate 10,000 SRS index draws of size k without replacement
            draws = np.empty((N_DRAWS_E1, k), dtype=np.int32)
            for r in range(N_DRAWS_E1):
                draws[r] = rng.choice(n_tot, size=k, replace=False)

            y_lab_mat = y_vec[draws]  # (R, k)
            neg_mask_mat = y_lab_mat == 0
            n0_lab = neg_mask_mat.sum(axis=1)  # (R,)
            n1_lab = k - n0_lab
            n0_rem = n0_tot - n0_lab
            n1_rem = n1_tot - n1_lab

            for fam in FAMILIES:
                sc = fam_scores[fam]
                sc_draws = sc[draws]  # (R, k)
                # Replace positive positions with -inf so descending sort puts labeled negatives first!
                neg_padded = np.where(neg_mask_mat, sc_draws, -np.inf)
                neg_desc_mat = np.sort(neg_padded, axis=1)[:, ::-1]  # (R, k)
                # Minimum labeled negative per draw (for allowed >= n0_lab edge case)
                neg_min = np.where(neg_mask_mat, sc_draws, np.inf).min(axis=1)

                s_neg_sorted = sorted_neg[fam]
                s_pos_sorted = sorted_pos[fam]

                for alpha in ALPHAS:
                    # 1. P0_COMMON on unlabeled remainder
                    tau_p0 = dev_ref[fam]["cuts"][alpha]["common"]
                    tot_fp_p0 = int((sc[y_vec == 0] >= tau_p0).sum())
                    tot_tp_p0 = int((sc[y_vec == 1] >= tau_p0).sum())
                    lab_fp_p0 = (neg_mask_mat & (sc_draws >= tau_p0)).sum(axis=1)
                    lab_tp_p0 = ((~neg_mask_mat) & (sc_draws >= tau_p0)).sum(axis=1)
                    fpr_p0 = (tot_fp_p0 - lab_fp_p0) / np.maximum(1, n0_rem)
                    rec_p0 = (tot_tp_p0 - lab_tp_p0) / np.maximum(1, n1_rem)

                    e1_rows.append(
                        {
                            "target": tname,
                            "model": fam,
                            "route": rname,
                            "policy": "P0_COMMON",
                            "alpha": alpha,
                            "k": k,
                            "draws": N_DRAWS_E1,
                            "certifiable_share": 1.0,
                            "analytic_p_enough_neg": 1.0,
                            "violation_prob": float((fpr_p0 > alpha).mean()),
                            "fpr_mean": float(fpr_p0.mean()),
                            "fpr_median": float(np.median(fpr_p0)),
                            "fpr_p05": float(np.quantile(fpr_p0, 0.05)),
                            "fpr_p95": float(np.quantile(fpr_p0, 0.95)),
                            "recall_mean": float(rec_p0.mean()),
                            "recall_median": float(np.median(rec_p0)),
                            "recall_p05": float(np.quantile(rec_p0, 0.05)),
                            "recall_p95": float(np.quantile(rec_p0, 0.95)),
                        }
                    )

                    # 2. P1 empirical target quantile on labeled negatives
                    has_neg = n0_lab > 0
                    allowed_p1 = np.floor(alpha * n0_lab).astype(np.int64)
                    safe_idx = np.clip(allowed_p1, 0, k - 1)
                    raw_cut_p1 = np.nextafter(neg_desc_mat[np.arange(N_DRAWS_E1), safe_idx], np.inf)
                    cut_p1 = np.where(allowed_p1 >= n0_lab, neg_min, raw_cut_p1)

                    v_idx = np.flatnonzero(has_neg)
                    cuts_v = cut_p1[v_idx]
                    tot_fp_v = n0_tot - np.searchsorted(s_neg_sorted, cuts_v, side="left")
                    tot_tp_v = n1_tot - np.searchsorted(s_pos_sorted, cuts_v, side="left")
                    lab_fp_v = (neg_mask_mat[v_idx] & (sc_draws[v_idx] >= cuts_v[:, None])).sum(axis=1)
                    lab_tp_v = ((~neg_mask_mat[v_idx]) & (sc_draws[v_idx] >= cuts_v[:, None])).sum(axis=1)
                    fpr_p1 = (tot_fp_v - lab_fp_v) / np.maximum(1, n0_rem[v_idx])
                    rec_p1 = (tot_tp_v - lab_tp_v) / np.maximum(1, n1_rem[v_idx])

                    e1_rows.append(
                        {
                            "target": tname,
                            "model": fam,
                            "route": rname,
                            "policy": "P1_EMPIRICAL",
                            "alpha": alpha,
                            "k": k,
                            "draws": N_DRAWS_E1,
                            "certifiable_share": float(has_neg.mean()),
                            "analytic_p_enough_neg": float(1.0 - hypergeom(n_tot, n0_tot, k).pmf(0)),
                            "violation_prob": float((fpr_p1 > alpha).mean()),
                            "fpr_mean": float(fpr_p1.mean()),
                            "fpr_median": float(np.median(fpr_p1)),
                            "fpr_p05": float(np.quantile(fpr_p1, 0.05)),
                            "fpr_p95": float(np.quantile(fpr_p1, 0.95)),
                            "recall_mean": float(rec_p1.mean()),
                            "recall_median": float(np.median(rec_p1)),
                            "recall_p05": float(np.quantile(rec_p1, 0.05)),
                            "recall_p95": float(np.quantile(rec_p1, 0.95)),
                        }
                    )

                    # 3. P2 Clopper-Pearson 95% certified target cutoff
                    n0_req = int(math.ceil(math.log(0.05) / math.log(1.0 - alpha)))
                    analytic_p2 = float(hypergeom(n_tot, n0_tot, k).sf(n0_req - 1)) if n0_tot >= n0_req else 0.0
                    allowed_p2 = cp_tables[alpha][n0_lab]
                    cert_mask = allowed_p2 >= 0
                    c_idx = np.flatnonzero(cert_mask)
                    if len(c_idx) > 0:
                        s_idx2 = np.clip(allowed_p2[c_idx], 0, k - 1)
                        cuts_c = np.nextafter(neg_desc_mat[c_idx, s_idx2], np.inf)
                        tot_fp_c = n0_tot - np.searchsorted(s_neg_sorted, cuts_c, side="left")
                        tot_tp_c = n1_tot - np.searchsorted(s_pos_sorted, cuts_c, side="left")
                        lab_fp_c = (neg_mask_mat[c_idx] & (sc_draws[c_idx] >= cuts_c[:, None])).sum(axis=1)
                        lab_tp_c = ((~neg_mask_mat[c_idx]) & (sc_draws[c_idx] >= cuts_c[:, None])).sum(axis=1)
                        fpr_p2 = (tot_fp_c - lab_fp_c) / np.maximum(1, n0_rem[c_idx])
                        rec_p2 = (tot_tp_c - lab_tp_c) / np.maximum(1, n1_rem[c_idx])
                        e1_rows.append(
                            {
                                "target": tname,
                                "model": fam,
                                "route": rname,
                                "policy": "P2_CERTIFIED_CP",
                                "alpha": alpha,
                                "k": k,
                                "draws": N_DRAWS_E1,
                                "certifiable_share": float(cert_mask.mean()),
                                "analytic_p_enough_neg": analytic_p2,
                                "violation_prob": float((fpr_p2 > alpha).mean()),
                                "fpr_mean": float(fpr_p2.mean()),
                                "fpr_median": float(np.median(fpr_p2)),
                                "fpr_p05": float(np.quantile(fpr_p2, 0.05)),
                                "fpr_p95": float(np.quantile(fpr_p2, 0.95)),
                                "recall_mean": float(rec_p2.mean()),
                                "recall_median": float(np.median(rec_p2)),
                                "recall_p05": float(np.quantile(rec_p2, 0.05)),
                                "recall_p95": float(np.quantile(rec_p2, 0.95)),
                            }
                        )
                    else:
                        e1_rows.append(
                            {
                                "target": tname,
                                "model": fam,
                                "route": rname,
                                "policy": "P2_CERTIFIED_CP",
                                "alpha": alpha,
                                "k": k,
                                "draws": N_DRAWS_E1,
                                "certifiable_share": 0.0,
                                "analytic_p_enough_neg": analytic_p2,
                                "violation_prob": float("nan"),
                                "fpr_mean": float("nan"),
                                "fpr_median": float("nan"),
                                "fpr_p05": float("nan"),
                                "fpr_p95": float("nan"),
                                "recall_mean": float("nan"),
                                "recall_median": float("nan"),
                                "recall_p05": float("nan"),
                                "recall_p95": float("nan"),
                            }
                        )

    df_e1 = pd.DataFrame(e1_rows)
    df_e1.to_csv(RESULTS_DIR / "exp3_exchangeable_e1_summary.csv", index=False)

    # =========================================================================
    # PART 3: 2,000-Replicate 16,729-Domain Cluster Bootstrap vs Row-Level Bootstrap on CEAS-08
    # + 925-missing-domain singleton sensitivity + Holm-Bonferroni FWER <= 0.05
    # =========================================================================
    print(f"[chronological] Running {N_BOOT_CEAS}-replicate CEAS-08 sender-domain cluster & row bootstrap...", flush=True)
    inv_comb, inv_sing, dom_info = extract_ceas08_domains(ceas_clean)
    n_ceas = len(y_ceas)
    g_comb = dom_info["n_clusters_combined_missing"]
    g_sing = dom_info["n_clusters_singleton_missing"]

    ceas_scores_final = {fam: p_ceas[f"final_{fam}"].to_numpy(np.float64) for fam in FAMILIES}
    ceas_cuts_01 = {fam: dev_ref[fam]["cuts"][0.01]["common"] for fam in FAMILIES}

    # Also include route-specific A2 vs A1 contrasts (fit_trec5, fit_trec6) to verify route reversals under clustering!
    for r_extra in ("fit_trec5", "fit_trec6"):
        for fam_ex in ("A1", "A1u", "A2", "A3a_PhishKAN"):
            ceas_scores_final[f"{r_extra}_{fam_ex}"] = p_ceas[f"{r_extra}_{fam_ex}"].to_numpy(np.float64)
            ceas_cuts_01[f"{r_extra}_{fam_ex}"] = dev_ref[fam_ex]["cuts"][0.01]["common"]

    point_est = weighted_metrics_pack(y_ceas, ceas_scores_final, ceas_cuts_01, np.ones(n_ceas, dtype=np.float64))

    contrasts_to_track = [
        ("final", "A2", "A1"),
        ("final", "A2", "A1u"),
        ("final", "A2", "A3a_PhishKAN"),
        ("final", "A2", "A3b_MLP"),
        ("final", "A2", "A3c_HistGBDT"),
        ("final", "A1u", "A1"),
        ("final", "A3a_PhishKAN", "A3b_MLP"),
        ("fit_trec5", "fit_trec5_A2", "fit_trec5_A1"),
        ("fit_trec6", "fit_trec6_A2", "fit_trec6_A1"),
    ]
    metric_names = ("auroc", "auprc", "log_loss", "brier", "fpr", "recall")

    def run_bootstrap_scheme(scheme_name: str, mode: str) -> list[dict]:
        rng = np.random.default_rng(RNG_SEED)
        diffs: dict[tuple[str, str, str, str], list[float]] = {
            (rt, fa, fb, mn): [] for (rt, fa, fb) in contrasts_to_track for mn in metric_names
        }
        for _ in range(N_BOOT_CEAS):
            if mode == "row_paired":
                picked_rows = rng.integers(0, n_ceas, size=n_ceas)
                w_row = np.bincount(picked_rows, minlength=n_ceas)
            elif mode == "cluster_16729_combined_missing":
                picked_c = rng.integers(0, g_comb, size=g_comb)
                c_weights = np.bincount(picked_c, minlength=g_comb)
                w_row = c_weights[inv_comb]
            elif mode == "cluster_17653_singleton_missing":
                picked_c = rng.integers(0, g_sing, size=g_sing)
                c_weights = np.bincount(picked_c, minlength=g_sing)
                w_row = c_weights[inv_sing]
            else:
                raise ValueError(mode)

            nz = w_row > 0
            yb = y_ceas[nz]
            if yb.min() == yb.max():
                continue
            sub_scores = {k: v[nz] for k, v in ceas_scores_final.items()}
            mp = weighted_metrics_pack(yb, sub_scores, ceas_cuts_01, w_row[nz])
            for rt, fa, fb in contrasts_to_track:
                for mn in metric_names:
                    diffs[(rt, fa, fb, mn)].append(mp[fa][mn] - mp[fb][mn])

        rows_out = []
        for rt, fa, fb in contrasts_to_track:
            for mn in metric_names:
                arr = np.asarray(diffs[(rt, fa, fb, mn)], dtype=np.float64)
                pt = point_est[fa][mn] - point_est[fb][mn]
                lo = float(np.quantile(arr, 0.025))
                hi = float(np.quantile(arr, 0.975))
                p_raw = float(2.0 * min((arr <= 0.0).mean(), (arr >= 0.0).mean()))
                p_raw = max(p_raw, 1.0 / len(arr))
                rows_out.append(
                    {
                        "bootstrap_scheme": scheme_name,
                        "n_replicates": len(arr),
                        "route": rt,
                        "model_a": fa,
                        "model_b": fb,
                        "contrast": f"{fa}_minus_{fb}",
                        "metric": mn,
                        "point_model_a": point_est[fa][mn],
                        "point_model_b": point_est[fb][mn],
                        "delta_point": pt,
                        "boot_se": float(arr.std(ddof=1)),
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                        "ci95_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                        "p_two_sided_raw": p_raw,
                    }
                )
        # Apply Holm-Bonferroni across the 7 final-route primary family contrasts per metric
        for mn in metric_names:
            idxs = [i for i, r in enumerate(rows_out) if r["route"] == "final" and r["metric"] == mn]
            adj_p = holm_bonferroni([rows_out[i]["p_two_sided_raw"] for i in idxs])
            for i, hp in zip(idxs, adj_p):
                rows_out[i]["p_holm_fwer"] = hp
                rows_out[i]["sig_holm_0.05"] = bool(hp <= 0.05)
        for r in rows_out:
            if "p_holm_fwer" not in r:
                r["p_holm_fwer"] = r["p_two_sided_raw"]
                r["sig_holm_0.05"] = bool(r["p_two_sided_raw"] <= 0.05)
        return rows_out

    boot_rows = []
    boot_rows.extend(run_bootstrap_scheme("row_level_paired", "row_paired"))
    boot_rows.extend(run_bootstrap_scheme("cluster_16729_combined_missing", "cluster_16729_combined_missing"))
    boot_rows.extend(run_bootstrap_scheme("cluster_17653_singleton_missing", "cluster_17653_singleton_missing"))

    df_boot = pd.DataFrame(boot_rows)
    df_boot.to_csv(RESULTS_DIR / "exp3_ceas08_cluster_bootstrap.csv", index=False)

    # Build concise summary receipt
    chrono_a2_01 = df_chrono[
        (df_chrono["split_regime"] == "PRIMARY_DATED_ONLY")
        & (df_chrono["model"] == "A2")
        & (df_chrono["route"] == "final")
        & (df_chrono["alpha"] == 0.01)
    ].set_index("policy")
    e1_spa_a2_438 = df_e1[
        (df_e1["target"] == "spaphish") & (df_e1["model"] == "A2") & (df_e1["alpha"] == 0.01) & (df_e1["k"] == 438)
    ].set_index("policy")
    e1_spa_a2_680 = df_e1[
        (df_e1["target"] == "spaphish") & (df_e1["model"] == "A2") & (df_e1["alpha"] == 0.01) & (df_e1["k"] == 680)
    ].set_index("policy")
    ceas_a2_a1 = df_boot[
        (df_boot["contrast"] == "A2_minus_A1") & (df_boot["metric"].isin(["auroc", "auprc", "log_loss", "fpr"]))
    ][["bootstrap_scheme", "metric", "delta_point", "boot_se", "ci95_lo", "ci95_hi", "ci95_excludes_zero", "p_holm_fwer"]].to_dict("records")

    receipt = {
        "schema": "chronological_and_cluster_certification_receipt",
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - t0, 2),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "ceas08_domain_structure": dom_info,
        "key_findings": {
            "spaphish_chrono_a2_final_alpha_0.01": chrono_a2_01[["certifiable", "cutoff", "fpr", "fpr_ci_lo", "fpr_ci_hi", "recall"]].to_dict("index"),
            "spaphish_exchangeable_a2_k438_alpha_0.01": e1_spa_a2_438[["certifiable_share", "violation_prob", "fpr_median", "recall_median"]].to_dict("index"),
            "spaphish_exchangeable_a2_k680_alpha_0.01": e1_spa_a2_680[["certifiable_share", "violation_prob", "fpr_median", "recall_median"]].to_dict("index"),
            "ceas08_a2_minus_a1_cluster_vs_row": ceas_a2_a1,
        },
        "outputs": {
            "exp3_spaphish_chronological.csv": sha256_file(RESULTS_DIR / "exp3_spaphish_chronological.csv"),
            "exp3_exchangeable_e1_summary.csv": sha256_file(RESULTS_DIR / "exp3_exchangeable_e1_summary.csv"),
            "exp3_ceas08_cluster_bootstrap.csv": sha256_file(RESULTS_DIR / "exp3_ceas08_cluster_bootstrap.csv"),
        },
    }
    (RUNS_DIR / "chronological_and_cluster_certification.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt["key_findings"], indent=2), flush=True)


if __name__ == "__main__":
    main()

