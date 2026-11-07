"""E1: restoring the FPR budget on a new stream under a label budget (IJIS package, 2026-11-06).

Specification: 9_Audit/review_pipeline_20261106_jisa_v1/00_pipeline/EXPERIMENT_PLAN_JISA.md, section
E1, with the changes in ../EXPERIMENT_PLAN_IJIS.md (CEAS-08 as fourth target, k cap so that at least
395 messages remain). Frozen scores only; nothing is trained, re-tuned or re-seeded.

Policies (all on frozen scores; decision rule score >= cutoff; comparisons in float64):
  P0  frozen development common cutoff at alpha (max of the two source-held-out cutoffs).
  P1  label k random target messages (simple random sample, natural prevalence); cutoff by the
      paper's order-statistic rule on the labelled negatives (allowed FP = floor(alpha * n0)).
  P2  as P1, but allowed FP = largest x with one-sided 95% Clopper-Pearson upper bound
      U(x, n0) <= alpha; if U(0, n0) > alpha the draw is "not certifiable" (no cutoff).
  P3  label-free quantile transfer: target scores mapped to the pooled development held-out score
      distribution by rank (empirical CDF of the unlabelled target scores, inverted-CDF development
      quantile), then the development common cutoff.
  P4  label-free Platt scaling fitted on the pooled development held-out scores only
      (logistic regression on logit(score), unpenalised), development common cutoff recomputed on
      the calibrated development scores with the same rule, applied to calibrated target scores.
Every policy is evaluated on the unlabelled remainder (N - k messages) of the same draw.

Grid: alpha in {0.005, 0.01, 0.02}; k in {50, 100, 200, 500, 1000, 2000} with k <= N - 395;
R = 1,000 draws per cell; seed 20261107. Draws are shared by every model, route, policy and alpha
of a target and k (SeedSequence(20261107, spawn_key=(target_index, k))), so cells are paired.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
EXP = OUT.parent
RUN = EXP / "rebaseline_rb_g1" / "runs" / "RB_G1_20261023_V1"
EXT = EXP / "top_q1_external_confirmation_20261105"
RESULT_DIR = OUT / "results"
SEED = 20261107
R = 1000
ALPHAS = [0.005, 0.01, 0.02]
KS = [50, 100, 200, 500, 1000, 2000]
MIN_REMAIN = 395
POLICIES = ["P0", "P1", "P2", "P3", "P4"]
CP_CONF = 0.95
LOGIT_EPS = 1e-12

DEV = {
    "trec6": {"route": "fit_trec5",
              "a1": RUN / "fit_trec5__val_trec6" / "A1_TFIDF_LR" / "tfidf_lr_validation.parquet",
              "a2": RUN / "fit_trec5__val_trec6" / "A2_MDISTILBERT" / "ensemble_validation.parquet"},
    "trec5": {"route": "fit_trec6",
              "a1": RUN / "fit_trec6__val_trec5" / "A1_TFIDF_LR" / "tfidf_lr_validation.parquet",
              "a2": RUN / "fit_trec6__val_trec5" / "A2_MDISTILBERT" / "ensemble_validation.parquet"},
}
SPA_PRED = EXT / "predictions" / "spaphish_frozen_scores.parquet"
SPA_DATA = EXT / "data" / "spaphish_confirmation.parquet"
CEAS_PRED = OUT / "predictions" / "ceas08_frozen_scores.parquet"
CEAS_DATA = OUT / "data" / "ceas08_confirmation.parquet"
TARGETS = ["trec6", "trec5", "spaphish", "ceas08"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cutoff_allowed(neg_desc: np.ndarray, allowed_fp: int) -> float:
    """Paper rule (rb_g1_semantic_anchor.threshold_at_max_fpr) with an explicit FP allowance."""
    if allowed_fp >= len(neg_desc):
        return float(np.min(neg_desc))
    return float(np.nextafter(neg_desc[allowed_fp], np.inf))


def threshold_at_max_fpr(y, s, alpha) -> float:
    neg = np.sort(np.asarray(s, dtype=np.float64)[np.asarray(y) == 0])[::-1]
    t = cutoff_allowed(neg, int(math.floor(alpha * len(neg))))
    assert int((neg >= t).sum()) <= int(math.floor(alpha * len(neg)))
    return t


def cp_allowance_table(alpha: float, nmax: int) -> np.ndarray:
    """xmax[n0] = largest x with one-sided CP upper bound U(x, n0) <= alpha, or -1 if none."""
    out = np.full(nmax + 1, -1, dtype=np.int64)
    for n0 in range(1, nmax + 1):
        x = np.arange(0, n0)
        u = beta.ppf(CP_CONF, x + 1, n0 - x)
        ok = np.flatnonzero(u <= alpha)
        out[n0] = int(ok.max()) if ok.size else -1
    return out


def load_targets():
    dev = {s: {m: pd.read_parquet(DEV[s][m]) for m in ("a1", "a2")} for s in DEV}
    inputs = {}
    for s in DEV:
        for m in ("a1", "a2"):
            inputs[f"dev_{s}_{m}"] = {"path": str(DEV[s][m]), "sha256": sha256(DEV[s][m]),
                                      "stored_dtype": str(dev[s][m]["score"].dtype)}
    targets = {}
    for s in ("trec6", "trec5"):
        d1, d2 = dev[s]["a1"], dev[s]["a2"]
        if not d1["row_id"].equals(d2["row_id"]) or not d1["label"].equals(d2["label"]):
            m = d1[["row_id", "label", "score"]].merge(d2[["row_id", "label", "score"]], on="row_id",
                                                        suffixes=("_a1", "_a2"), validate="one_to_one")
            if not (m["label_a1"] == m["label_a2"]).all():
                raise RuntimeError(f"label mismatch between A1 and A2 for {s}")
            y = m["label_a1"].to_numpy(int)
            sc = {("a1", DEV[s]["route"]): m["score_a1"].to_numpy(np.float64),
                  ("a2", DEV[s]["route"]): m["score_a2"].to_numpy(np.float64)}
        else:
            y = d1["label"].to_numpy(int)
            sc = {("a1", DEV[s]["route"]): d1["score"].to_numpy(np.float64),
                  ("a2", DEV[s]["route"]): d2["score"].to_numpy(np.float64)}
        targets[s] = {"y": y, "scores": sc}
    for name, pp, dp in (("spaphish", SPA_PRED, SPA_DATA), ("ceas08", CEAS_PRED, CEAS_DATA)):
        pred = pd.read_parquet(pp)
        data = pd.read_parquet(dp)
        if "contamination_removed" in data.columns:
            data = data.loc[~data["contamination_removed"]].reset_index(drop=True)
        if not pred["row_id"].equals(data["row_id"]):
            raise RuntimeError(f"row order mismatch for {name}")
        inputs[f"{name}_predictions"] = {"path": str(pp), "sha256": sha256(pp)}
        inputs[f"{name}_adapter"] = {"path": str(dp), "sha256": sha256(dp)}
        sc = {}
        for route in ("final", "fit_trec5", "fit_trec6"):
            for m in ("a1", "a2"):
                sc[(m, route)] = pred[f"{route}_{m}"].to_numpy(np.float64)
        targets[name] = {"y": data["label"].to_numpy(int), "scores": sc}
    for name, t in targets.items():
        if set(np.unique(t["y"])) != {0, 1}:
            raise RuntimeError(f"{name} labels not binary")
        for key, s in t["scores"].items():
            if len(s) != len(t["y"]) or not np.isfinite(s).all() or (s < 0).any() or (s > 1).any():
                raise RuntimeError(f"STOP RULE: invalid scores or row count for {name} {key}")
    return dev, targets, inputs


def development_reference(dev):
    """Pooled development held-out scores, common cutoffs, Platt maps per model."""
    ref = {}
    for m in ("a1", "a2"):
        per = {s: {"y": dev[s][m]["label"].to_numpy(int), "s": dev[s][m]["score"].to_numpy(np.float64)} for s in DEV}
        pooled = np.sort(np.concatenate([per["trec5"]["s"], per["trec6"]["s"]]))
        pooled_y = np.concatenate([per["trec5"]["y"], per["trec6"]["y"]])
        pooled_raw = np.concatenate([per["trec5"]["s"], per["trec6"]["s"]])
        x = np.log(np.clip(pooled_raw, LOGIT_EPS, 1 - LOGIT_EPS) / (1 - np.clip(pooled_raw, LOGIT_EPS, 1 - LOGIT_EPS)))
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
        lr.fit(x.reshape(-1, 1), pooled_y)
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])

        def platt(s, a=a, b=b):
            z = np.log(np.clip(s, LOGIT_EPS, 1 - LOGIT_EPS) / (1 - np.clip(s, LOGIT_EPS, 1 - LOGIT_EPS)))
            return 1.0 / (1.0 + np.exp(-(a * z + b)))

        cut, cut_cal, share = {}, {}, {}
        for alpha in ALPHAS:
            per_src = {s: threshold_at_max_fpr(per[s]["y"], per[s]["s"], alpha) for s in DEV}
            per_src_cal = {s: threshold_at_max_fpr(per[s]["y"], platt(per[s]["s"]), alpha) for s in DEV}
            cut[alpha] = {"per_source": per_src, "common": max(per_src.values())}
            cut_cal[alpha] = {"per_source": per_src_cal, "common": max(per_src_cal.values())}
            share[alpha] = float((pooled >= cut[alpha]["common"]).mean())
        ref[m] = {"pooled_sorted": pooled, "platt": platt, "platt_coef": a, "platt_intercept": b,
                  "cut": cut, "cut_cal": cut_cal, "pooled_share_ge_cut": share, "n_pooled": int(len(pooled))}
    return ref


def p3_mapped(target_scores: np.ndarray, pooled_sorted: np.ndarray) -> np.ndarray:
    st = np.sort(target_scores)
    u = np.searchsorted(st, target_scores, side="right") / len(st)
    return np.quantile(pooled_sorted, u, method="inverted_cdf")


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["target", "model", "route", "policy", "alpha", "k"]
    for key, g in df.groupby(keys, sort=False):
        d = g[g["defined"]]
        rec = dict(zip(keys, key))
        rec.update({
            "n_total": int(g["n_total"].iloc[0]), "n_eval": int(g["n_eval"].iloc[0]), "draws": int(len(g)),
            "draws_with_cutoff": int(len(d)), "no_cutoff_share": float(1 - len(d) / len(g)),
            "labelled_negatives_median": float(g["n0_lab"].median()),
            "labelled_negatives_p05": float(g["n0_lab"].quantile(0.05)),
            "labelled_negatives_p95": float(g["n0_lab"].quantile(0.95)),
        })
        if len(d):
            rec.update({
                "budget_violation_prob": float((d["fpr"] > rec["alpha"]).mean()),
                "fpr_median": float(d["fpr"].median()), "fpr_p05": float(d["fpr"].quantile(0.05)),
                "fpr_p95": float(d["fpr"].quantile(0.95)),
                "recall_median": float(d["recall"].median()), "recall_p05": float(d["recall"].quantile(0.05)),
                "recall_p95": float(d["recall"].quantile(0.95)),
                "cutoff_median": float(d["cutoff"].median()),
            })
        else:
            rec.update({k: None for k in ["budget_violation_prob", "fpr_median", "fpr_p05", "fpr_p95",
                                          "recall_median", "recall_p05", "recall_p95", "cutoff_median"]})
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    t_start = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    long_path = RESULT_DIR / "e1_results_long.csv"
    if long_path.exists():
        raise RuntimeError(f"{long_path} exists; refusing to overwrite")
    dev, targets, inputs = load_targets()
    ref = development_reference(dev)
    nmax = max(len(t["y"]) for t in targets.values())
    nmax = min(nmax, max(KS))
    cp = {alpha: cp_allowance_table(alpha, nmax) for alpha in ALPHAS}
    n0_min = {alpha: int(np.flatnonzero(cp[alpha] >= 0).min()) for alpha in ALPHAS}
    print(json.dumps({"cp_min_negatives": n0_min}), flush=True)

    records = []
    grid_log = {}
    for t_idx, tname in enumerate(TARGETS):
        t = targets[tname]
        y = t["y"]
        n = len(y)
        neg_total = int((y == 0).sum())
        pos_total = int((y == 1).sum())
        ks = [k for k in KS if n - k >= MIN_REMAIN]
        grid_log[tname] = {"n": n, "negatives": neg_total, "positives": pos_total, "k_grid": ks,
                           "k_excluded": [k for k in KS if k not in ks],
                           "combos": [f"{m}|{r}" for (m, r) in t["scores"]]}
        # fixed masks for label-free policies, per combo and alpha
        fixed = {}
        for (m, route), s in t["scores"].items():
            mapped = p3_mapped(s, ref[m]["pooled_sorted"])
            cal = ref[m]["platt"](s)
            for alpha in ALPHAS:
                fixed[(m, route, "P0", alpha)] = (s >= ref[m]["cut"][alpha]["common"], ref[m]["cut"][alpha]["common"])
                fixed[(m, route, "P3", alpha)] = (mapped >= ref[m]["cut"][alpha]["common"], ref[m]["cut"][alpha]["common"])
                fixed[(m, route, "P4", alpha)] = (cal >= ref[m]["cut_cal"][alpha]["common"], ref[m]["cut_cal"][alpha]["common"])
        fixed_tot = {key: (int((mask & (y == 0)).sum()), int((mask & (y == 1)).sum())) for key, (mask, _) in fixed.items()}
        sorted_full = {key: (np.sort(s[y == 0]), np.sort(s[y == 1])) for key, s in t["scores"].items()}
        for k in ks:
            rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(t_idx, k)))
            draws = np.stack([rng.choice(n, size=k, replace=False) for _ in range(R)])
            for r in range(R):
                idx = draws[r]
                yl = y[idx]
                negl = yl == 0
                n0l = int(negl.sum())
                n1l = k - n0l
                n0r = neg_total - n0l
                n1r = pos_total - n1l
                for (m, route), s in t["scores"].items():
                    sl = s[idx]
                    neg_desc = np.sort(sl[negl])[::-1]
                    nf, pf = sorted_full[(m, route)]
                    for alpha in ALPHAS:
                        base = {"target": tname, "model": m, "route": route, "alpha": alpha, "k": k, "draw": r,
                                "n_total": n, "n_eval": n - k, "n0_lab": n0l, "n1_lab": n1l}
                        for pol in ("P0", "P3", "P4"):
                            mask, cval = fixed[(m, route, pol, alpha)]
                            flag_neg_tot, flag_pos_tot = fixed_tot[(m, route, pol, alpha)]
                            ml = mask[idx]
                            fp_r = flag_neg_tot - int((ml & negl).sum())
                            tp_r = flag_pos_tot - int((ml & ~negl).sum())
                            records.append({**base, "policy": pol, "defined": True, "cutoff": cval,
                                            "allowed_fp_lab": -1, "fp": fp_r, "tp": tp_r,
                                            "fpr": fp_r / n0r, "recall": tp_r / n1r})
                        for pol in ("P1", "P2"):
                            if n0l == 0:
                                allowed = -1
                            elif pol == "P1":
                                allowed = int(math.floor(alpha * n0l))
                            else:
                                allowed = int(cp[alpha][n0l])
                            if allowed < 0:
                                records.append({**base, "policy": pol, "defined": False, "cutoff": np.nan,
                                                "allowed_fp_lab": allowed, "fp": -1, "tp": -1,
                                                "fpr": np.nan, "recall": np.nan})
                                continue
                            c = cutoff_allowed(neg_desc, allowed)
                            fp_all = len(nf) - int(np.searchsorted(nf, c, side="left"))
                            tp_all = len(pf) - int(np.searchsorted(pf, c, side="left"))
                            fp_r = fp_all - int((sl[negl] >= c).sum())
                            tp_r = tp_all - int((sl[~negl] >= c).sum())
                            records.append({**base, "policy": pol, "defined": True, "cutoff": c,
                                            "allowed_fp_lab": allowed, "fp": fp_r, "tp": tp_r,
                                            "fpr": fp_r / n0r, "recall": tp_r / n1r})
            print(json.dumps({"target": tname, "k": k, "records": len(records), "elapsed_s": round(time.time() - t_start, 1)}), flush=True)

    draws_df = pd.DataFrame.from_records(records)
    for col in ("fpr", "recall", "cutoff"):
        if draws_df.loc[draws_df["defined"], col].isna().any():
            raise RuntimeError(f"STOP RULE: non-finite {col} in a defined draw")
    draws_path = RESULT_DIR / "e1_draws.parquet"
    draws_df.to_parquet(draws_path, index=False)
    summ = summarise(draws_df)
    summ.to_csv(long_path, index=False)

    # P2 certifiable k: depends only on the labelled negatives, hence identical for every model and route
    p2 = {}
    for tname in TARGETS:
        p2[tname] = {}
        sub = summ[(summ.target == tname) & (summ.policy == "P2")]
        for alpha in ALPHAS:
            s = sub[sub.alpha == alpha]
            share = s.groupby("k")["draws_with_cutoff"].max().sort_index() / R
            all_k = [int(k) for k, v in share.items() if v >= 1.0]
            any_k = [int(k) for k, v in share.items() if v > 0.0]
            p2[tname][str(alpha)] = {
                "certified_share_by_k": {str(int(k)): float(v) for k, v in share.items()},
                "smallest_k_certifiable_in_all_draws": min(all_k) if all_k else None,
                "smallest_k_certifiable_in_any_draw": min(any_k) if any_k else None,
                "min_labelled_negatives_for_certification": n0_min[alpha],
            }
    summary = {
        "schema": "P8_IJIS_E1_LABEL_BUDGET_V1",
        "study_id": "P8-IJIS-E1-20261106",
        "seed": SEED, "draws_per_cell": R, "alphas": ALPHAS, "k_grid": KS, "min_remaining": MIN_REMAIN,
        "policies": POLICIES, "clopper_pearson_one_sided_confidence": CP_CONF,
        "decision_rule": "score >= cutoff, float64 comparison",
        "targets": grid_log,
        "inputs": inputs,
        "development_reference": {m: {"n_pooled": ref[m]["n_pooled"], "platt_coef": ref[m]["platt_coef"],
                                      "platt_intercept": ref[m]["platt_intercept"],
                                      "common_cutoff": {str(a): ref[m]["cut"][a] for a in ALPHAS},
                                      "common_cutoff_calibrated": {str(a): ref[m]["cut_cal"][a] for a in ALPHAS},
                                      "pooled_share_ge_common_cutoff": {str(a): ref[m]["pooled_share_ge_cut"][a] for a in ALPHAS}}
                                  for m in ("a1", "a2")},
        "p2_certifiable": p2,
        "cells": int(len(summ)),
        "draw_rows": int(len(draws_df)),
        "outputs": {"e1_results_long.csv": sha256(long_path), "e1_draws.parquet": sha256(draws_path)},
        "wall_seconds": time.time() - t_start,
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
    }
    (RESULT_DIR / "e1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "cells": len(summ), "draw_rows": len(draws_df), "wall_s": summary["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
