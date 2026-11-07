"""R2 review analyses for Manuscript_IJIS.tex (author-approved 2026-11-06; saved scores only).

No model is trained, scored, re-seeded, or tuned. Inputs are the frozen score files and saved
E1 draws written by the E1/E2 study; every input is hashed into the output.

  A (F1)  same-model held-out cutoffs per route, applied to SpaPhish v5 and CEAS-08, next to the
          shared (common) cutoff; development cross-check against the locked thresholds.
  B (F1, F20) per-route score ceilings, storage dtype, saturation and ties at each cutoff.
  C (F3)  mean realized FPR (with the violation probability) of P0 to P4 at every alpha and k.
  D (F19) full-label reference cutoff on each target stream and its recall.
  E (F2)  analytic label requirement: negatives needed, expected k, and the hypergeometric
          probability that a k-message sample holds enough negatives, against the observed
          certifiable share of P2.

Usage: python -I r2_analyses.py   (paths resolve from this file; output in ../results/)
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "results"
EXP = HERE.parents[2]
RUN = EXP / "rebaseline_rb_g1" / "runs" / "RB_G1_20261023_V1"
EXT = EXP / "top_q1_external_confirmation_20261105"
IJ = EXP / "ijis_e1e2_20261106"
ALPHAS = [0.005, 0.01, 0.02]
DELTA = 0.05
LOCKED = {"a1": 0.8367079295539985, "a2": 0.9996199607849122}
VAL = {  # route -> held-out source and score files (the route model's own held-out scores)
    "fit_trec5": ("trec6", RUN / "fit_trec5__val_trec6" / "A1_TFIDF_LR" / "tfidf_lr_validation.parquet",
                  RUN / "fit_trec5__val_trec6" / "A2_MDISTILBERT" / "ensemble_validation.parquet"),
    "fit_trec6": ("trec5", RUN / "fit_trec6__val_trec5" / "A1_TFIDF_LR" / "tfidf_lr_validation.parquet",
                  RUN / "fit_trec6__val_trec5" / "A2_MDISTILBERT" / "ensemble_validation.parquet"),
}
EXTERNAL = {
    "spaphish": (EXT / "predictions" / "spaphish_frozen_scores.parquet", EXT / "data" / "spaphish_confirmation.parquet"),
    "ceas08": (IJ / "predictions" / "ceas08_frozen_scores.parquet", IJ / "data" / "ceas08_confirmation.parquet"),
}
INPUTS = {}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    INPUTS[str(p.relative_to(EXP)).replace("\\", "/")] = h.hexdigest()
    return h.hexdigest()


def threshold_at_max_fpr(y, s, alpha):
    """Same rule as rb_g1_semantic_anchor.threshold_at_max_fpr (smallest cutoff with at most
    floor(alpha * n0) false positives; decisions are score >= cutoff)."""
    neg = np.sort(s[y == 0])[::-1]
    allowed = int(math.floor(alpha * len(neg)))
    if allowed >= len(neg):
        return float(np.min(neg))
    return float(np.nextafter(neg[allowed], np.inf))


def rates(y, s, t):
    f = s >= t
    return {"fpr": float(f[y == 0].mean()), "recall": float(f[y == 1].mean()),
            "fp": int(f[y == 0].sum()), "tp": int(f[y == 1].sum()), "flagged": int(f.sum())}


def load_val(route, m):
    src, p1, p2 = VAL[route]
    p = p1 if m == "a1" else p2
    sha(p)
    d = pd.read_parquet(p)
    return src, d["label"].to_numpy(int), d["score"].to_numpy(np.float64), str(d["score"].dtype)


def load_external(name):
    pp, dp = EXTERNAL[name]
    sha(pp), sha(dp)
    pred = pd.read_parquet(pp)
    data = pd.read_parquet(dp, columns=[c for c in ("row_id", "label", "contamination_removed")
                                        if c in pd.read_parquet(dp).columns])
    if "contamination_removed" in data.columns:
        data = data.loc[~data["contamination_removed"].astype(bool)].reset_index(drop=True)
    if not pred["row_id"].equals(data["row_id"]):
        raise RuntimeError(f"row order mismatch for {name}")
    return data["label"].to_numpy(int), pred, {c: str(pred[c].dtype) for c in pred.columns[1:]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"schema": "P8_IJIS_R2_REVIEW_ANALYSES_V1", "approved": "author, 2026-11-06 (revise per review)",
           "no_model_run": True}
    # A. own and common cutoffs
    own, val_dtype = {}, {}
    for route in VAL:
        for m in ("a1", "a2"):
            src, y, s, dt = load_val(route, m)
            val_dtype[f"{route}_{m}"] = dt
            own[(route, m)] = {str(a): threshold_at_max_fpr(y, s, a) for a in ALPHAS}
            own[(route, m)]["heldout_source"] = src
            own[(route, m)]["heldout_rates_at_0.01"] = rates(y, s, own[(route, m)]["0.01"])
            # shared cutoff on the same held-out source (locked value; equality to the recomputed
            # common cutoff is checked below in dev_check_common_equals_locked_0.01)
            own[(route, m)]["heldout_rates_at_shared_0.01"] = rates(y, s, LOCKED[m])
            own[(route, m)]["heldout_max_score"] = float(s.max())
            own[(route, m)]["heldout_ceiling_vs_other_cutoff"] = None
    common = {m: {str(a): max(own[("fit_trec5", m)][str(a)], own[("fit_trec6", m)][str(a)]) for a in ALPHAS}
              for m in ("a1", "a2")}
    res["dev_check_common_equals_locked_0.01"] = {m: common[m]["0.01"] == LOCKED[m] for m in ("a1", "a2")}
    res["common_set_by_route"] = {m: {str(a): ("fit_trec5" if own[("fit_trec5", m)][str(a)] >= own[("fit_trec6", m)][str(a)]
                                                else "fit_trec6") for a in ALPHAS} for m in ("a1", "a2")}
    res["own_cutoffs"] = {f"{r}_{m}": v for (r, m), v in own.items()}
    res["common_cutoffs"] = common
    res["validation_score_dtype"] = val_dtype

    cells, ceilings = [], {}
    for name in EXTERNAL:
        y, pred, dtypes = load_external(name)
        res.setdefault("external_score_dtype", {})[name] = dtypes
        for route in ("final", "fit_trec5", "fit_trec6"):
            for m in ("a1", "a2"):
                s = pred[f"{route}_{m}"].to_numpy(np.float64)
                ceilings[f"{name}_{route}_{m}"] = {
                    "max_score": float(s.max()), "n_distinct": int(np.unique(s).size), "n": int(s.size),
                    "share_pos_ge_0.999": float((s[y == 1] >= 0.999).mean()),
                    "share_neg_ge_0.999": float((s[y == 0] >= 0.999).mean()),
                    "n_at_max": int((s == s.max()).sum()),
                }
                for a in ALPHAS:
                    c = common[m][str(a)]
                    row = {"corpus": name, "route": route, "model": m, "alpha": a,
                           "common_cutoff": c, "common": rates(y, s, c),
                           "ties_at_common": int((s == c).sum()),
                           "common_above_max_score": bool(c > s.max())}
                    if route in VAL:
                        o = own[(route, m)][str(a)]
                        row.update({"own_cutoff": o, "own": rates(y, s, o), "ties_at_own": int((s == o).sum()),
                                    "own_equals_common": o == c})
                    cells.append(row)
    res["external_cells"] = cells
    res["score_ceilings"] = ceilings
    for route in VAL:
        for m in ("a1", "a2"):
            other = common[m]["0.01"]
            own[(route, m)]["heldout_ceiling_vs_other_cutoff"] = own[(route, m)]["heldout_max_score"] < other
    res["own_cutoffs"] = {f"{r}_{m}": v for (r, m), v in own.items()}

    def pick(c, which):
        """common: shared cutoff everywhere; own: route model's own held-out cutoff, final route
        keeps the shared cutoff it borrows; own_routes_only: source-trained routes only."""
        if which == "common":
            return c["common"]
        if which == "own":
            return c["own"] if "own" in c else c["common"]
        return c.get("own")

    def tally(a, which):
        out = {"cells": 0, "exceed": 0, "flag_nothing": 0, "hold": 0}
        for c in cells:
            if c["alpha"] != a:
                continue
            r = pick(c, which)
            if r is None:
                continue
            out["cells"] += 1
            if r["flagged"] == 0:
                out["flag_nothing"] += 1
            elif r["fpr"] > a:
                out["exceed"] += 1
            else:
                out["hold"] += 1
        return out
    res["tally"] = {str(a): {"common_all12": tally(a, "common"), "own_where_defined_else_borrowed": tally(a, "own"),
                             "own_route_cells_only": tally(a, "own_routes_only")} for a in ALPHAS}

    # C. mean realized FPR from saved E1 draws
    dpath = IJ / "results" / "e1_draws.parquet"
    sha(dpath)
    d = pd.read_parquet(dpath)
    d = d[d["defined"]]
    d["viol"] = d["fpr"] > d["alpha"]
    g = d.groupby(["target", "model", "route", "policy", "alpha", "k"])
    c = g.agg(mean_fpr=("fpr", "mean"), median_fpr=("fpr", "median"), viol=("viol", "mean"), n=("fpr", "size"),
              mean_recall=("recall", "mean"), median_recall=("recall", "median")).reset_index()
    c.to_csv(OUT / "r2_mean_fpr_long.csv", index=False)
    a01 = c[c["alpha"] == 0.01]
    summ = {}
    for pol in ("P0", "P1", "P2", "P3", "P4"):
        x = a01[a01["policy"] == pol]
        summ[pol] = {"mean_fpr_min": float(x["mean_fpr"].min()), "mean_fpr_max": float(x["mean_fpr"].max()),
                     "viol_min": float(x["viol"].min()), "viol_max": float(x["viol"].max())}
    big = a01[(a01["policy"] == "P1") & (a01["k"] >= 500)]
    summ["P1_k_ge_500"] = {"mean_fpr_min": float(big["mean_fpr"].min()), "mean_fpr_max": float(big["mean_fpr"].max())}
    res["mean_fpr_alpha_0.01"] = summ

    # D. full-label reference cutoff on each target (oracle reference for recall)
    lab = pd.read_csv(IJ / "results" / "e1_results_long.csv")
    combos = lab[["target", "model", "route"]].drop_duplicates().to_dict("records")
    ref = []
    ext = {n: load_external(n) for n in EXTERNAL}
    for cb in combos:
        if cb["target"] in ("trec6", "trec5"):
            route = cb["route"]
            _, y, s, _ = load_val(route, cb["model"])
        else:
            y, pred, _ = ext[cb["target"]]
            s = pred[f"{cb['route']}_{cb['model']}"].to_numpy(np.float64)
        for a in ALPHAS:
            t = threshold_at_max_fpr(y, s, a)
            r = rates(y, s, t)
            ref.append({**cb, "alpha": a, "ref_cutoff": t, "ref_recall": r["recall"], "ref_fpr": r["fpr"]})
    pd.DataFrame(ref).to_csv(OUT / "r2_full_label_reference.csv", index=False)
    res["full_label_reference"] = ref

    # E. analytic label requirement
    counts = {}
    for t in ("trec6", "trec5"):
        _, y, _, _ = load_val("fit_trec5" if t == "trec6" else "fit_trec6", "a1")
        counts[t] = (int(y.size), int((y == 0).sum()))
    for n in EXTERNAL:
        y = ext[n][0]
        counts[n] = (int(y.size), int((y == 0).sum()))
    req = {}
    for a in ALPHAS:
        n0 = math.ceil(math.log(DELTA) / math.log(1 - a))
        req[str(a)] = {"n0_required": n0, "targets": {}}
        for t, (n, n0t) in counts.items():
            kk = sorted(lab.loc[lab["target"] == t, "k"].unique())
            obs = lab[(lab["target"] == t) & (lab["policy"] == "P2") & (lab["alpha"] == a)]
            obs = obs.groupby("k")["no_cutoff_share"].max().to_dict()
            req[str(a)]["targets"][t] = {
                "n": n, "negatives": n0t, "negative_share": n0t / n,
                "expected_k": n0 / (n0t / n) if n0t >= n0 else None,
                "feasible": n0t >= n0,
                "per_k": {str(k): {"p_enough_negatives": float(hypergeom(n, n0t, k).sf(n0 - 1)),
                                   "observed_certifiable_share": float(1 - obs.get(k, float("nan")))} for k in kk},
            }
    res["label_requirement"] = req
    res["counts"] = counts
    res["inputs_sha256"] = INPUTS
    (OUT / "r2_results.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"tally": res["tally"], "dev_check": res["dev_check_common_equals_locked_0.01"],
                      "common_set_by_route": res["common_set_by_route"], "mean_fpr": summ}, indent=1))


if __name__ == "__main__":
    main()
