"""Independent verification of the IJIS E1/E2 outputs.

1. Hashes: CEAS-08 source, adapter, split, predictions, frozen model files (A1 bundles and A2 trees)
   against adapter_freeze.json and scoring_receipt.json; E1 output hashes against e1_summary.json.
2. E2 headline numbers recomputed from predictions with direct formulas (no sklearn metric calls
   except where noted), compared with e2_ceas08_results.json.
3. E1: every cell of e1_results_long.csv recomputed from e1_draws.parquet; a subset of draws
   regenerated from the seed and re-evaluated with explicit boolean masks on the remainder.
Exit 0 only if every check passes.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
EXP = OUT.parent
PAPER = EXP.parent
RUN = EXP / "rebaseline_rb_g1" / "runs" / "RB_G1_20261023_V1"
sys.path.insert(0, str(HERE.parent))
import e1_label_budget as e1  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree(path: Path) -> str:
    files = {str(p.relative_to(path)).replace("\\", "/"): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()


def auroc(y, s):
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ss = s[order]
    # average ranks for ties
    uniq, start, counts = np.unique(ss, return_index=True, return_counts=True)
    avg = start + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def avg_precision(y, s):
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tps = np.cumsum(ys); fps = np.cumsum(1 - ys)
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
    tp, fp = tps[last], fps[last]
    prec = tp / (tp + fp); rec = tp / tps[-1]
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def ece15(y, p, bins=15):
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.minimum(np.digitize(p, edges[1:-1], right=False), bins - 1)
    acc = 0.0
    for b in range(bins):
        m = ids == b
        if m.any():
            acc += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(acc)


def main():
    rc = OUT / "receipts"
    frz = json.loads((rc / "adapter_freeze.json").read_text(encoding="utf-8"))
    scr = json.loads((rc / "scoring_receipt.json").read_text(encoding="utf-8"))
    res = json.loads((OUT / "results" / "e2_ceas08_results.json").read_text(encoding="utf-8"))
    adapter = OUT / "data" / "ceas08_confirmation.parquet"
    split = OUT / "data" / "ceas08_confirmation_split.csv"
    pred_p = OUT / "predictions" / "ceas08_frozen_scores.parquet"

    # ---- 1. hashes
    check("ceas08 source sha256", sha256(PAPER / "DATASET" / "raw" / "CEAS_08.csv") == frz["source_sha256"])
    check("adapter sha256 (freeze)", sha256(adapter) == frz["adapter_sha256"])
    check("split sha256 (freeze)", sha256(split) == frz["split_sha256"])
    check("adapter sha256 (scoring)", sha256(adapter) == scr["adapter_sha256"])
    check("predictions sha256 (scoring)", sha256(pred_p) == scr["predictions_sha256"])
    check("predictions sha256 (results)", sha256(pred_p) == res["predictions_sha256"])
    check("freeze frozen before scoring", frz["status"] == "FROZEN_BEFORE_MODEL_SCORING"
          and frz["frozen_local_time"] <= scr["started_local"], f"{frz['frozen_local_time']} <= {scr['started_local']}")
    for route, r in scr["routes"].items():
        check(f"{route} A1 bundle sha256", sha256(Path(r["a1_path"])) == r["a1_sha256"])
        check(f"{route} A2 tree sha256", tree(Path(r["a2_root"])) == r["a2_tree"]["tree_sha256"])
    check("runtime compatibility PASS", scr["runtime_compatibility"]["status"] == "PASS")

    # ---- 2. E2 headline numbers
    ad = pd.read_parquet(adapter)
    check("contamination removed count", int(ad["contamination_removed"].sum()) == frz["removed_exact_or_near"])
    ad = ad.loc[~ad["contamination_removed"]].reset_index(drop=True)
    pred = pd.read_parquet(pred_p)
    check("prediction rows == evaluation rows", len(pred) == len(ad) == frz["evaluation_rows"], f"{len(pred)}")
    check("prediction row order", pred["row_id"].equals(ad["row_id"]))
    y = ad["label"].to_numpy(int)
    th = json.loads((RUN / "final_fit_receipt.json").read_text(encoding="utf-8"))["locked_thresholds"]
    thr = {"a1": float(th["A1_TFIDF_LR"]), "a2": float(th["A2_MDISTILBERT"])}
    for route in ("final", "fit_trec5", "fit_trec6"):
        for m in ("a1", "a2"):
            s = pred[f"{route}_{m}"].to_numpy(np.float64)
            ok_range = bool(np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all())
            check(f"{route} {m} scores finite in [0,1]", ok_range)
            pc = np.clip(s, 1e-7, 1 - 1e-7)
            mine = {
                "auroc": auroc(y, s), "average_precision": avg_precision(y, s),
                "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
                "brier": float(np.mean((s - y) ** 2)), "ece15": ece15(y, s),
                "fpr": float(((s >= thr[m]) & (y == 0)).sum() / (y == 0).sum()),
                "recall": float(((s >= thr[m]) & (y == 1)).sum() / (y == 1).sum()),
            }
            rep = res["routes"][route][m]
            for k, v in mine.items():
                check(f"E2 {route} {m} {k}", abs(v - rep[k]) < 1e-9, f"{v:.6f} vs {rep[k]:.6f}")

    # ---- 3. E1
    e1s = json.loads((OUT / "results" / "e1_summary.json").read_text(encoding="utf-8"))
    long_p = OUT / "results" / "e1_results_long.csv"
    draws_p = OUT / "results" / "e1_draws.parquet"
    check("e1_results_long.csv sha256", sha256(long_p) == e1s["outputs"]["e1_results_long.csv"])
    check("e1_draws.parquet sha256", sha256(draws_p) == e1s["outputs"]["e1_draws.parquet"])
    for k, v in e1s["inputs"].items():
        check(f"E1 input {k} sha256", sha256(Path(v["path"])) == v["sha256"])
    long = pd.read_csv(long_p)
    draws = pd.read_parquet(draws_p)
    # expected cell count
    exp_cells = 0
    for t, g in e1s["targets"].items():
        exp_cells += len(g["k_grid"]) * len(g["combos"]) * len(e1.ALPHAS) * len(e1.POLICIES)
        check(f"{t} k grid leaves >= 395", all(g["n"] - k >= 395 for k in g["k_grid"]))
    check("E1 cell count", len(long) == exp_cells, f"{len(long)} vs {exp_cells}")
    check("E1 draws per cell", (long["draws"] == e1.R).all())
    re = e1.summarise(draws)
    keys = ["target", "model", "route", "policy", "alpha", "k"]
    mm = long.merge(re, on=keys, suffixes=("", "_re"), validate="one_to_one")
    worst = 0.0
    for col in ["budget_violation_prob", "fpr_median", "fpr_p05", "fpr_p95", "recall_median", "recall_p05", "recall_p95", "no_cutoff_share"]:
        a, b = mm[col].to_numpy(float), mm[col + "_re"].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.abs(a - b))
        worst = max(worst, float(np.nanmax(d)))
        check(f"E1 cells recomputed from draws: {col}", bool(np.all(np.where(both_nan, True, d < 1e-9))))

    # regenerate draws independently for the first 25 draws of every (target, k)
    dev, targets, _ = e1.load_targets()
    ref = e1.development_reference(dev)
    mism = 0
    tested = 0
    cp_cache = {}
    for t_idx, tname in enumerate(e1.TARGETS):
        t = targets[tname]
        yy = t["y"]; n = len(yy)
        for k in e1s["targets"][tname]["k_grid"]:
            rng = np.random.default_rng(np.random.SeedSequence(e1.SEED, spawn_key=(t_idx, k)))
            dr = np.stack([rng.choice(n, size=k, replace=False) for _ in range(e1.R)])
            sub = draws[(draws.target == tname) & (draws.k == k) & (draws.draw < 25)]
            sub_idx = {}
            for rec in sub.to_dict("records"):
                sub_idx.setdefault((rec["draw"], rec["model"], rec["route"], rec["alpha"], rec["policy"]), []).append(rec)
            for r in range(25):
                lab = np.zeros(n, bool); lab[dr[r]] = True
                rem = ~lab
                n0l = int((yy[lab] == 0).sum())
                for (m, route), s in t["scores"].items():
                    for alpha in e1.ALPHAS:
                        out = {}
                        # P0 explicit
                        flag = s >= ref[m]["cut"][alpha]["common"]
                        out["P0"] = flag
                        # P1 explicit order statistic
                        negs = np.sort(s[lab & (yy == 0)])[::-1]
                        a1w = int(math.floor(alpha * n0l))
                        c1 = negs.min() if a1w >= len(negs) else np.nextafter(negs[a1w], np.inf)
                        out["P1"] = s >= c1
                        # P2 explicit CP search
                        if (n0l, alpha) not in cp_cache:
                            xr = np.arange(n0l)
                            cp_cache[(n0l, alpha)] = np.flatnonzero(beta.ppf(0.95, xr + 1, n0l - xr) <= alpha)
                        xs = cp_cache[(n0l, alpha)]
                        if xs.size:
                            x = int(xs.max())
                            c2 = negs.min() if x >= len(negs) else np.nextafter(negs[x], np.inf)
                            out["P2"] = s >= c2
                        else:
                            out["P2"] = None
                        for pol, f in out.items():
                            rows = sub_idx.get((r, m, route, alpha, pol), [])
                            tested += 1
                            if len(rows) != 1:
                                mism += 1
                                continue
                            row = rows[0]
                            if f is None:
                                mism += int(bool(row["defined"]))
                                continue
                            fpr = (f & rem & (yy == 0)).sum() / (rem & (yy == 0)).sum()
                            rec = (f & rem & (yy == 1)).sum() / (rem & (yy == 1)).sum()
                            if not (row["defined"] and abs(fpr - row["fpr"]) < 1e-12 and abs(rec - row["recall"]) < 1e-12):
                                mism += 1
    check("E1 regenerated draws (P0, P1, P2, first 25 draws per target and k)", mism == 0, f"{tested} checks, {mism} mismatches")

    # locked cutoff equals recomputed P0 common cutoff at alpha 0.01
    check("P0 A1 cutoff at 0.01 equals locked threshold", ref["a1"]["cut"][0.01]["common"] == thr["a1"])
    check("P0 A2 cutoff at 0.01 equals locked threshold", ref["a2"]["cut"][0.01]["common"] == thr["a2"])

    print(json.dumps({"status": "PASS" if not FAIL else "FAIL", "failed": FAIL}, indent=2))
    sys.exit(0 if not FAIL else 1)


if __name__ == "__main__":
    main()
