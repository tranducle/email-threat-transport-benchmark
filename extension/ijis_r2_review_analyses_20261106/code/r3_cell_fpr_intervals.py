"""Exact 95% Clopper-Pearson intervals for the realized FPR of every frozen external cell at alpha = 0.01.

Usage: python -I r3_cell_fpr_intervals.py <r2_results.json> <out.json>
Reads only the saved false-positive counts of results/r2_results.json (no model, score file or label
draw is touched) and the negative counts of the two external evaluation sets.
A cell "clearly exceeds" when the lower bound is above alpha, "clearly keeps" when the upper bound is
below alpha, and is "within sampling error" otherwise.
"""
import json
import sys
from pathlib import Path

from scipy.stats import beta

sys.stdout.reconfigure(encoding="utf-8")
ALPHA = 0.01
N0 = {"spaphish": 664, "ceas08": 17246}


def cp(x, n):
    lo = 0.0 if x == 0 else float(beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(0.975, x + 1, n - x))
    return lo, hi


def classify(fp, flagged, n):
    if flagged == 0:
        return "flags nothing", None, None
    lo, hi = cp(fp, n)
    if lo > ALPHA:
        return "clearly exceeds", lo, hi
    if hi < ALPHA:
        return "clearly keeps", lo, hi
    return "within sampling error", lo, hi


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    d = json.loads(src.read_text(encoding="utf-8"))
    rows, tally = [], {}
    for c in d["external_cells"]:
        if abs(c["alpha"] - ALPHA) > 1e-12:
            continue
        n0 = N0[c["corpus"]]
        for policy in ("common", "own"):
            r = c.get(policy)
            if r is None or (policy == "own" and c["route"] == "final"):
                continue
            v, lo, hi = classify(int(r["fp"]), int(r["flagged"]), n0)
            rows.append({"corpus": c["corpus"], "route": c["route"], "model": c["model"], "policy": policy,
                         "fp": int(r["fp"]), "n0": n0, "fpr": r["fpr"], "ci95": [lo, hi], "verdict": v})
            tally.setdefault(policy, {}).setdefault(v, 0)
            tally[policy][v] += 1
    for r in rows:
        ci = "" if r["ci95"][0] is None else f"[{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]"
        print(f"{r['corpus']:9s} {r['route']:10s} {r['model']} {r['policy']:6s} "
              f"{r['fp']:>4d}/{r['n0']:<6d} fpr={r['fpr']:.4f} {ci:18s} {r['verdict']}")
    print(json.dumps(tally, indent=1))
    out.write_text(json.dumps({"source": str(src.name), "alpha": ALPHA, "no_model_run": True,
                               "rows": rows, "tally": tally}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
