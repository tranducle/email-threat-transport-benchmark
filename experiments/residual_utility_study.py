#!/usr/bin/env python3
"""Stage 2: De-identified residual structural utility probe on top of A2_MDISTILBERT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics_and_thresholds import compute_metric_pack, logit, sigmoid


def fit_and_evaluate_residual_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    p_train_oof: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    p_val_base: np.ndarray,
    locked_threshold: float,
    ridge_alpha: float = 1.0,
    coeff_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
    selection_seed: int = 20261025,
) -> Dict[str, object]:
    """Fit StandardScaler + Ridge on residual (y - p_oof), select fusion weight c, and score validation fold."""
    idx = np.arange(len(y_train))
    fit_idx, sel_idx = train_test_split(
        idx, test_size=0.20, stratify=y_train, random_state=selection_seed
    )

    scaler_inner = StandardScaler().fit(x_train[fit_idx])
    ridge_inner = Ridge(alpha=ridge_alpha).fit(
        scaler_inner.transform(x_train[fit_idx]),
        y_train[fit_idx] - p_train_oof[fit_idx],
    )
    r_sel = ridge_inner.predict(scaler_inner.transform(x_train[sel_idx]))

    sel_losses: Dict[str, float] = {}
    best_c = 0.0
    best_ll = float("inf")
    for c in coeff_grid:
        p_cand = sigmoid(logit(p_train_oof[sel_idx]) + float(c) * r_sel)
        ll = float(log_loss(y_train[sel_idx], np.clip(p_cand, 1e-7, 1.0 - 1e-7), labels=[0, 1]))
        sel_losses[str(c)] = ll
        if ll < best_ll - 1e-12 or (abs(ll - best_ll) <= 1e-12 and c < best_c):
            best_ll = ll
            best_c = float(c)

    scaler_full = StandardScaler().fit(x_train)
    ridge_full = Ridge(alpha=ridge_alpha).fit(
        scaler_full.transform(x_train),
        y_train - p_train_oof,
    )
    r_val = ridge_full.predict(scaler_full.transform(x_val))
    p_val_fused = sigmoid(logit(p_val_base) + best_c * r_val)

    base_metrics = compute_metric_pack(y_val, p_val_base, locked_threshold)
    fused_metrics = compute_metric_pack(y_val, p_val_fused, locked_threshold)
    delta = {
        k: float(fused_metrics[k]) - float(base_metrics[k])
        for k in ("auprc", "log_loss", "brier", "ece15", "fpr", "recall")
    }
    return {
        "coefficient": best_c,
        "coefficient_selection_log_loss": sel_losses,
        "baseline": base_metrics,
        "fused": fused_metrics,
        "delta": delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 residual structural utility probe.")
    parser.add_argument("--verify-frozen", action="store_true", help="Print and verify frozen Stage 2 results")
    args = parser.parse_args()

    summary = json.loads((ROOT / "configs/published_results_summary.json").read_text(encoding="utf-8"))
    folds = summary["stage2_residual_utility"]["folds"]
    print("Stage 2 Frozen Residual Utility Results:")
    for fold in folds:
        val_src = fold["validation_source"]
        for bname, res in fold["blocks"].items():
            d = res["delta"]
            print(
                f"  val={val_src} | {bname} (c={res['coefficient']}): "
                f"dAUPRC={d['auprc']:+.6f}, dLogLoss={d['log_loss']:+.6f}, "
                f"dBrier={d['brier']:+.6f}, dRecall={d['recall']:+.6f}, dFPR={d['fpr']:+.6f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
