#!/usr/bin/env python3
"""Paired class-stratified row bootstrap for cross-corpus email threat evaluation."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, log_loss


def scalar_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute scalar metrics evaluated inside each bootstrap draw."""
    p_clip = np.clip(p, 1e-7, 1.0 - 1e-7)
    pred = p >= threshold
    pos = y == 1
    neg = ~pos
    tp = int(np.sum(pred & pos))
    fn = int(np.sum((~pred) & pos))
    fp = int(np.sum(pred & neg))
    tn = int(np.sum((~pred) & neg))
    return {
        "auprc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p_clip, labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)),
        "recall": float(tp / max(1, tp + fn)),
        "fpr": float(fp / max(1, fp + tn)),
    }


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample positive and negative indices with replacement preserving class counts."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate([
        rng.choice(pos, size=len(pos), replace=True),
        rng.choice(neg, size=len(neg), replace=True),
    ])


def paired_bootstrap_delta(
    y: np.ndarray,
    p_base: np.ndarray,
    p_alt: np.ndarray,
    threshold_base: float,
    threshold_alt: float | None = None,
    reps: int = 1000,
    seed: int = 20261024,
) -> Dict[str, object]:
    """Compute point estimates and 95% percentile bootstrap intervals for (alt minus base)."""
    if threshold_alt is None:
        threshold_alt = threshold_base
    rng = np.random.default_rng(seed)
    keys = ("auprc", "log_loss", "brier", "recall", "fpr")
    observed_base = scalar_metrics(y, p_base, threshold_base)
    observed_alt = scalar_metrics(y, p_alt, threshold_alt)
    observed = {k: observed_alt[k] - observed_base[k] for k in keys}
    draws = {k: np.empty(reps, dtype=np.float64) for k in keys}
    for i in range(reps):
        idx = stratified_bootstrap_indices(y, rng)
        mb = scalar_metrics(y[idx], p_base[idx], threshold_base)
        ma = scalar_metrics(y[idx], p_alt[idx], threshold_alt)
        for k in keys:
            draws[k][i] = ma[k] - mb[k]
    ci = {
        k: {
            "estimate": float(observed[k]),
            "ci95_low": float(np.quantile(draws[k], 0.025)),
            "ci95_high": float(np.quantile(draws[k], 0.975)),
            "p_boot_le_zero": float(np.mean(draws[k] <= 0.0)),
            "p_boot_ge_zero": float(np.mean(draws[k] >= 0.0)),
        }
        for k in keys
    }
    return {"base": observed_base, "alt": observed_alt, "delta": ci}
