#!/usr/bin/env python3
"""Threshold selection, calibration diagnostics, and binary classification metrics."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, log_loss


def threshold_at_max_fpr(y: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01) -> float:
    """Compute the smallest threshold on a validation fold achieving empirical FPR <= max_fpr."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y) != len(scores) or len(y) == 0:
        raise ValueError("y and scores must be non-empty and equal length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    negatives = scores[y == 0]
    if len(negatives) == 0:
        raise ValueError("at least one negative example is required")
    allowed_fp = int(math.floor(max_fpr * len(negatives)))
    descending = np.sort(negatives)[::-1]
    if allowed_fp >= len(descending):
        threshold = float(np.min(descending))
    else:
        threshold = float(np.nextafter(descending[allowed_fp], np.inf))
    actual_fp = int((negatives >= threshold).sum())
    if actual_fp > allowed_fp:
        raise AssertionError("threshold construction violated max_fpr")
    return threshold


def lock_conservative_threshold(fold_thresholds: Dict[str, float]) -> float:
    """Lock the operational threshold as the maximum across reciprocal validation folds."""
    if not fold_thresholds:
        raise ValueError("fold_thresholds must not be empty")
    return float(max(fold_thresholds.values()))


def ece15(y: np.ndarray, p: np.ndarray) -> float:
    """Compute 15-bin equal-width expected calibration error (ECE-15)."""
    edges = np.linspace(0.0, 1.0, 16)
    value = 0.0
    for i in range(15):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < 14 else (p <= edges[i + 1]))
        if m.any():
            value += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return value


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically clipped logistic sigmoid."""
    x_clip = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def logit(p: np.ndarray) -> np.ndarray:
    """Numerically clipped log-odds transform."""
    p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_clip / (1.0 - p_clip))


def compute_metric_pack(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float | int]:
    """Compute AUPRC, LogLoss, Brier, ECE-15, Recall, FPR, and confusion counts at threshold."""
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=float)
    pred = (p_arr >= threshold).astype(int)
    tn, fp, fn, tp = [int(x) for x in confusion_matrix(y_arr, pred, labels=[0, 1]).ravel()]
    eps = 1e-7
    return {
        "n": int(len(y_arr)),
        "positives": int(y_arr.sum()),
        "auprc": float(average_precision_score(y_arr, p_arr)),
        "log_loss": float(log_loss(y_arr, np.clip(p_arr, eps, 1.0 - eps), labels=[0, 1])),
        "brier": float(brier_score_loss(y_arr, p_arr)),
        "ece15": float(ece15(y_arr, p_arr)),
        "recall": float(tp / max(1, tp + fn)),
        "fpr": float(fp / max(1, fp + tn)),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "threshold": float(threshold),
    }
