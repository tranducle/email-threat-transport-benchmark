#!/usr/bin/env python3
"""Stage 1: Reciprocal source-held-out semantic anchor evaluation (A1_TFIDF_LR vs A2_MDISTILBERT)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics_and_thresholds import (
    compute_metric_pack,
    lock_conservative_threshold,
    threshold_at_max_fpr,
)

RECIPROCAL_FOLDS = [
    ("trec5", "trec6"),
    ("trec6", "trec5"),
]


def run_tfidf_lr_folds(data_path: Path, output_dir: Path) -> Dict[str, object]:
    """Execute the reciprocal source-held-out protocol for A1_TFIDF_LR."""
    df = pd.read_parquet(data_path, columns=["row_id", "label", "source", "text_clean"])
    df = df[df["source"].isin(["trec5", "trec6"])].copy()
    df["label"] = df["label"].astype(int)
    df["text_clean"] = df["text_clean"].fillna("").astype(str)

    fold_thresholds: Dict[str, float] = {}
    fold_predictions: Dict[str, pd.DataFrame] = {}

    for fit_src, val_src in RECIPROCAL_FOLDS:
        tr = df[df["source"] == fit_src].reset_index(drop=True)
        va = df[df["source"] == val_src].reset_index(drop=True)

        vec = TfidfVectorizer(
            max_features=20000,
            min_df=2,
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        x_tr = vec.fit_transform(tr["text_clean"])
        x_va = vec.transform(va["text_clean"])

        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )
        clf.fit(x_tr, tr["label"].to_numpy(int))
        scores = clf.predict_proba(x_va)[:, 1].astype(float)

        fold_thresholds[val_src] = threshold_at_max_fpr(va["label"].to_numpy(int), scores, max_fpr=0.01)
        fold_predictions[val_src] = pd.DataFrame({
            "row_id": va["row_id"],
            "label": va["label"],
            "score": scores,
        })

    locked_tau = lock_conservative_threshold(fold_thresholds)
    by_source = {
        src: compute_metric_pack(
            pred_df["label"].to_numpy(int),
            pred_df["score"].to_numpy(float),
            locked_tau,
        )
        for src, pred_df in fold_predictions.items()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "anchor": "A1_TFIDF_LR",
        "fold_thresholds": fold_thresholds,
        "locked_threshold": locked_tau,
        "by_source": by_source,
    }
    (output_dir / "a1_tfidf_lr_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1 semantic anchor evaluation.")
    parser.add_argument("--data-path", type=Path, default=Path("data/meajor_preprocessed.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1"))
    parser.add_argument("--verify-frozen", action="store_true", help="Print and verify frozen Stage 1 metrics")
    args = parser.parse_args()

    if args.verify_frozen:
        summary = json.loads((ROOT / "configs/published_results_summary.json").read_text(encoding="utf-8"))
        metrics = summary["stage1_semantic_anchors"]["metrics"]
        print("Stage 1 Frozen Semantic Anchor Results:")
        for anchor in ["A1_TFIDF_LR", "A2_MDISTILBERT"]:
            tau = metrics[anchor]["locked_threshold"]
            for src in ["trec6", "trec5"]:
                m = metrics[anchor]["by_source"][src]
                op = m["locked_operating_point"]
                print(
                    f"  {anchor} on {src} (N={m['n']}, tau*={tau:.4f}): "
                    f"AUPRC={m['auprc']:.4f}, Brier={m['brier']:.4f}, ECE15={m['ece15']:.4f}, "
                    f"Recall={op['recall']:.4f}, FPR={op['fpr']:.4f}"
                )
        return 0

    run_tfidf_lr_folds(args.data_path, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
