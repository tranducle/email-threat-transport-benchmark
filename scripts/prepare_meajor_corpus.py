#!/usr/bin/env python3
"""Verify local MeAJOR preprocessed parquet table required for full retraining."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPECTED_SOURCES = {
    "trec6": {"rows": 14685, "positives": 3503},
    "trec5": {"rows": 46640, "positives": 19235},
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local MeAJOR preprocessed parquet partition counts.")
    parser.add_argument("--data-path", type=Path, default=Path("data/meajor_preprocessed.parquet"))
    args = parser.parse_args()

    if not args.data_path.exists():
        print(f"Dataset file not found at {args.data_path}. See DATA.md for acquisition instructions.")
        return 1

    import pandas as pd

    df = pd.read_parquet(args.data_path, columns=["row_id", "source", "label"])
    for src, exp in EXPECTED_SOURCES.items():
        sub = df[df["source"] == src]
        n_rows = len(sub)
        n_pos = int(sub["label"].astype(int).sum())
        if n_rows != exp["rows"] or n_pos != exp["positives"]:
            print(f"FAIL: {src} count mismatch: got ({n_rows}, {n_pos}), expected ({exp['rows']}, {exp['positives']})")
            return 1
        print(f"PASS: {src} verified (N={n_rows}, Positives={n_pos})")

    print("Both active reciprocal development sources (trec6, trec5; N=61,325) verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
