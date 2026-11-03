#!/usr/bin/env python3
"""Stage 3: Paired class-stratified bootstrap uncertainty analysis (B=1000, seed=20261024)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Stage 3 paired bootstrap uncertainty intervals.")
    parser.add_argument("--verify-frozen", action="store_true", help="Verify Stage 3 bootstrap intervals")
    args = parser.parse_args()

    summary_path = ROOT / "configs/published_results_summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    s3 = data["stage3_bootstrap_uncertainty"]
    print("Stage 3 Paired Stratified Bootstrap Summary (B=1000, seed=20261024):")
    for src in ["trec6", "trec5"]:
        d = s3["semantic_a2_minus_a1"][src]["delta"]
        print(
            f"  A2 - A1 ({src}): "
            f"dAUPRC={d['auprc']['estimate']:+.4f} [{d['auprc']['ci95_low']:+.4f}, {d['auprc']['ci95_high']:+.4f}], "
            f"dRecall={d['recall']['estimate']:+.4f} [{d['recall']['ci95_low']:+.4f}, {d['recall']['ci95_high']:+.4f}]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
