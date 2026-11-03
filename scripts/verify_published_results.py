#!/usr/bin/env python3
"""Verify all published tables and numerical claims in the IEEE TDSC manuscript."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def assert_close(actual: float, expected: float, tol: float, label: str) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{label}: actual={actual} vs expected={expected} (tol={tol})")


def main() -> int:
    summary_path = ROOT / "configs/published_results_summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    # 1. Corpus totals (Table I)
    c = data["corpus_summary"]
    assert c["trec6"]["n"] == 14685 and c["trec6"]["positives"] == 3503
    assert c["trec5"]["n"] == 46640 and c["trec5"]["positives"] == 19235
    assert c["pooled_development"]["n"] == 61325 and c["pooled_development"]["positives"] == 22738

    # 2. Stage 1 semantic anchor metrics (Table II & Figure 2)
    s1 = data["stage1_semantic_anchors"]["metrics"]
    assert_close(s1["A1_TFIDF_LR"]["locked_threshold"], 0.8367079295539985, 1e-9, "A1 locked threshold")
    assert_close(s1["A2_MDISTILBERT"]["locked_threshold"], 0.9996199607849122, 1e-9, "A2 locked threshold")

    a1_t6 = s1["A1_TFIDF_LR"]["by_source"]["trec6"]
    a2_t6 = s1["A2_MDISTILBERT"]["by_source"]["trec6"]
    a1_t5 = s1["A1_TFIDF_LR"]["by_source"]["trec5"]
    a2_t5 = s1["A2_MDISTILBERT"]["by_source"]["trec5"]

    assert_close(a1_t6["auprc"], 0.928061, 1e-5, "A1 trec6 AUPRC")
    assert_close(a2_t6["auprc"], 0.966141, 1e-5, "A2 trec6 AUPRC")
    assert_close(a1_t5["auprc"], 0.959218, 1e-5, "A1 trec5 AUPRC")
    assert_close(a2_t5["auprc"], 0.971450, 1e-5, "A2 trec5 AUPRC")

    # Common-threshold collapse on trec5 under A2_MDISTILBERT
    assert a2_t5["locked_operating_point"]["tp"] == 0
    assert_close(a2_t5["locked_operating_point"]["recall"], 0.0, 1e-12, "A2 trec5 Recall")
    assert_close(a2_t5["locked_operating_point"]["fpr"], 0.0, 1e-12, "A2 trec5 FPR")

    # 3. Stage 2 residual structural utility (Table III & Figure 3)
    s2 = data["stage2_residual_utility"]
    for bname, bdec in s2["block_decisions"].items():
        assert bdec["candidate_useful"] is False
        for fold_delta in bdec["fold_deltas"]:
            assert abs(fold_delta["auprc"]) < 1e-4, f"{bname} exceeded 1e-4 AUPRC bound: {fold_delta['auprc']}"

    # 4. Stage 3 bootstrap reconstruction tolerance and source isolation
    s3 = data["stage3_bootstrap_uncertainty"]
    assert s3["reconstruction"]["all_matches"] is True
    assert s3["bootstrap"]["repetitions"] == 1000
    assert s3["bootstrap"]["seed"] == 20261024
    assert s3["source_isolation"]["spaphish_accessed"] is False
    assert s3["source_isolation"]["trec7_accessed"] is False

    print("PASS: All published numerical claims in Table I, Table II, Table III, Figure 2, and Figure 3 verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
