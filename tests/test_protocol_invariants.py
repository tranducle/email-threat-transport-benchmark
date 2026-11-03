#!/usr/bin/env python3
"""Unit tests for threshold selection, structural feature blocks, bootstrap inference, and published claims."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bootstrap_inference import paired_bootstrap_delta
from src.metrics_and_thresholds import (
    compute_metric_pack,
    lock_conservative_threshold,
    threshold_at_max_fpr,
)
from src.structural_features import FEATURE_BLOCKS, feature_matrix


def test_threshold_at_max_fpr_respects_constraint() -> None:
    rng = np.random.default_rng(42)
    y = np.array([0] * 1000 + [1] * 400, dtype=int)
    scores = np.concatenate([rng.uniform(0.0, 0.9, 1000), rng.uniform(0.5, 1.0, 400)])
    tau = threshold_at_max_fpr(y, scores, max_fpr=0.01)
    m = compute_metric_pack(y, scores, tau)
    assert float(m["fpr"]) <= 0.01 + 1e-12


def test_lock_conservative_threshold_takes_maximum() -> None:
    locked = lock_conservative_threshold({"trec6": 0.812, "trec5": 0.9996})
    assert abs(locked - 0.9996) < 1e-12


def test_structural_feature_blocks_match_frozen_schema() -> None:
    assert len(FEATURE_BLOCKS["U_URL_STRUCTURE"]) == 7
    assert len(FEATURE_BLOCKS["D_DOMAIN_SHAPE_NO_IDENTITY"]) == 4
    assert len(FEATURE_BLOCKS["M_MESSAGE_STRUCTURE"]) == 6
    assert sum(len(v) for v in FEATURE_BLOCKS.values()) == 17

    df = pd.DataFrame({
        "url_indicator": [1, 0],
        "url_count": [2, 0],
        "url_count_parsed": [2, 0],
        "url_length_max": [45, 0],
        "url_length_avg": [38.5, 0.0],
        "url_subdom_max": [2, 0],
        "url_subdom_avg": [1.5, 0.0],
        "sender_domain_norm": ["example.com", "bank.org"],
        "receiver_domain_norm": ["user.net", "bank.org"],
        "url_domains_json": ['["example.com"]', '["phish.biz"]'],
        "text_length_chars": [500, 120],
        "token_count": [80, 20],
        "attachment_count": [0, 1],
        "has_attachments": [0, 1],
        "content_types": ['["text/plain"]', '["text/html", "application/pdf"]'],
        "attachment_types": ["[]", '["application/pdf"]'],
    })
    for bname, expected_cols in FEATURE_BLOCKS.items():
        mat, names = feature_matrix(df, bname)
        assert mat.shape == (2, len(expected_cols))
        assert names == expected_cols
        assert np.isfinite(mat).all()


def test_paired_bootstrap_delta_intervals() -> None:
    y = np.array([0] * 100 + [1] * 100, dtype=int)
    p_base = np.concatenate([np.linspace(0.01, 0.4, 100), np.linspace(0.5, 0.95, 100)])
    p_alt = np.clip(p_base + 0.03, 1e-4, 0.9999)
    res = paired_bootstrap_delta(y, p_base, p_alt, threshold_base=0.35, threshold_alt=0.35, reps=50, seed=42)
    assert res["delta"]["auprc"]["ci95_low"] <= res["delta"]["auprc"]["ci95_high"]


def test_published_results_verification_script() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_published_results.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
