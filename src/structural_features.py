#!/usr/bin/env python3
"""De-identified structural feature blocks U, D, and M for residual utility testing."""
from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

FEATURE_BLOCKS: Dict[str, List[str]] = {
    "U_URL_STRUCTURE": [
        "url_indicator",
        "url_count",
        "url_count_parsed",
        "url_length_max",
        "url_length_avg",
        "url_subdom_max",
        "url_subdom_avg",
    ],
    "D_DOMAIN_SHAPE_NO_IDENTITY": [
        "sender_domain_length",
        "receiver_domain_length",
        "sender_receiver_domain_equal",
        "sender_url_domain_match",
    ],
    "M_MESSAGE_STRUCTURE": [
        "text_length_chars",
        "token_count",
        "attachment_count",
        "has_attachments",
        "content_type_count",
        "attachment_type_count",
    ],
}


def count_listish(value: object) -> float:
    """Count elements in a JSON list or comma/semicolon-separated string field."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "[]", "()"}:
        return 0.0
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return float(len(obj))
    except Exception:
        pass
    return float(len([x for x in s.replace(";", ",").split(",") if x.strip()]))


def sender_url_match(sender: object, url_domains: object) -> float:
    """Return 1.0 if normalized sender domain appears in url_domains_json, else 0.0."""
    sd = "" if sender is None else str(sender).casefold().strip().rstrip(".")
    if not sd:
        return 0.0
    try:
        values = json.loads(str(url_domains))
        if isinstance(values, list):
            domains = {str(x).casefold().strip().rstrip(".") for x in values}
            return float(sd in domains)
    except Exception:
        return 0.0
    return 0.0


def feature_matrix(df: pd.DataFrame, block: str) -> Tuple[np.ndarray, List[str]]:
    """Construct the numeric design matrix for a de-identified structural feature block."""
    if block == "U_URL_STRUCTURE":
        names = FEATURE_BLOCKS["U_URL_STRUCTURE"]
        x = np.column_stack([pd.to_numeric(df[n], errors="coerce").fillna(0.0).to_numpy(float) for n in names])
    elif block == "D_DOMAIN_SHAPE_NO_IDENTITY":
        s = df["sender_domain_norm"].fillna("").astype(str)
        r = df["receiver_domain_norm"].fillna("").astype(str)
        names = FEATURE_BLOCKS["D_DOMAIN_SHAPE_NO_IDENTITY"]
        x = np.column_stack([
            s.str.len().to_numpy(float),
            r.str.len().to_numpy(float),
            ((s != "") & (s == r)).astype(float).to_numpy(),
            np.array([sender_url_match(a, b) for a, b in zip(s, df["url_domains_json"])], dtype=float),
        ])
    elif block == "M_MESSAGE_STRUCTURE":
        names = FEATURE_BLOCKS["M_MESSAGE_STRUCTURE"]
        x = np.column_stack([
            pd.to_numeric(df["text_length_chars"], errors="coerce").fillna(0.0).to_numpy(float),
            pd.to_numeric(df["token_count"], errors="coerce").fillna(0.0).to_numpy(float),
            pd.to_numeric(df["attachment_count"], errors="coerce").fillna(0.0).to_numpy(float),
            pd.to_numeric(df["has_attachments"], errors="coerce").fillna(0.0).to_numpy(float),
            np.array([count_listish(v) for v in df["content_types"]], dtype=float),
            np.array([count_listish(v) for v in df["attachment_types"]], dtype=float),
        ])
    else:
        raise KeyError(f"Unknown feature block: {block}")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), list(names)
