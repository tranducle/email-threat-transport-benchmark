"""SpaPhish v5 adapter and scoring with the frozen anchor models.

prepare: verifies the SHA-256 of the SpaPhish CSV, maps subject and body to the development
text interface (NUL to space, whitespace collapse, trim), derives the canonical hash, assigns
the temporal group from the message date (pre_2023, 2023_to_2025, date_missing), computes the
seven URL features, and screens every message against raw TREC5/TREC6 by canonical hash and
character 5-gram cosine similarity. Labels are checked only for completeness and class counts.
Writes processed/spaphish_confirmation.parquet and its split file.
score: scores the adapter table with A1 and A2 on the three routes (final, fit_trec5,
fit_trec6) and writes scores/spaphish_frozen_scores.parquet.

The URL parser in this file returns no URLs for SpaPhish's "[u1],[u2]" format, so all seven URL
features are zero in spaphish_confirmation.parquet. rebuild_spaphish_url_features.py rebuilds
them with url_parser.py; the experiments read the rebuilt table.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
DATA_ROOT = Path(os.environ.get("THRESHOLD_TRANSPORT_DATA", REPO_ROOT / "data"))
RAW = DATA_ROOT / "raw"
ANCHOR_SCRIPT = REPO_ROOT / "threshold_transport" / "training" / "train_anchor_models.py"
RUN = DATA_ROOT / "models" / "anchors"
CONFIG_PATH = REPO_ROOT / "threshold_transport" / "configs" / "anchor_models.json"
SPAPHISH = RAW / "SpaPhish_v5" / "Spaphish dataset - DiB.csv"
# SHA-256 of "Spaphish dataset - DiB.csv" in SpaPhish v5 (Mendeley Data, DOI 10.17632/hz2d6gz7pc.5)
SPAPHISH_SHA256 = "656b2245d58da72d640680e5c2a168673a130b38607f2a427c773bbb167e995e"
MEAJOR_RAW = RAW / "meajor_cleaned_preprocessed.parquet.gzip"
DATA_DIR = DATA_ROOT / "processed"
PRED_DIR = DATA_ROOT / "scores"
RECEIPT_DIR = DATA_ROOT / "results" / "receipts"
URL_RE = re.compile(r"https?://[^\s,'\"<>\\)\\]]+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def normalize_space(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).replace("\x00", " ")
    return SPACE_RE.sub(" ", text).strip()


def canonical_text(subject, body) -> str:
    return normalize_space(f"{normalize_space(subject)} {normalize_space(body)}").lower()


def canonical_hash(subject, body) -> str:
    return hashlib.sha256(canonical_text(subject, body).encode("utf-8", errors="ignore")).hexdigest()


def parse_urls(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "[]"}:
        return []
    items = []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            v = ast.literal_eval(raw)
            if isinstance(v, (list, tuple)):
                items = [str(x) for x in v if x]
        except Exception:
            items = []
    candidates = items or URL_RE.findall(raw)
    out = []
    for x in candidates:
        x = x.strip().strip(",.'\"<>[]()")
        if x.startswith(("http://", "https://")):
            out.append(x)
    return out


def clean_domain(v: str) -> str:
    v = str(v or "").strip().lower().rstrip(".")
    if "@" in v:
        v = v.rsplit("@", 1)[-1]
    if ":" in v:
        v = v.split(":", 1)[0]
    return v


def subdomain_depth(domain: str) -> int:
    d = clean_domain(domain)
    if not d:
        return 0
    labels = [x for x in d.split(".") if x]
    return max(0, len(labels) - 2)


def url_features(value) -> dict:
    urls = parse_urls(value)
    lengths = [len(u) for u in urls]
    domains = []
    for u in urls:
        try:
            domains.append(urlparse(u).netloc)
        except Exception:
            pass
    depths = [subdomain_depth(d) for d in domains if d]
    return {
        "url_indicator": int(bool(urls)),
        "url_count_raw": len(urls),
        "url_count_parsed": len(urls),
        "url_length_max": float(max(lengths) if lengths else 0.0),
        "url_length_avg": float(np.mean(lengths) if lengths else 0.0),
        "url_subdom_max": float(max(depths) if depths else 0.0),
        "url_subdom_avg": float(np.mean(depths) if depths else 0.0),
    }


def load_anchor_module():
    p = ANCHOR_SCRIPT
    spec = importlib.util.spec_from_file_location("train_anchor_models", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def prepare() -> None:
    for d in [DATA_DIR, PRED_DIR, RECEIPT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    expected_sha = SPAPHISH_SHA256
    actual_sha = sha256(SPAPHISH)
    if actual_sha != expected_sha:
        raise RuntimeError(f"SpaPhish hash mismatch: {actual_sha} != {expected_sha}")

    df = pd.read_csv(SPAPHISH, encoding="utf-8-sig")
    required = {"subject", "body", "date", "urls", "url_count", "Label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing SpaPhish columns: {missing}")

    out = pd.DataFrame()
    out["row_id"] = [f"spaphish_v5_{i:07d}" for i in range(len(df))]
    out["label"] = pd.to_numeric(df["Label"], errors="coerce")
    if out["label"].isna().any() or set(out["label"].astype(int).unique()) != {0, 1}:
        raise RuntimeError("SpaPhish labels are not complete binary 0/1")
    out["label"] = out["label"].astype(int)
    out["subject_clean"] = df["subject"].map(normalize_space)
    out["body_clean"] = df["body"].map(normalize_space)
    out["text_clean"] = (out["subject_clean"] + " " + out["body_clean"]).map(normalize_space)
    if out["text_clean"].eq("").any():
        raise RuntimeError("Empty text after frozen normalization")
    out["canonical_hash"] = [canonical_hash(a, b) for a, b in zip(df["subject"], df["body"])]
    if out["canonical_hash"].duplicated().any():
        raise RuntimeError("Duplicate canonical hash inside SpaPhish")
    out["date_raw"] = df["date"].fillna("").astype(str)
    out["date_parsed"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    out["temporal_group"] = np.where(
        out["date_parsed"].isna(),
        "date_missing",
        np.where(out["date_parsed"].dt.year >= 2023, "2023_to_2025", "pre_2023"),
    )
    out["urls_raw"] = df["urls"].fillna("").astype(str)
    out["url_count_source"] = pd.to_numeric(df["url_count"], errors="coerce").fillna(0).astype(int)
    out["attachment_count"] = pd.to_numeric(df.get("attachments_count", 0), errors="coerce").fillna(0).astype(int)
    out["attachment_types"] = df.get("attachment_types", pd.Series([""] * len(df))).fillna("").astype(str)
    u = pd.DataFrame([url_features(x) for x in out["urls_raw"]])
    for col in u.columns:
        out[col] = u[col].to_numpy()

    raw = pd.read_parquet(MEAJOR_RAW, columns=["source", "subject", "body"])
    raw["source_norm"] = raw["source"].astype(str).str.lower()
    trec = raw[raw["source_norm"].isin(["trec5", "trec6", "trec-05", "trec-06", "trec05", "trec06"])].copy()
    trec_text = [normalize_space(f"{normalize_space(a)} {normalize_space(b)}") for a, b in zip(trec["subject"], trec["body"])]
    trec_hash = {hashlib.sha256(t.lower().encode("utf-8", errors="ignore")).hexdigest() for t in trec_text}
    exact_overlap = int(out["canonical_hash"].isin(trec_hash).sum())

    hv = HashingVectorizer(
        analyzer="char_wb", ngram_range=(5, 5), n_features=2**18,
        alternate_sign=False, norm="l2", lowercase=True
    )
    trec_x = hv.transform(trec_text)
    sp_x = hv.transform(out["text_clean"].tolist())
    nn = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="cosine", n_jobs=-1)
    nn.fit(trec_x)
    dist, idx = nn.kneighbors(sp_x, return_distance=True)
    sim = 1.0 - dist[:, 0]
    out["nearest_trec_similarity_char5"] = sim
    near_095 = int((sim >= 0.95).sum())
    near_099 = int((sim >= 0.99).sum())

    adapter_path = DATA_DIR / "spaphish_confirmation.parquet"
    out.to_parquet(adapter_path, index=False)
    split = out[["row_id", "label", "canonical_hash"]].copy()
    split["split"] = "test"
    split_path = DATA_DIR / "spaphish_confirmation_split.csv"
    split.to_csv(split_path, index=False)

    receipt = {
        "schema": "spaphish_adapter",
        "status": "FROZEN_BEFORE_MODEL_SCORING",
        "source_path": "raw/SpaPhish_v5/Spaphish dataset - DiB.csv",
        "source_sha256": actual_sha,
        "adapter_sha256": sha256(adapter_path),
        "split_sha256": sha256(split_path),
        "rows": int(len(out)),
        "label_counts": {str(k): int(v) for k, v in out["label"].value_counts().sort_index().items()},
        "missing_subject": int((out["subject_clean"] == "").sum()),
        "missing_body": int((out["body_clean"] == "").sum()),
        "missing_date": int(out["date_parsed"].isna().sum()),
        "temporal_counts": {str(k): int(v) for k, v in out["temporal_group"].value_counts().items()},
        "unique_canonical_hashes": int(out["canonical_hash"].nunique()),
        "raw_trec_rows_screened": int(len(trec)),
        "exact_hash_overlap_with_raw_trec5_trec6": exact_overlap,
        "near_duplicate_similarity_threshold": 0.95,
        "near_duplicate_count_ge_0_95": near_095,
        "near_duplicate_count_ge_0_99": near_099,
        "nearest_similarity_max": float(sim.max()),
        "nearest_similarity_p99": float(np.quantile(sim, 0.99)),
        "nearest_similarity_median": float(np.median(sim)),
        "normalization": "subject plus body, NUL to space, whitespace collapse, trim; lowercase only for canonical hash",
        "label_use_before_scoring": "label checked only for completeness and class counts; no label used for model fitting, threshold selection, calibration, seed selection, subgroup definition, or feature selection",
        "url_safety": "URLs parsed as strings only; no URL resolution or network access",
    }
    write_json(RECEIPT_DIR / "spaphish_adapter.json", receipt)
    print(json.dumps(receipt, indent=2))


def hash_tree(path: Path) -> dict:
    files = {}
    for p in sorted(path.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(path)).replace("\\", "/")] = sha256(p)
    payload = json.dumps(files, sort_keys=True).encode("utf-8")
    return {"tree_sha256": hashlib.sha256(payload).hexdigest(), "files": files}


def score() -> None:
    adapter = DATA_DIR / "spaphish_confirmation.parquet"
    receipt_path = RECEIPT_DIR / "spaphish_adapter.json"
    if not adapter.exists() or not receipt_path.exists():
        raise RuntimeError("Run prepare before score")
    frozen = json.loads(receipt_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN_BEFORE_MODEL_SCORING" or sha256(adapter) != frozen.get("adapter_sha256"):
        raise RuntimeError("Adapter bytes changed after freeze")
    data = pd.read_parquet(adapter)
    model_data = data[["row_id", "text_clean"]].copy()
    anchor = load_anchor_module()
    config = anchor.load_config()

    routes = {
        "final": {
            "a1": RUN / "final_models" / "A1_TFIDF_LR" / "tfidf_lr.joblib",
            "a2": RUN / "final_models" / "A2_MDISTILBERT",
        },
        "fit_trec5": {
            "a1": RUN / "fit_trec5__val_trec6" / "A1_TFIDF_LR" / "tfidf_lr.joblib",
            "a2": RUN / "fit_trec5__val_trec6" / "A2_MDISTILBERT",
        },
        "fit_trec6": {
            "a1": RUN / "fit_trec6__val_trec5" / "A1_TFIDF_LR" / "tfidf_lr.joblib",
            "a2": RUN / "fit_trec6__val_trec5" / "A2_MDISTILBERT",
        },
    }
    pred = pd.DataFrame({"row_id": data["row_id"]})
    model_receipt = {
        "schema": "spaphish_scoring",
        "adapter_sha256": sha256(adapter),
        "config_sha256": sha256(CONFIG_PATH),
        "routes": {},
        "runtime": {"python": platform.python_version()},
        "started_unix": time.time(),
    }
    try:
        import torch
        import transformers
        model_receipt["runtime"]["torch"] = torch.__version__
        model_receipt["runtime"]["transformers"] = transformers.__version__
        model_receipt["runtime"]["cuda_available"] = bool(torch.cuda.is_available())
        model_receipt["runtime"]["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:
        model_receipt["runtime"]["stack_error"] = repr(e)

    for name, paths in routes.items():
        if not paths["a1"].exists() or not paths["a2"].exists():
            raise RuntimeError(f"Missing frozen model route {name}")
        a1 = anchor.score_tfidf(paths["a1"], model_data)
        a2 = anchor.score_transformer_ensemble(config, paths["a2"], model_data)
        if len(a1) != len(data) or len(a2) != len(data):
            raise RuntimeError("Prediction row count mismatch")
        if not np.isfinite(a1).all() or not np.isfinite(a2).all():
            raise RuntimeError("Non-finite prediction")
        if ((a1 < 0) | (a1 > 1)).any() or ((a2 < 0) | (a2 > 1)).any():
            raise RuntimeError("Prediction outside [0,1]")
        pred[f"{name}_a1"] = a1
        pred[f"{name}_a2"] = a2
        model_receipt["routes"][name] = {
            "a1_path": paths["a1"].relative_to(DATA_ROOT).as_posix(),
            "a1_sha256": sha256(paths["a1"]),
            "a2_root": paths["a2"].relative_to(DATA_ROOT).as_posix(),
            "a2_tree": hash_tree(paths["a2"]),
        }

    pred_path = PRED_DIR / "spaphish_frozen_scores.parquet"
    pred.to_parquet(pred_path, index=False)
    model_receipt["predictions_sha256"] = sha256(pred_path)
    model_receipt["rows"] = int(len(pred))
    model_receipt["finished_unix"] = time.time()
    write_json(RECEIPT_DIR / "spaphish_scoring.json", model_receipt)
    print(json.dumps({"status": "PASS", "rows": len(pred), "predictions": str(pred_path), "sha256": sha256(pred_path)}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["prepare", "score"])
    args = ap.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        score()


if __name__ == "__main__":
    main()
