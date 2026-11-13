"""CEAS-08 adapter and scoring with the frozen anchor models.

prepare: checks CEAS_08.csv against its Zenodo record (size and MD5), its columns, row count and
class counts; applies the development normalization (subject plus body, empty-text removal,
removal of hash groups with conflicting labels, exact duplicates removed keeping the first);
and screens every message against raw TREC5/TREC6 by canonical hash and by character 5-gram
cosine similarity (>= 0.95). Matches are flagged as removed and are not scored. No model is
loaded during prepare. Writes processed/ceas08_confirmation.parquet and its split file.
score: checks that the frozen anchor models reproduce their saved validation scores on a fixed
sample, then scores the retained messages with A1 and A2 on the three routes (CUDA required) and
writes scores/ceas08_frozen_scores.parquet.

No model is trained, fine-tuned, re-seeded or re-tuned. CEAS-08 labels are used before scoring
only for completeness, class counts and the conflicting-label rule.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import joblib  # noqa: F401  (loaded by the frozen A1 bundles)
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
CEAS = RAW / "CEAS_08.csv"
# CEAS_08.csv in Zenodo record 8339691 (DOI 10.5281/zenodo.8339691, concept DOI
# 10.5281/zenodo.8339690, "Phishing Email Curated Datasets", CC BY 4.0).
CEAS_RECORD = {"zenodo_record": 8339691, "doi": "10.5281/zenodo.8339691", "license": "cc-by-4.0",
               "size": 67902056, "md5": "1f0d59191eec892a709995d48cd8decb"}
MEAJOR_RAW = RAW / "meajor_cleaned_preprocessed.parquet.gzip"
DEV_DATA = DATA_ROOT / "processed" / "meajor_trec56_reconstructed.parquet"
DATA_DIR = DATA_ROOT / "processed"
PRED_DIR = DATA_ROOT / "scores"
RECEIPT_DIR = DATA_ROOT / "results" / "receipts"
ADAPTER = DATA_DIR / "ceas08_confirmation.parquet"
SPLIT = DATA_DIR / "ceas08_confirmation_split.csv"
PRED = PRED_DIR / "ceas08_frozen_scores.parquet"
SIM_THRESHOLD = 0.95
MAX_REMOVED_SHARE = 0.05
EXPECTED = {"rows": 39154, "spam": 21842, "ham": 17312,
            "columns": ["sender", "receiver", "date", "subject", "body", "label", "urls"]}
SPACE_RE = re.compile(r"\s+")
ROUTES = ["final", "fit_trec5", "fit_trec6"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


# Same definitions as in spaphish_adapter_and_scoring.py.
def normalize_space(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).replace("\x00", " ")
    return SPACE_RE.sub(" ", text).strip()


def canonical_text(subject, body) -> str:
    return normalize_space(f"{normalize_space(subject)} {normalize_space(body)}").lower()


def canonical_hash(subject, body) -> str:
    return hashlib.sha256(canonical_text(subject, body).encode("utf-8", errors="ignore")).hexdigest()


def load_anchor_module():
    p = ANCHOR_SCRIPT
    spec = importlib.util.spec_from_file_location("train_anchor_models", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod
# End of the definitions shared with spaphish_adapter_and_scoring.py.


def stop(msg: str, receipt: dict | None = None) -> None:
    if receipt is not None:
        receipt["status"] = "STOPPED_BEFORE_SCORING"
        receipt["stop_reason"] = msg
        write_json(RECEIPT_DIR / "ceas08_adapter.json", receipt)
    raise SystemExit(f"STOP RULE: {msg}")


def prepare() -> None:
    for d in [DATA_DIR, PRED_DIR, RECEIPT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("model stack imported before adapter freeze")
    receipt = {
        "schema": "ceas08_adapter",
        "source_path": "raw/CEAS_08.csv",
    }
    # 1. provenance: Zenodo record 8339691 lists CEAS_08.csv with size and md5
    exp_md5 = CEAS_RECORD["md5"]
    obs_md5 = md5(CEAS)
    obs_sha = sha256(CEAS)
    size = CEAS.stat().st_size
    receipt.update({
        "provenance_record": {"zenodo_record": CEAS_RECORD["zenodo_record"], "doi": CEAS_RECORD["doi"],
                              "license": CEAS_RECORD["license"],
                              "expected_size": CEAS_RECORD["size"], "expected_md5": exp_md5},
        "source_size": size, "source_md5": obs_md5, "source_sha256": obs_sha,
    })
    if obs_md5 != exp_md5 or size != CEAS_RECORD["size"]:
        stop(f"CEAS-08 file hash or size mismatch: md5 {obs_md5} vs {exp_md5}, size {size} vs {CEAS_RECORD['size']}", receipt)

    # 2. columns and label completeness (labels used only for completeness and class counts)
    df = pd.read_csv(CEAS)
    receipt["source_columns"] = list(df.columns)
    if list(df.columns) != EXPECTED["columns"]:
        stop(f"CEAS-08 columns differ: {list(df.columns)}", receipt)
    lab = pd.to_numeric(df["label"], errors="coerce")
    if len(df) != EXPECTED["rows"] or lab.isna().any() or not set(lab.astype(int).unique()) <= {0, 1}:
        stop("CEAS-08 row count or label completeness check failed", receipt)
    counts = lab.astype(int).value_counts().to_dict()
    receipt["source_rows"] = int(len(df))
    receipt["source_label_counts"] = {str(k): int(v) for k, v in sorted(counts.items())}
    if counts.get(1) != EXPECTED["spam"] or counts.get(0) != EXPECTED["ham"]:
        stop(f"CEAS-08 class counts differ: {counts}", receipt)

    # 3. development adapter, unchanged: subject and body normalisation, empty-text removal,
    #    conflicting-label hash groups removed, exact duplicates removed (keep first).
    out = pd.DataFrame()
    out["row_id"] = [f"ceas08_{i:07d}" for i in range(len(df))]
    out["original_index"] = np.arange(len(df), dtype=np.int64)
    out["label"] = lab.astype(int).to_numpy()
    out["subject_clean"] = df["subject"].map(normalize_space)
    out["body_clean"] = df["body"].map(normalize_space)
    out["text_clean"] = (out["subject_clean"] + " " + out["body_clean"]).map(normalize_space)
    out["canonical_hash"] = [canonical_hash(a, b) for a, b in zip(df["subject"], df["body"])]
    out["date_raw"] = df["date"].fillna("").astype(str)
    out["urls_source_indicator"] = pd.to_numeric(df["urls"], errors="coerce").fillna(-1).astype(int)
    empty = out["text_clean"].eq("")
    receipt["dropped_empty_text"] = int(empty.sum())
    out = out.loc[~empty].copy()
    nun = out.groupby("canonical_hash")["label"].nunique()
    conflict = out["canonical_hash"].isin(nun[nun > 1].index)
    receipt["dropped_conflicting_label_rows"] = int(conflict.sum())
    out = out.loc[~conflict].copy()
    dup = out["canonical_hash"].duplicated(keep="first")
    receipt["dropped_internal_exact_duplicates"] = int(dup.sum())
    out = out.loc[~dup].reset_index(drop=True)
    receipt["rows_after_adapter"] = int(len(out))

    # 4. URL residual test feasibility: the seven URL features need URL strings
    uvals = sorted(int(v) for v in out["urls_source_indicator"].unique())
    receipt["url_field_distinct_values"] = uvals
    receipt["url_residual_test"] = {
        "testable": False,
        "reason": "CEAS-08 `urls` field holds a 0/1 indicator only (distinct values "
                  f"{uvals}); URL count, length and subdomain features cannot be rebuilt from it, "
                  "so the seven-feature URL block cannot be reconstructed with the corrected URL parser.",
    } if set(uvals) <= {0, 1} else {"testable": None, "reason": "urls field not binary; inspect before use"}

    # 5. contamination screen against raw TREC5/TREC6, exactly as for SpaPhish
    t0 = time.time()
    raw = pd.read_parquet(MEAJOR_RAW, columns=["source", "subject", "body"])
    raw["source_norm"] = raw["source"].astype(str).str.lower()
    trec = raw[raw["source_norm"].isin(["trec5", "trec6", "trec-05", "trec-06", "trec05", "trec06"])].copy()
    trec_text = [normalize_space(f"{normalize_space(a)} {normalize_space(b)}") for a, b in zip(trec["subject"], trec["body"])]
    trec_hash = {hashlib.sha256(t.lower().encode("utf-8", errors="ignore")).hexdigest() for t in trec_text}
    exact = out["canonical_hash"].isin(trec_hash).to_numpy()
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(5, 5), n_features=2**18,
                           alternate_sign=False, norm="l2", lowercase=True)
    trec_x = hv.transform(trec_text)
    cx = hv.transform(out["text_clean"].tolist())
    nn = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="cosine", n_jobs=-1)
    nn.fit(trec_x)
    dist, idx = nn.kneighbors(cx, return_distance=True)
    sim = 1.0 - dist[:, 0]
    near = sim >= SIM_THRESHOLD
    removed = exact | near
    out["exact_trec_hash_match"] = exact
    out["nearest_trec_similarity_char5"] = sim
    out["contamination_removed"] = removed
    receipt.update({
        "raw_trec_rows_screened": int(len(trec)),
        "raw_trec_source_counts": {str(k): int(v) for k, v in trec["source_norm"].value_counts().sort_index().items()},
        "exact_hash_overlap_with_raw_trec5_trec6": int(exact.sum()),
        "near_duplicate_similarity_threshold": SIM_THRESHOLD,
        "near_duplicate_count_ge_0_95": int(near.sum()),
        "near_duplicate_count_ge_0_99": int((sim >= 0.99).sum()),
        "removed_exact_or_near": int(removed.sum()),
        "removed_share": float(removed.mean()),
        "removed_label_counts": {str(k): int(v) for k, v in out.loc[removed, "label"].value_counts().sort_index().items()},
        "nearest_similarity_max": float(sim.max()),
        "nearest_similarity_p99": float(np.quantile(sim, 0.99)),
        "nearest_similarity_median": float(np.median(sim)),
        "screen_wall_seconds": time.time() - t0,
    })
    if removed.mean() > MAX_REMOVED_SHARE:
        out.to_parquet(ADAPTER, index=False)
        stop(f"contamination screen removed {removed.mean():.4f} > {MAX_REMOVED_SHARE} of messages; inspect the screen before scoring", receipt)

    eval_rows = out.loc[~removed]
    receipt["evaluation_rows"] = int(len(eval_rows))
    receipt["evaluation_label_counts"] = {str(k): int(v) for k, v in eval_rows["label"].value_counts().sort_index().items()}
    out.to_parquet(ADAPTER, index=False)
    split = out[["row_id", "label", "canonical_hash"]].copy()
    split["split"] = np.where(removed, "removed_contamination", "test")
    split.to_csv(SPLIT, index=False)
    receipt.update({
        "status": "FROZEN_BEFORE_MODEL_SCORING",
        "adapter_path": "processed/ceas08_confirmation.parquet",
        "adapter_sha256": sha256(ADAPTER),
        "split_path": "processed/ceas08_confirmation_split.csv",
        "split_sha256": sha256(SPLIT),
        "frozen_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_stack_imported_before_freeze": bool("torch" in sys.modules or "transformers" in sys.modules),
        "normalization": "subject plus body, NUL to space, whitespace collapse, trim; lowercase only for canonical hash",
        "adapter_rule_source": "development adapter rule of preprocess_corpora.normalize_dataset: empty-text removal, conflicting-label hash groups removed, exact duplicates removed keeping first",
        "label_use_before_scoring": "label checked only for completeness and class counts (and the development adapter conflicting-label check, which removed the rows counted above); no label used for model fitting, threshold selection, calibration, seed selection, subgroup definition, or feature selection",
        "url_safety": "URLs never parsed, resolved or fetched",
    })
    write_json(RECEIPT_DIR / "ceas08_adapter.json", receipt)
    print(json.dumps(receipt, indent=2))


def hash_tree(path: Path) -> dict:
    files = {}
    for p in sorted(path.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(path)).replace("\\", "/")] = sha256(p)
    payload = json.dumps(files, sort_keys=True).encode("utf-8")
    return {"tree_sha256": hashlib.sha256(payload).hexdigest(), "files": files}


ROUTE_PATHS = {
    "final": {"a1": RUN / "final_models" / "A1_TFIDF_LR" / "tfidf_lr.joblib", "a2": RUN / "final_models" / "A2_MDISTILBERT"},
    "fit_trec5": {"a1": RUN / "fit_trec5__val_trec6" / "A1_TFIDF_LR" / "tfidf_lr.joblib", "a2": RUN / "fit_trec5__val_trec6" / "A2_MDISTILBERT"},
    "fit_trec6": {"a1": RUN / "fit_trec6__val_trec5" / "A1_TFIDF_LR" / "tfidf_lr.joblib", "a2": RUN / "fit_trec6__val_trec5" / "A2_MDISTILBERT"},
}


def runtime_compatibility(anchor, config) -> dict:
    """Rescore 64 rows per class of each development validation fold with the saved models and
    compare with the saved scores (A1 within 1e-12, A2 within 1e-7)."""
    dev = pd.read_parquet(DEV_DATA)
    rep = {"status": "PASS", "sample_rule": "64 deterministic evenly spaced rows per class per validation source", "routes": {}}
    for name, valsrc, root in [("fit_trec5", "trec6", RUN / "fit_trec5__val_trec6"), ("fit_trec6", "trec5", RUN / "fit_trec6__val_trec5")]:
        saved_a2 = pd.read_parquet(root / "A2_MDISTILBERT" / "ensemble_validation.parquet")
        saved_a1 = pd.read_parquet(root / "A1_TFIDF_LR" / "tfidf_lr_validation.parquet")
        parts = []
        for y in [0, 1]:
            g = saved_a2[saved_a2.label.eq(y)].sort_values("row_id")
            parts.append(g.iloc[np.linspace(0, len(g) - 1, 64, dtype=int)])
        s = pd.concat(parts).sort_values("row_id").reset_index(drop=True)
        rows = dev[dev.source.eq(valsrc)].merge(s[["row_id"]], on="row_id", validate="one_to_one")
        rows = rows.set_index("row_id").loc[s.row_id].reset_index()
        now_a1 = anchor.score_tfidf(root / "A1_TFIDF_LR" / "tfidf_lr.joblib", rows)
        now_a2 = anchor.score_transformer_ensemble(config, root / "A2_MDISTILBERT", rows)
        old_a1 = s[["row_id"]].merge(saved_a1, on="row_id", validate="one_to_one")["score"].to_numpy(float)
        old_a2 = s["score"].to_numpy(float)
        d1, d2 = np.abs(now_a1 - old_a1), np.abs(now_a2 - old_a2)
        r = {"rows": int(len(s)), "a1_max_abs_diff": float(d1.max()), "a2_max_abs_diff": float(d2.max()),
             "a1_exact_count": int((d1 == 0).sum()), "a2_exact_count": int((d2 == 0).sum())}
        rep["routes"][name] = r
        if r["a1_max_abs_diff"] > 1e-12 or r["a2_max_abs_diff"] > 1e-7:
            rep["status"] = "FAIL"
    return rep


def score() -> None:
    receipt_path = RECEIPT_DIR / "ceas08_adapter.json"
    if not ADAPTER.exists() or not receipt_path.exists():
        raise RuntimeError("Run prepare before score")
    frozen = json.loads(receipt_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN_BEFORE_MODEL_SCORING" or sha256(ADAPTER) != frozen.get("adapter_sha256") \
            or sha256(SPLIT) != frozen.get("split_sha256"):
        raise RuntimeError("Adapter bytes changed after freeze or adapter not frozen")
    if PRED.exists():
        raise RuntimeError(f"{PRED} already exists; refusing to overwrite")
    data = pd.read_parquet(ADAPTER)
    data = data.loc[~data["contamination_removed"]].reset_index(drop=True)
    if len(data) != frozen["evaluation_rows"]:
        raise RuntimeError("evaluation row count differs from freeze receipt")
    model_data = data[["row_id", "text_clean"]].copy()
    anchor = load_anchor_module()
    config = anchor.load_config()
    t_start = time.time()
    model_receipt = {
        "schema": "ceas08_scoring",
        "adapter_sha256": sha256(ADAPTER),
        "split_sha256": sha256(SPLIT),
        "config_sha256": sha256(CONFIG_PATH),
        "anchor_module_sha256": sha256(ANCHOR_SCRIPT),
        "routes": {},
        "runtime": {"python": platform.python_version()},
        "started_unix": t_start,
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    import sklearn
    import torch
    import transformers
    model_receipt["runtime"].update({
        "torch": torch.__version__, "transformers": transformers.__version__, "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; A2 scoring runs on a CUDA device")
    compat = runtime_compatibility(anchor, config)
    model_receipt["runtime_compatibility"] = compat
    print(json.dumps({"runtime_compatibility": compat}, indent=2), flush=True)
    if compat["status"] != "PASS":
        write_json(RECEIPT_DIR / "ceas08_scoring.json", model_receipt)
        raise RuntimeError("runtime compatibility check failed")

    pred = pd.DataFrame({"row_id": data["row_id"]})
    for name in ROUTES:
        paths = ROUTE_PATHS[name]
        if not paths["a1"].exists() or not paths["a2"].exists():
            raise RuntimeError(f"Missing frozen model route {name}")
        t0 = time.time()
        a1 = anchor.score_tfidf(paths["a1"], model_data)
        t1 = time.time()
        a2 = anchor.score_transformer_ensemble(config, paths["a2"], model_data)
        t2 = time.time()
        if len(a1) != len(data) or len(a2) != len(data):
            raise RuntimeError("Prediction row count mismatch")
        if not np.isfinite(a1).all() or not np.isfinite(a2).all():
            raise RuntimeError("Non-finite prediction")
        if ((a1 < 0) | (a1 > 1)).any() or ((a2 < 0) | (a2 > 1)).any():
            raise RuntimeError("Prediction outside [0,1]")
        pred[f"{name}_a1"] = a1
        pred[f"{name}_a2"] = a2
        model_receipt["routes"][name] = {
            "a1_path": paths["a1"].relative_to(DATA_ROOT).as_posix(), "a1_sha256": sha256(paths["a1"]),
            "a2_root": paths["a2"].relative_to(DATA_ROOT).as_posix(), "a2_tree": hash_tree(paths["a2"]),
            "a1_seconds": t1 - t0, "a2_seconds": t2 - t1,
            "a2_dtype": str(np.asarray(a2).dtype),
        }
        print(json.dumps({"route": name, "a1_s": t1 - t0, "a2_s": t2 - t1}), flush=True)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(PRED, index=False)
    model_receipt["predictions_path"] = "scores/ceas08_frozen_scores.parquet"
    model_receipt["predictions_sha256"] = sha256(PRED)
    model_receipt["rows"] = int(len(pred))
    model_receipt["finished_unix"] = time.time()
    model_receipt["finished_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    model_receipt["wall_seconds"] = model_receipt["finished_unix"] - t_start
    write_json(RECEIPT_DIR / "ceas08_scoring.json", model_receipt)
    print(json.dumps({"status": "PASS", "rows": len(pred), "predictions": str(PRED), "sha256": sha256(PRED)}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["prepare", "score"])
    args = ap.parse_args()
    {"prepare": prepare, "score": score}[args.action]()


if __name__ == "__main__":
    main()
