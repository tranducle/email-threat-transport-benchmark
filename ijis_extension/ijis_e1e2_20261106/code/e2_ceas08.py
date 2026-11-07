"""E2: CEAS-08 as a second external corpus, frozen scoring only (IJIS package, 2026-11-06).

Adapted copy of top_q1_external_confirmation_20261105/code/external_confirmation.py (read only,
not modified). Changes: input corpus (CEAS-08), output paths (this folder), contamination matches
are removed before scoring (plan item E2 step 2), internal exact duplicates are dropped with the
development adapter rule (keep first canonical hash), stop rules of EXPERIMENT_PLAN_IJIS.md, an
AUROC column requested by the caller, and per-model bootstrap intervals drawn from the same paired
class-stratified resamples as the A2 minus A1 differences.

Actions: prepare (adapter freeze, no model loaded), score (GPU), analyze.
No model is trained, fine-tuned, re-seeded or re-tuned. CEAS-08 labels are used before scoring only
for completeness and class counts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import re
import sys
import time
from pathlib import Path

import joblib  # noqa: F401  (loaded by the frozen A1 bundles)
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
EXP = OUT.parent
PAPER = EXP.parent
RAW = PAPER / "DATASET" / "raw"
META = PAPER / "DATASET" / "metadata"
RB = EXP / "rebaseline_rb_g1"
RUN = RB / "runs" / "RB_G1_20261023_V1"
EXT = EXP / "top_q1_external_confirmation_20261105"
FREEZE_PATH = PAPER / "8_Project_Management" / "PROJECT_REBASELINE_20261022" / "17_RB_G1_SEMANTIC_ANCHOR_FREEZE.json"
CEAS = RAW / "CEAS_08.csv"
CEAS_META = META / "zenodo_8339691.json"
MEAJOR_RAW = RAW / "meajor_cleaned_preprocessed.parquet.gzip"
DEV_DATA = EXT / "data" / "meajor_trec56_reconstructed.parquet"
DATA_DIR = OUT / "data"
PRED_DIR = OUT / "predictions"
RESULT_DIR = OUT / "results"
RECEIPT_DIR = OUT / "receipts"
ADAPTER = DATA_DIR / "ceas08_confirmation.parquet"
SPLIT = DATA_DIR / "ceas08_confirmation_split.csv"
PRED = PRED_DIR / "ceas08_frozen_scores.parquet"
BOOT_SEED = 20261105  # same bootstrap seed as the SpaPhish external confirmation
BOOT_REPS = 1000
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


# ---- verbatim from the 2026-11-05 adapter ------------------------------------------------------
def normalize_space(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).replace("\x00", " ")
    return SPACE_RE.sub(" ", text).strip()


def canonical_text(subject, body) -> str:
    return normalize_space(f"{normalize_space(subject)} {normalize_space(body)}").lower()


def canonical_hash(subject, body) -> str:
    return hashlib.sha256(canonical_text(subject, body).encode("utf-8", errors="ignore")).hexdigest()


def load_rb_module():
    p = RB / "rb_g1_semantic_anchor.py"
    spec = importlib.util.spec_from_file_location("rb_g1_semantic_anchor_ijis", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod
# -------------------------------------------------------------------------------------------------


def stop(msg: str, receipt: dict | None = None) -> None:
    if receipt is not None:
        receipt["status"] = "STOPPED_BEFORE_SCORING"
        receipt["stop_reason"] = msg
        write_json(RECEIPT_DIR / "adapter_freeze.json", receipt)
    raise SystemExit(f"STOP RULE: {msg}")


def prepare() -> None:
    for d in [DATA_DIR, PRED_DIR, RESULT_DIR, RECEIPT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("model stack imported before adapter freeze")
    receipt = {
        "schema": "P8_CEAS08_ADAPTER_FREEZE_V1",
        "study_id": "P8-IJIS-E2-20261106",
        "source_path": str(CEAS),
        "metadata_path": str(CEAS_META),
    }
    # 1. provenance: Zenodo record 8339691 lists CEAS_08.csv with size and md5
    meta = json.loads(CEAS_META.read_text(encoding="utf-8"))
    rec = next((f for f in meta.get("files", []) if f.get("key") == "CEAS_08.csv"), None)
    if rec is None:
        stop("CEAS_08.csv not listed in the provenance manifest", receipt)
    exp_md5 = str(rec["checksum"]).split(":", 1)[1]
    obs_md5 = md5(CEAS)
    obs_sha = sha256(CEAS)
    size = CEAS.stat().st_size
    receipt.update({
        "provenance_record": {"zenodo_record": meta.get("id"), "doi": meta.get("doi"),
                              "license": meta.get("metadata", {}).get("license", {}).get("id"),
                              "expected_size": rec.get("size"), "expected_md5": exp_md5},
        "source_size": size, "source_md5": obs_md5, "source_sha256": obs_sha,
        "metadata_sha256": sha256(CEAS_META),
    })
    if obs_md5 != exp_md5 or size != rec.get("size"):
        stop(f"CEAS-08 file hash or size mismatch: md5 {obs_md5} vs {exp_md5}, size {size} vs {rec.get('size')}", receipt)

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
                  "so the seven-feature URL block cannot be reconstructed faithfully with the E-FIX parser.",
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
        stop(f"contamination screen removed {removed.mean():.4f} > {MAX_REMOVED_SHARE} of messages; rerun the gate", receipt)

    eval_rows = out.loc[~removed]
    receipt["evaluation_rows"] = int(len(eval_rows))
    receipt["evaluation_label_counts"] = {str(k): int(v) for k, v in eval_rows["label"].value_counts().sort_index().items()}
    out.to_parquet(ADAPTER, index=False)
    split = out[["row_id", "label", "canonical_hash"]].copy()
    split["split"] = np.where(removed, "removed_contamination", "test")
    split.to_csv(SPLIT, index=False)
    receipt.update({
        "status": "FROZEN_BEFORE_MODEL_SCORING",
        "adapter_path": str(ADAPTER),
        "adapter_sha256": sha256(ADAPTER),
        "split_path": str(SPLIT),
        "split_sha256": sha256(SPLIT),
        "frozen_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_stack_imported_before_freeze": bool("torch" in sys.modules or "transformers" in sys.modules),
        "normalization": "subject plus body, NUL to space, whitespace collapse, trim; lowercase only for canonical hash",
        "adapter_rule_source": "historical development adapter preprocess_email_data.normalize_dataset: empty-text removal, conflicting-label hash groups removed, exact duplicates removed keeping first",
        "label_use_before_scoring": "label checked only for completeness and class counts (and the development adapter conflicting-label check, which removed the rows counted above); no label used for model fitting, threshold selection, calibration, seed selection, subgroup definition, or feature selection",
        "url_safety": "URLs never parsed, resolved or fetched",
    })
    write_json(RECEIPT_DIR / "adapter_freeze.json", receipt)
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


def runtime_compatibility(rb, freeze) -> dict:
    """Same check as top_q1_external_confirmation_20261105/code/runtime_compatibility.py."""
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
        now_a1 = rb.score_tfidf(root / "A1_TFIDF_LR" / "tfidf_lr.joblib", rows)
        now_a2 = rb.score_transformer_ensemble(freeze, root / "A2_MDISTILBERT", rows)
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
    receipt_path = RECEIPT_DIR / "adapter_freeze.json"
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
    rb = load_rb_module()
    freeze = rb.load_freeze()
    t_start = time.time()
    model_receipt = {
        "schema": "P8_CEAS08_SCORING_MODELS_V1",
        "study_id": "P8-IJIS-E2-20261106",
        "adapter_sha256": sha256(ADAPTER),
        "split_sha256": sha256(SPLIT),
        "freeze_sha256": sha256(FREEZE_PATH),
        "rb_module_sha256": sha256(RB / "rb_g1_semantic_anchor.py"),
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
        raise RuntimeError("CUDA not available; the plan scores on the RTX 3090")
    compat = runtime_compatibility(rb, freeze)
    model_receipt["runtime_compatibility"] = compat
    print(json.dumps({"runtime_compatibility": compat}, indent=2), flush=True)
    if compat["status"] != "PASS":
        write_json(RECEIPT_DIR / "scoring_receipt.json", model_receipt)
        raise RuntimeError("runtime compatibility check failed")

    pred = pd.DataFrame({"row_id": data["row_id"]})
    for name in ROUTES:
        paths = ROUTE_PATHS[name]
        if not paths["a1"].exists() or not paths["a2"].exists():
            raise RuntimeError(f"Missing frozen model route {name}")
        t0 = time.time()
        a1 = rb.score_tfidf(paths["a1"], model_data)
        t1 = time.time()
        a2 = rb.score_transformer_ensemble(freeze, paths["a2"], model_data)
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
            "a1_path": str(paths["a1"]), "a1_sha256": sha256(paths["a1"]),
            "a2_root": str(paths["a2"]), "a2_tree": hash_tree(paths["a2"]),
            "a1_seconds": t1 - t0, "a2_seconds": t2 - t1,
            "a2_dtype": str(np.asarray(a2).dtype),
        }
        print(json.dumps({"route": name, "a1_s": t1 - t0, "a2_s": t2 - t1}), flush=True)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(PRED, index=False)
    model_receipt["predictions_path"] = str(PRED)
    model_receipt["predictions_sha256"] = sha256(PRED)
    model_receipt["rows"] = int(len(pred))
    model_receipt["finished_unix"] = time.time()
    model_receipt["finished_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    model_receipt["wall_seconds"] = model_receipt["finished_unix"] - t_start
    write_json(RECEIPT_DIR / "scoring_receipt.json", model_receipt)
    print(json.dumps({"status": "PASS", "rows": len(pred), "predictions": str(PRED), "sha256": sha256(PRED)}, indent=2))


# ---- metrics: verbatim from the 2026-11-05 code, plus AUROC (caller request) -------------------
def ece(y, p, bins=15) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.minimum(np.digitize(p, edges[1:-1], right=False), bins - 1)
    total = len(y)
    acc = 0.0
    for b in range(bins):
        mask = ids == b
        if not np.any(mask):
            continue
        acc += (mask.sum() / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(acc)


def metric_block(y, p, threshold) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    eps = 1e-7
    return {
        "n": int(len(y)), "positives": int(y.sum()), "negatives": int((y == 0).sum()),
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, eps, 1 - eps), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece10": ece(y, p, 10), "ece15": ece(y, p, 15), "ece20": ece(y, p, 20),
        "threshold": float(threshold), "recall": float(tp / max(1, tp + fn)),
        "fpr": float(fp / max(1, fp + tn)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


BOOT_KEYS = ["auroc", "average_precision", "log_loss", "brier", "ece15", "recall", "fpr"]


def paired_bootstrap(y, a1, a2, threshold_a1, threshold_a2, reps=BOOT_REPS, seed=BOOT_SEED) -> dict:
    y = np.asarray(y, dtype=int)
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    vals = {m: {k: [] for k in BOOT_KEYS} for m in ("a1", "a2", "diff")}
    for _ in range(reps):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        yy = y[idx]
        m1 = metric_block(yy, a1[idx], threshold_a1)
        m2 = metric_block(yy, a2[idx], threshold_a2)
        for k in BOOT_KEYS:
            vals["a1"][k].append(m1[k])
            vals["a2"][k].append(m2[k])
            vals["diff"][k].append(m2[k] - m1[k])
    out = {"reps": reps, "seed": seed, "class_stratified": True, "paired": True,
           "per_model_ci95": {"a1": {}, "a2": {}}, "differences": {}}
    for m in ("a1", "a2"):
        for k in BOOT_KEYS:
            arr = np.asarray(vals[m][k], dtype=float)
            out["per_model_ci95"][m][k] = [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]
    for k in BOOT_KEYS:
        arr = np.asarray(vals["diff"][k], dtype=float)
        out["differences"][k] = {"a2_minus_a1_mean": float(arr.mean()),
                                 "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]}
    return out


def dev_fold_paths() -> dict:
    return {
        "trec6": {"a1": RUN / "fit_trec5__val_trec6" / "A1_TFIDF_LR" / "tfidf_lr_validation.parquet",
                  "a2": RUN / "fit_trec5__val_trec6" / "A2_MDISTILBERT" / "ensemble_validation.parquet"},
        "trec5": {"a1": RUN / "fit_trec6__val_trec5" / "A1_TFIDF_LR" / "tfidf_lr_validation.parquet",
                  "a2": RUN / "fit_trec6__val_trec5" / "A2_MDISTILBERT" / "ensemble_validation.parquet"},
    }


def thresholds_for_caps(rb, caps):
    out = {}
    for cap in caps:
        per = {}
        for src, d in dev_fold_paths().items():
            per[src] = {}
            for model, p in d.items():
                f = pd.read_parquet(p)
                per[src][model] = float(rb.threshold_at_max_fpr(f["label"].to_numpy(), f["score"].to_numpy(dtype=np.float64), max_fpr=cap))
        out[str(cap)] = {"per_source": per,
                         "common": {"a1": max(per["trec5"]["a1"], per["trec6"]["a1"]),
                                    "a2": max(per["trec5"]["a2"], per["trec6"]["a2"])}}
    return out


def analyze() -> None:
    adapter_all = pd.read_parquet(ADAPTER)
    adapter = adapter_all.loc[~adapter_all["contamination_removed"]].reset_index(drop=True)
    pred = pd.read_parquet(PRED)
    if not pred["row_id"].equals(adapter["row_id"]):
        raise RuntimeError("Prediction row order differs from frozen adapter")
    for c in pred.columns[1:]:
        v = pred[c].to_numpy(dtype=np.float64)
        if not np.isfinite(v).all() or (v < 0).any() or (v > 1).any():
            raise RuntimeError(f"invalid scores in {c}")
    y = adapter["label"].to_numpy(dtype=int)
    rb = load_rb_module()
    final_receipt = json.loads((RUN / "final_fit_receipt.json").read_text(encoding="utf-8"))
    th = final_receipt["locked_thresholds"]
    threshold_a1 = float(th["A1_TFIDF_LR"])
    threshold_a2 = float(th["A2_MDISTILBERT"])
    summary = {
        "schema": "P8_CEAS08_EXTERNAL_RESULTS_V1",
        "study_id": "P8-IJIS-E2-20261106",
        "adapter_sha256": sha256(ADAPTER),
        "split_sha256": sha256(SPLIT),
        "predictions_sha256": sha256(PRED),
        "evaluation_rows": int(len(y)),
        "evaluation_label_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "locked_thresholds": {"a1": threshold_a1, "a2": threshold_a2},
        "score_comparison_dtype": "float64",
        "routes": {},
    }
    long_rows = []
    for route in ROUTES:
        a1 = pred[f"{route}_a1"].to_numpy(dtype=np.float64)
        a2 = pred[f"{route}_a2"].to_numpy(dtype=np.float64)
        m1 = metric_block(y, a1, threshold_a1)
        m2 = metric_block(y, a2, threshold_a2)
        boot = paired_bootstrap(y, a1, a2, threshold_a1, threshold_a2)
        summary["routes"][route] = {"a1": m1, "a2": m2, "paired_bootstrap": boot}
        for model, m in [("a1", m1), ("a2", m2)]:
            for metric in ["auroc", "average_precision", "log_loss", "brier", "ece10", "ece15", "ece20", "recall", "fpr", "fp", "fn", "tp", "tn"]:
                ci = boot["per_model_ci95"][model].get(metric, [None, None])
                long_rows.append({"route": route, "model": model, "metric": metric, "value": m[metric],
                                  "ci95_low": ci[0], "ci95_high": ci[1]})
        for metric, d in boot["differences"].items():
            long_rows.append({"route": route, "model": "a2_minus_a1", "metric": metric, "value": d["a2_minus_a1_mean"],
                              "ci95_low": d["ci95"][0], "ci95_high": d["ci95"][1]})
        print(json.dumps({"route": route, "a1": {k: m1[k] for k in BOOT_KEYS}, "a2": {k: m2[k] for k in BOOT_KEYS}}), flush=True)

    caps = thresholds_for_caps(rb, [0.005, 0.01, 0.02])
    sens = {}
    for cap, rec in caps.items():
        sens[cap] = {"thresholds": rec["common"], "per_source_thresholds": rec["per_source"], "routes": {}}
        for route in ROUTES:
            sens[cap]["routes"][route] = {
                "a1": metric_block(y, pred[f"{route}_a1"].to_numpy(dtype=np.float64), rec["common"]["a1"]),
                "a2": metric_block(y, pred[f"{route}_a2"].to_numpy(dtype=np.float64), rec["common"]["a2"]),
            }
    summary["threshold_cap_sensitivity"] = sens
    if abs(caps["0.01"]["common"]["a1"] - threshold_a1) > 0 or abs(caps["0.01"]["common"]["a2"] - threshold_a2) > 0:
        summary["warning_locked_vs_recomputed_cutoff"] = {"recomputed": caps["0.01"]["common"], "locked": [threshold_a1, threshold_a2]}

    q = {}
    for route in ROUTES:
        q[route] = {}
        for model in ["a1", "a2"]:
            s = pred[f"{route}_{model}"].to_numpy(dtype=np.float64)
            q[route][model] = {
                "q01": float(np.quantile(s, 0.01)), "q05": float(np.quantile(s, 0.05)),
                "q50": float(np.quantile(s, 0.50)), "q95": float(np.quantile(s, 0.95)),
                "q99": float(np.quantile(s, 0.99)), "min": float(s.min()), "max": float(s.max()),
                "share_negatives_ge_0_999": float((s[y == 0] >= 0.999).mean()),
                "share_positives_ge_0_999": float((s[y == 1] >= 0.999).mean()),
            }
    summary["score_quantiles"] = q
    frz = json.loads((RECEIPT_DIR / "adapter_freeze.json").read_text(encoding="utf-8"))
    summary["url_residual_test"] = frz.get("url_residual_test")
    summary["temporal_groups"] = "not applicable: CEAS-08 messages are from the 2008 challenge; no temporal split was pre-specified"
    write_json(RESULT_DIR / "e2_ceas08_results.json", summary)
    pd.DataFrame(long_rows).to_csv(RESULT_DIR / "e2_metrics_long.csv", index=False)
    print(json.dumps({"results": str(RESULT_DIR / "e2_ceas08_results.json")}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "score", "analyze"])
    args = ap.parse_args()
    {"prepare": prepare, "score": score, "analyze": analyze}[args.action]()


if __name__ == "__main__":
    main()
