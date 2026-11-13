"""Cross-fitted A2 scores on the development sources and the static residual probe.

Phase crossfit: within each of TREC5 and TREC6, a 2-fold stratified split (seed in
configs/residual_probe.json) trains the three A2 seeds on one half and scores the other, so no
message is scored by a model trained on it. Writes crossfit/<source>_a2_crossfit.parquet.
Phase residual-fit: for each static feature block (URL structure, domain shape without
identity, message structure) fits a ridge residual probe on the cross-fitted scores of one
source and evaluates it on the other source at the locked A2 cutoff.
Phase audit: checks the feature blocks on the development data; fits nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_ROOT = Path(os.environ.get("THRESHOLD_TRANSPORT_DATA", REPO_ROOT / "data"))
DATA = DATA_ROOT / "processed" / "meajor_preprocessed.parquet"
CONFIG = REPO_ROOT / "threshold_transport" / "configs" / "residual_probe.json"
ANCHOR_RUN = DATA_ROOT / "models" / "anchors"
RUN_DIR = DATA_ROOT / "models" / "crossfit"
ANCHOR_SCRIPT = HERE / "train_anchor_models.py"
AUDIT_OUT = DATA_ROOT / "results" / "receipts" / "crossfit_data_audit.json"

ALLOWED_SOURCES = ("trec5", "trec6")
BASE_COLUMNS = [
    "row_id", "source", "label", "text_clean", "sender_domain_norm",
    "receiver_domain_norm", "text_length_chars", "token_count", "url_indicator",
    "url_count", "url_count_parsed", "url_length_max", "url_length_avg",
    "url_subdom_max", "url_subdom_avg", "url_domains_json", "attachment_count",
    "has_attachments", "attachment_types", "content_types",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_dev() -> pd.DataFrame:
    df = pd.read_parquet(DATA, columns=BASE_COLUMNS, filters=[("source", "in", list(ALLOWED_SOURCES))])
    if not set(df["source"].unique()).issubset(ALLOWED_SOURCES):
        raise RuntimeError("non-development source materialized")
    return df.reset_index(drop=True)


def count_listish(value: object) -> float:
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


def feature_matrix(df: pd.DataFrame, block: str) -> tuple[np.ndarray, list[str]]:
    if block == "U_URL_STRUCTURE":
        names = ["url_indicator", "url_count", "url_count_parsed", "url_length_max", "url_length_avg", "url_subdom_max", "url_subdom_avg"]
        x = np.column_stack([pd.to_numeric(df[n], errors="coerce").fillna(0).to_numpy(float) for n in names])
    elif block == "D_DOMAIN_SHAPE_NO_IDENTITY":
        s = df["sender_domain_norm"].fillna("").astype(str)
        r = df["receiver_domain_norm"].fillna("").astype(str)
        names = ["sender_domain_length", "receiver_domain_length", "sender_receiver_domain_equal", "sender_url_domain_match"]
        x = np.column_stack([
            s.str.len().to_numpy(float), r.str.len().to_numpy(float),
            ((s != "") & (s == r)).astype(float).to_numpy(),
            np.array([sender_url_match(a, b) for a, b in zip(s, df["url_domains_json"])], dtype=float),
        ])
    elif block == "M_MESSAGE_STRUCTURE":
        names = ["text_length_chars", "token_count", "attachment_count", "has_attachments", "content_type_count", "attachment_type_count"]
        x = np.column_stack([
            pd.to_numeric(df["text_length_chars"], errors="coerce").fillna(0).to_numpy(float),
            pd.to_numeric(df["token_count"], errors="coerce").fillna(0).to_numpy(float),
            pd.to_numeric(df["attachment_count"], errors="coerce").fillna(0).to_numpy(float),
            pd.to_numeric(df["has_attachments"], errors="coerce").fillna(0).to_numpy(float),
            np.array([count_listish(v) for v in df["content_types"]], dtype=float),
            np.array([count_listish(v) for v in df["attachment_types"]], dtype=float),
        ])
    else:
        raise KeyError(block)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), names


def ece15(y: np.ndarray, p: np.ndarray) -> float:
    edges = np.linspace(0, 1, 16)
    value = 0.0
    for i in range(15):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < 14 else (p <= edges[i + 1]))
        if m.any():
            value += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return value


def metric_pack(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1))); fn = int(np.sum((~pred) & (y == 1)))
    fp = int(np.sum(pred & (y == 0))); tn = int(np.sum((~pred) & (y == 0)))
    eps = 1e-7
    return {
        "auprc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, eps, 1 - eps))),
        "brier": float(brier_score_loss(y, p)),
        "ece15": ece15(y, p),
        "recall": tp / (tp + fn),
        "fpr": fp / (fp + tn),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "threshold": float(threshold),
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def import_anchor_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_anchor_models", ANCHOR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def crossfit_source(source: str, out_dir: Path, config: dict) -> Path:
    data = load_dev()
    df = data[data["source"] == source].copy().reset_index(drop=True)
    y = df["label"].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=int(config["semantic_anchor"]["inner_folds"]), shuffle=True, random_state=int(config["semantic_anchor"]["inner_split_seed"]))
    scores = np.full(len(df), np.nan, dtype=float)
    anchor = import_anchor_module()
    anchor_cfg = anchor.load_config()
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        train_df = df.iloc[train_idx].copy(); val_df = df.iloc[val_idx].copy()
        per_seed = []
        for seed in anchor_cfg["anchors"][1]["seeds"]:
            model_dir = out_dir / source / f"inner_{fold_index}" / f"seed_{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            score_path = model_dir / "validation.parquet"
            if score_path.exists():
                saved = pd.read_parquet(score_path)
                expected_ids = val_df["row_id"].astype(str).tolist()
                if saved["row_id"].astype(str).tolist() != expected_ids:
                    raise RuntimeError(f"resume score row mismatch for {source} fold={fold_index} seed={seed}")
                seed_scores = saved["score"].to_numpy(dtype=float)
            else:
                seed_scores = anchor.fit_transformer_once(anchor_cfg, train_df, val_df, int(seed), model_dir)
                pd.DataFrame({
                    "row_id": val_df["row_id"].astype(str).tolist(),
                    "label": val_df["label"].to_numpy(dtype=int),
                    "score": seed_scores,
                }).to_parquet(score_path, index=False)
            if len(seed_scores) != len(val_idx) or not np.isfinite(seed_scores).all():
                raise RuntimeError(f"invalid inner-fold scores for {source} fold={fold_index} seed={seed}")
            per_seed.append(seed_scores)
        scores[val_idx] = np.mean(np.stack(per_seed, axis=0), axis=0)
    if np.isnan(scores).any():
        raise RuntimeError(f"incomplete cross-fit scores for {source}")
    out = pd.DataFrame({"row_id": df["row_id"], "source": source, "label": y, "score": scores})
    path = out_dir / f"{source}_a2_crossfit.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def anchor_outer_scores(validation_source: str) -> pd.DataFrame:
    fit_source = "trec5" if validation_source == "trec6" else "trec6"
    path = ANCHOR_RUN / f"fit_{fit_source}__val_{validation_source}" / "A2_MDISTILBERT" / "ensemble_validation.parquet"
    return pd.read_parquet(path)


def fit_probe(train_df: pd.DataFrame, train_scores: pd.DataFrame, val_df: pd.DataFrame, val_scores: pd.DataFrame, block: str, config: dict, threshold: float) -> dict:
    tr = train_df.merge(train_scores[["row_id", "score"]], on="row_id", validate="one_to_one")
    va = val_df.merge(val_scores[["row_id", "score"]], on="row_id", validate="one_to_one")
    if len(tr) != len(train_df) or len(va) != len(val_df):
        raise RuntimeError("semantic-score join lost or duplicated rows")
    x, names = feature_matrix(tr, block); xv, _ = feature_matrix(va, block)
    if not np.isfinite(x).all() or not np.isfinite(xv).all():
        raise RuntimeError(f"non-finite residual features in {block}")
    y = tr["label"].to_numpy(int); yv = va["label"].to_numpy(int)
    p0 = tr["score"].to_numpy(float); pv0 = va["score"].to_numpy(float)
    idx_fit, idx_sel = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=int(config["probe"]["selection_seed"]))
    scaler = StandardScaler().fit(x[idx_fit])
    ridge = Ridge(alpha=float(config["probe"]["ridge_alpha"])).fit(scaler.transform(x[idx_fit]), y[idx_fit] - p0[idx_fit])
    r_sel = ridge.predict(scaler.transform(x[idx_sel]))
    candidates = []
    for a in config["probe"]["coefficient_grid"]:
        p = sigmoid(logit(p0[idx_sel]) + float(a) * r_sel)
        candidates.append((float(log_loss(y[idx_sel], np.clip(p, 1e-7, 1 - 1e-7))), float(a)))
    candidates.sort(key=lambda z: (z[0], z[1]))
    chosen = candidates[0][1]
    scaler = StandardScaler().fit(x)
    ridge = Ridge(alpha=float(config["probe"]["ridge_alpha"])).fit(scaler.transform(x), y - p0)
    rv = ridge.predict(scaler.transform(xv))
    pv = sigmoid(logit(pv0) + chosen * rv)
    base = metric_pack(yv, pv0, threshold); fused = metric_pack(yv, pv, threshold)
    delta = {"auprc": fused["auprc"] - base["auprc"], "log_loss": fused["log_loss"] - base["log_loss"], "brier": fused["brier"] - base["brier"], "ece15": fused["ece15"] - base["ece15"], "fpr": fused["fpr"] - base["fpr"], "recall": fused["recall"] - base["recall"]}
    return {"features": names, "coefficient": chosen, "coefficient_selection_log_loss": {str(a): loss for loss, a in candidates}, "baseline": base, "fused": fused, "delta": delta}


def audit() -> dict:
    config = load_json(CONFIG)
    if tuple(config["semantic_anchor"]["primary"] for _ in [0]) != ("A2_MDISTILBERT",):
        raise RuntimeError("unexpected semantic anchor")
    if set(ALLOWED_SOURCES) != {"trec5", "trec6"}:
        raise RuntimeError("development source allow-list changed")
    prohibited_columns = {"source_id", "sender", "receiver", "urls", "language", "canonical_hash"}
    feature_names = {name for names in config["feature_blocks"].values() for name in names}
    if feature_names & prohibited_columns:
        raise RuntimeError("prohibited residual feature in the configuration")
    data = load_dev()
    if set(data["source"].unique()) != {"trec5", "trec6"}:
        raise RuntimeError("unexpected development-source set")
    block_shapes = {}
    for block, expected_names in config["feature_blocks"].items():
        x, names = feature_matrix(data.head(1024), block)
        if names != expected_names or x.shape != (min(1024, len(data)), len(expected_names)) or not np.isfinite(x).all():
            raise RuntimeError(f"feature block audit failed: {block}")
        block_shapes[block] = list(x.shape)
    payload = {
        "schema": "crossfit_data_audit",
        "status": "PASS",
        "config_sha256": sha256(CONFIG),
        "script_sha256": sha256(Path(__file__)),
        "data_sha256": sha256(DATA),
        "allowed_sources": list(ALLOWED_SOURCES),
        "base_columns": BASE_COLUMNS,
        "feature_blocks": config["feature_blocks"],
        "feature_block_dry_shapes": block_shapes,
        "development_rows_seen": int(len(data)),
        "development_source_counts": {str(k): int(v) for k, v in data["source"].value_counts().to_dict().items()},
        "raw_identity_features_used": False,
        "spaphish_fit_path": False,
        "trec7_fit_path": False,
        "anchor_threshold_retuning": False,
        "note": "Executable loader filters parquet to trec5/trec6 before materialization; residual features are numeric shape/count derivatives only.",
    }
    return payload


def crossfit(run_dir: Path) -> dict:
    config = load_json(CONFIG)
    out = run_dir / "crossfit"
    receipts = {}
    for source in ALLOWED_SOURCES:
        path = out / f"{source}_a2_crossfit.parquet"
        if not path.exists():
            path = crossfit_source(source, out, config)
        receipts[source] = {"path": path.name, "sha256": sha256(path), "rows": int(len(pd.read_parquet(path, columns=["row_id"]))) }
    payload = {"schema": "anchor_crossfit", "status": "CROSSFIT_COMPLETE", "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "sources": receipts}
    write_json(run_dir / "crossfit_receipt.json", payload)
    return payload


def residual_fit(run_dir: Path) -> dict:
    config = load_json(CONFIG)
    receipt = load_json(run_dir / "crossfit_receipt.json")
    if receipt.get("status") != "CROSSFIT_COMPLETE":
        raise RuntimeError("crossfit must complete first")
    anchor_final = load_json(ANCHOR_RUN / "final_fit_receipt.json")
    threshold = float(anchor_final["locked_thresholds"]["A2_MDISTILBERT"])
    data = load_dev(); results = []
    for fit_source, val_source in [("trec5", "trec6"), ("trec6", "trec5")]:
        train_df = data[data.source == fit_source].copy(); val_df = data[data.source == val_source].copy()
        train_scores = pd.read_parquet(run_dir / "crossfit" / f"{fit_source}_a2_crossfit.parquet")
        val_scores = anchor_outer_scores(val_source)
        fold = {"fit_source": fit_source, "validation_source": val_source, "blocks": {}}
        for block in config["feature_blocks"]:
            fold["blocks"][block] = fit_probe(train_df, train_scores, val_df, val_scores, block, config, threshold)
        results.append(fold)
    blocks = {}
    for block in config["feature_blocks"]:
        ds = [f["blocks"][block]["delta"] for f in results]
        survives = all(d["auprc"] > 0 and d["log_loss"] < 0 and d["fpr"] <= 0.002 for d in ds)
        blocks[block] = {"candidate_useful": bool(survives), "fold_deltas": ds}
    payload = {"schema": "residual_probe_results", "status": "RESIDUAL_RESULTS_COMPLETE", "config_sha256": sha256(CONFIG), "runner_sha256": sha256(Path(__file__)), "anchor_final_receipt_sha256": sha256(ANCHOR_RUN / "final_fit_receipt.json"), "folds": results, "block_decisions": blocks, "spaPhish_accessed": False, "trec7_accessed": False}
    write_json(run_dir / "residual_probe_results.json", payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["audit", "crossfit", "residual-fit"], default="audit")
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR,
                    help="run directory (default: <data root>/models/crossfit, which the downstream scripts read)")
    args = ap.parse_args()
    if args.phase == "audit":
        result = audit(); write_json(AUDIT_OUT, result)
    else:
        run = args.run_dir.resolve()
        run.mkdir(parents=True, exist_ok=True)
        result = crossfit(run) if args.phase == "crossfit" else residual_fit(run)
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
