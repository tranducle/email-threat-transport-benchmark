#!/usr/bin/env python3
"""RB-G1 semantic-anchor runner for the Paper8 re-baseline.

The runner is intentionally phase separated. The default ``audit`` phase performs
no fitting. ``fit-folds`` may access only TREC5/TREC6 development rows and creates
source-held-out thresholds. ``final-fit`` may run only after a completed folds
receipt and still may access only TREC5/TREC6. No phase in this file reads
SpaPhish. TREC7 is available only through the explicit retrospective phase and
cannot alter a model, threshold, or configuration.

Official execution remains subject to the project supervisor and SCIE-Q1 evidence
coverage gate. Merely having this script does not authorize training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix


PAPER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PAPER_ROOT.parents[1]
FREEZE = PAPER_ROOT / "8_Project_Management/PROJECT_REBASELINE_20261022/17_RB_G1_SEMANTIC_ANCHOR_FREEZE.json"
SUPERVISOR = PAPER_ROOT / "8_Project_Management/PROJECT_REBASELINE_20261022/19_RB_G1_SUPERVISOR_DECISION.json"
DATA = PAPER_ROOT / "5_Experiments_Simulations/preprocessed/meajor_preprocessed.parquet"
RUN_ROOT = PAPER_ROOT / "5_Experiments_Simulations/rebaseline_rb_g1/runs"

ALLOWED_DEV_SOURCES = ("trec5", "trec6")
RETROSPECTIVE_SOURCE = "trec7"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def load_freeze() -> dict:
    if not FREEZE.exists():
        raise RuntimeError(f"Missing freeze artifact: {FREEZE}")
    freeze = read_json(FREEZE)
    if freeze.get("schema") != "PHISHKAN_RB_G1_SEMANTIC_ANCHOR_FREEZE_V1":
        raise RuntimeError("Unexpected RB-G1 freeze schema")
    if freeze.get("status") != "FROZEN_SPEC_NO_FIT":
        raise RuntimeError("RB-G1 freeze is not in FROZEN_SPEC_NO_FIT state")
    if len(freeze.get("anchors", [])) != 2:
        raise RuntimeError("RB-G1 must contain exactly two anchors")
    return freeze


def require_supervisor_implementation_permission() -> dict:
    if not SUPERVISOR.exists():
        raise RuntimeError(f"Missing supervisor decision: {SUPERVISOR}")
    decision = read_json(SUPERVISOR)
    if decision.get("verdict") != "CONFIRMED_PASS_DESIGN_ONLY":
        raise RuntimeError("RB-G1 supervisor design gate is not PASS")
    auth = decision.get("authorization", {})
    if not auth.get("implementation_of_phase_separated_runner"):
        raise RuntimeError("Supervisor did not authorize runner implementation")
    return decision


def require_training_authorization(path: Path) -> dict:
    """Require an explicit later receipt before any empirical fit phase.

    This intentionally does not infer permission from RB-G1 design approval.
    """
    if not path.exists():
        raise RuntimeError(
            "Training is blocked. Supply an existing SCIE-Q1 pre-training gate receipt "
            "whose launch_decision.official_training_authorized is true."
        )
    payload = read_json(path)
    if payload.get("launch_decision", {}).get("official_training_authorized") is not True:
        raise RuntimeError("Provided pre-training gate receipt does not authorize official training")
    return payload


def load_meajor(*, sources: Iterable[str]) -> pd.DataFrame:
    requested = tuple(sources)
    unknown = sorted(set(requested) - set(ALLOWED_DEV_SOURCES) - {RETROSPECTIVE_SOURCE})
    if unknown:
        raise RuntimeError(f"Unsupported source request: {unknown}")
    frame = pd.read_parquet(DATA, columns=["row_id", "label", "source", "text_clean", "canonical_hash"])
    frame = frame[frame["label"].isin([0.0, 1.0])].copy()
    frame = frame[frame["source"].isin(requested)].copy()
    if frame.empty:
        raise RuntimeError(f"No rows found for sources={requested}")
    if frame["row_id"].duplicated().any():
        raise RuntimeError("Duplicate row_id in selected MeAJOR rows")
    frame["label"] = frame["label"].astype(int)
    frame["text_clean"] = frame["text_clean"].fillna("").astype(str)
    if (frame["text_clean"].str.len() == 0).any():
        raise RuntimeError("Empty text_clean row in selected data")
    for source, group in frame.groupby("source"):
        if set(group["label"].unique()) != {0, 1}:
            raise RuntimeError(f"Source {source} lacks both classes")
    return frame[["row_id", "label", "source", "text_clean", "canonical_hash"]].reset_index(drop=True)


def threshold_at_max_fpr(y: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01) -> float:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y) != len(scores) or len(y) == 0:
        raise ValueError("y and scores must be non-empty and equal length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    negatives = scores[y == 0]
    if len(negatives) == 0:
        raise ValueError("at least one negative is required")
    allowed_fp = int(math.floor(max_fpr * len(negatives)))
    descending = np.sort(negatives)[::-1]
    if allowed_fp >= len(descending):
        threshold = float(np.min(descending))
    else:
        threshold = float(np.nextafter(descending[allowed_fp], np.inf))
    actual_fp = int((negatives >= threshold).sum())
    if actual_fp > allowed_fp:
        raise AssertionError("threshold construction violated max_fpr")
    return threshold


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = [int(x) for x in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "auprc": float(average_precision_score(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
        "threshold": float(threshold),
        "fpr": fp / max(1, tn + fp),
        "recall": tp / max(1, tp + fn),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def make_tfidf(freeze: dict) -> TfidfVectorizer:
    cfg = freeze["anchors"][0]["vectorizer"]
    return TfidfVectorizer(
        max_features=int(cfg["max_features"]),
        min_df=int(cfg["min_df"]),
        ngram_range=tuple(cfg["ngram_range"]),
        lowercase=bool(cfg["lowercase"]),
        strip_accents=str(cfg["strip_accents"]),
        sublinear_tf=bool(cfg["sublinear_tf"]),
    )


def make_lr(freeze: dict) -> LogisticRegression:
    cfg = freeze["anchors"][0]["classifier"]
    return LogisticRegression(
        C=float(cfg["C"]),
        class_weight=cfg["class_weight"],
        solver=cfg["solver"],
        max_iter=int(cfg["max_iter"]),
    )


@dataclass(frozen=True)
class Fold:
    fit_source: str
    validation_source: str


def folds_from_freeze(freeze: dict) -> list[Fold]:
    folds = [Fold(**item) for item in freeze["development_data"]["source_folds"]]
    observed = {(f.fit_source, f.validation_source) for f in folds}
    expected = {("trec5", "trec6"), ("trec6", "trec5")}
    if observed != expected:
        raise RuntimeError(f"Unexpected source folds: {observed}")
    return folds


def audit() -> dict:
    freeze = load_freeze()
    supervisor = require_supervisor_implementation_permission()
    data = load_meajor(sources=ALLOWED_DEV_SOURCES)
    source_counts = {}
    for source, group in data.groupby("source"):
        source_counts[source] = {
            "rows": int(len(group)),
            "label_counts": {str(int(k)): int(v) for k, v in group["label"].value_counts().sort_index().items()},
            "unique_canonical_hashes": int(group["canonical_hash"].nunique()),
        }
    payload = {
        "schema": "PHISHKAN_RB_G1_RUNNER_AUDIT_V1",
        "status": "PASS_NO_FIT",
        "freeze_sha256": sha256(FREEZE),
        "supervisor_sha256": sha256(SUPERVISOR),
        "data_sha256": sha256(DATA),
        "script_sha256": sha256(Path(__file__)),
        "sources": source_counts,
        "folds": [f.__dict__ for f in folds_from_freeze(freeze)],
        "training_authorized_by_supervisor": bool(supervisor["authorization"].get("empirical_fitting")),
        "spaPhish_accessed": False,
        "trec7_accessed": False,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
        },
    }
    try:
        import torch
        import transformers

        payload["runtime"].update({"torch": torch.__version__, "transformers": transformers.__version__})
    except ImportError as exc:
        payload["runtime"]["transformer_stack_error"] = repr(exc)
    return payload


def fit_tfidf_fold(freeze: dict, fit_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: Path) -> dict:
    vectorizer = make_tfidf(freeze)
    train_x = vectorizer.fit_transform(fit_df["text_clean"].tolist())
    val_x = vectorizer.transform(val_df["text_clean"].tolist())
    model = make_lr(freeze)
    model.fit(train_x, fit_df["label"].to_numpy())
    scores = model.predict_proba(val_x)[:, 1]
    threshold = threshold_at_max_fpr(val_df["label"].to_numpy(), scores)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": model}, out_dir / "tfidf_lr.joblib", compress=1)
    pd.DataFrame(
        {"row_id": val_df["row_id"], "label": val_df["label"], "score": scores}
    ).to_parquet(out_dir / "tfidf_lr_validation.parquet", index=False)
    return {"threshold": threshold, "metrics": metrics(val_df["label"].to_numpy(), scores, threshold)}


def transformer_stack():
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    return torch, DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def fit_transformer_once(
    freeze: dict,
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int,
    out_dir: Path,
) -> np.ndarray:
    torch, DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer, scheduler_factory = transformer_stack()
    cfg = freeze["anchors"][1]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2).to(device)

    class TextDataset(Dataset):
        def __init__(self, frame: pd.DataFrame) -> None:
            self.texts = frame["text_clean"].tolist()
            self.labels = frame["label"].to_numpy(dtype=np.int64)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int):
            encoded = tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=int(cfg["max_length"]),
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in encoded.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(TextDataset(fit_df), batch_size=int(cfg["batch_size"]), shuffle=True, generator=generator)
    val_loader = DataLoader(TextDataset(val_df), batch_size=int(cfg["batch_size"]), shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    total_steps = max(1, len(train_loader) * int(cfg["epochs"]))
    warmup = max(1, int(round(total_steps * float(cfg["warmup_fraction"]))))
    scheduler = scheduler_factory(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)

    for _epoch in range(int(cfg["epochs"])):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip_norm"]))
            optimizer.step()
            scheduler.step()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in val_loader:
            batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            prob = torch.softmax(model(**batch).logits, dim=-1)[:, 1]
            probs.append(prob.detach().cpu().numpy())
    return np.concatenate(probs)


def fit_folds(run_dir: Path, auth_receipt: Path) -> dict:
    freeze = load_freeze()
    require_supervisor_implementation_permission()
    auth = require_training_authorization(auth_receipt)
    run_dir.mkdir(parents=True, exist_ok=False)
    all_dev = load_meajor(sources=ALLOWED_DEV_SOURCES)
    fold_records = []
    thresholds_a1 = []
    thresholds_a2 = []

    for fold in folds_from_freeze(freeze):
        fit_df = all_dev[all_dev["source"] == fold.fit_source].reset_index(drop=True)
        val_df = all_dev[all_dev["source"] == fold.validation_source].reset_index(drop=True)
        fold_dir = run_dir / f"fit_{fold.fit_source}__val_{fold.validation_source}"
        a1 = fit_tfidf_fold(freeze, fit_df, val_df, fold_dir / "A1_TFIDF_LR")
        thresholds_a1.append(float(a1["threshold"]))

        seed_scores = []
        for seed in freeze["anchors"][1]["seeds"]:
            scores = fit_transformer_once(freeze, fit_df, val_df, int(seed), fold_dir / "A2_MDISTILBERT" / f"seed_{seed}")
            seed_scores.append(scores)
        ensemble = np.mean(np.stack(seed_scores, axis=0), axis=0)
        threshold_a2 = threshold_at_max_fpr(val_df["label"].to_numpy(), ensemble)
        thresholds_a2.append(float(threshold_a2))
        pd.DataFrame(
            {"row_id": val_df["row_id"], "label": val_df["label"], "score": ensemble}
        ).to_parquet(fold_dir / "A2_MDISTILBERT" / "ensemble_validation.parquet", index=False)
        fold_records.append(
            {
                "fit_source": fold.fit_source,
                "validation_source": fold.validation_source,
                "A1_TFIDF_LR": a1,
                "A2_MDISTILBERT": {
                    "threshold": float(threshold_a2),
                    "metrics": metrics(val_df["label"].to_numpy(), ensemble, threshold_a2),
                },
            }
        )

    payload = {
        "schema": "PHISHKAN_RB_G1_FOLD_FIT_V1",
        "status": "FROZEN_FOLDS_COMPLETE",
        "freeze_sha256": sha256(FREEZE),
        "script_sha256": sha256(Path(__file__)),
        "training_authorization_sha256": sha256(auth_receipt),
        "training_authorization_schema": auth.get("schema"),
        "folds": fold_records,
        "locked_thresholds": {
            "A1_TFIDF_LR": max(thresholds_a1),
            "A2_MDISTILBERT": max(thresholds_a2),
        },
        "spaPhish_accessed": False,
        "trec7_accessed": False,
    }
    write_json(run_dir / "fold_fit_receipt.json", payload)
    return payload


def fit_tfidf_final(freeze: dict, data: pd.DataFrame, out_dir: Path) -> None:
    vectorizer = make_tfidf(freeze)
    x = vectorizer.fit_transform(data["text_clean"].tolist())
    model = make_lr(freeze)
    model.fit(x, data["label"].to_numpy())
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": model}, out_dir / "tfidf_lr.joblib", compress=1)


def fit_transformer_final_once(freeze: dict, data: pd.DataFrame, seed: int, out_dir: Path) -> None:
    # Reuse the fixed training implementation with an internal empty-free validation substitute
    # would risk changing semantics. Keep a dedicated final-fit loop instead.
    torch, DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer, scheduler_factory = transformer_stack()
    cfg = freeze["anchors"][1]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2).to(device)

    class TextDataset(Dataset):
        def __len__(self) -> int:
            return len(data)

        def __getitem__(self, idx: int):
            encoded = tokenizer(
                str(data.iloc[idx]["text_clean"]), truncation=True, padding="max_length",
                max_length=int(cfg["max_length"]), return_tensors="pt"
            )
            item = {k: v.squeeze(0) for k, v in encoded.items()}
            item["labels"] = torch.tensor(int(data.iloc[idx]["label"]), dtype=torch.long)
            return item

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(TextDataset(), batch_size=int(cfg["batch_size"]), shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    total_steps = max(1, len(loader) * int(cfg["epochs"]))
    warmup = max(1, int(round(total_steps * float(cfg["warmup_fraction"]))))
    scheduler = scheduler_factory(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)
    for _epoch in range(int(cfg["epochs"])):
        model.train()
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip_norm"]))
            optimizer.step()
            scheduler.step()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def final_fit(run_dir: Path, auth_receipt: Path) -> dict:
    freeze = load_freeze()
    require_supervisor_implementation_permission()
    require_training_authorization(auth_receipt)
    fold_receipt_path = run_dir / "fold_fit_receipt.json"
    if not fold_receipt_path.exists():
        raise RuntimeError("final-fit requires an existing fold_fit_receipt.json")
    fold_receipt = read_json(fold_receipt_path)
    if fold_receipt.get("status") != "FROZEN_FOLDS_COMPLETE":
        raise RuntimeError("fold-fit receipt is not complete")
    if fold_receipt.get("freeze_sha256") != sha256(FREEZE):
        raise RuntimeError("freeze artifact changed after fold fitting")
    if fold_receipt.get("script_sha256") != sha256(Path(__file__)):
        raise RuntimeError("runner changed after fold fitting")

    data = load_meajor(sources=ALLOWED_DEV_SOURCES)
    final_dir = run_dir / "final_models"
    if final_dir.exists():
        raise RuntimeError("final_models already exists; refuse overwrite")
    fit_tfidf_final(freeze, data, final_dir / "A1_TFIDF_LR")
    for seed in freeze["anchors"][1]["seeds"]:
        fit_transformer_final_once(freeze, data, int(seed), final_dir / "A2_MDISTILBERT" / f"seed_{seed}")
    payload = {
        "schema": "PHISHKAN_RB_G1_FINAL_FIT_V1",
        "status": "FINAL_MODELS_FROZEN",
        "freeze_sha256": sha256(FREEZE),
        "script_sha256": sha256(Path(__file__)),
        "fold_fit_receipt_sha256": sha256(fold_receipt_path),
        "locked_thresholds": fold_receipt["locked_thresholds"],
        "fit_sources": list(ALLOWED_DEV_SOURCES),
        "spaPhish_accessed": False,
        "trec7_accessed": False,
    }
    write_json(run_dir / "final_fit_receipt.json", payload)
    return payload


def score_tfidf(bundle_path: Path, data: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(bundle_path)
    return bundle["model"].predict_proba(bundle["vectorizer"].transform(data["text_clean"].tolist()))[:, 1]


def score_transformer_ensemble(freeze: dict, model_root: Path, data: pd.DataFrame) -> np.ndarray:
    torch, DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer, _ = transformer_stack()
    cfg = freeze["anchors"][1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_seed = []
    for seed in cfg["seeds"]:
        path = model_root / f"seed_{seed}"
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(device)

        class TextDataset(Dataset):
            def __len__(self) -> int:
                return len(data)

            def __getitem__(self, idx: int):
                encoded = tokenizer(
                    str(data.iloc[idx]["text_clean"]), truncation=True, padding="max_length",
                    max_length=int(cfg["max_length"]), return_tensors="pt"
                )
                return {k: v.squeeze(0) for k, v in encoded.items()}

        loader = DataLoader(TextDataset(), batch_size=int(cfg["batch_size"]), shuffle=False)
        model.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                probs.append(torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu().numpy())
        per_seed.append(np.concatenate(probs))
    return np.mean(np.stack(per_seed, axis=0), axis=0)


def retrospective_score(run_dir: Path) -> dict:
    """Score legacy TREC7 with frozen models. This phase never writes selection state."""
    freeze = load_freeze()
    final_receipt_path = run_dir / "final_fit_receipt.json"
    if not final_receipt_path.exists():
        raise RuntimeError("retrospective scoring requires final_fit_receipt.json")
    final_receipt = read_json(final_receipt_path)
    if final_receipt.get("freeze_sha256") != sha256(FREEZE):
        raise RuntimeError("freeze changed after final fit")
    if final_receipt.get("script_sha256") != sha256(Path(__file__)):
        raise RuntimeError("runner changed after final fit")
    data = load_meajor(sources=(RETROSPECTIVE_SOURCE,))
    y = data["label"].to_numpy()
    thresholds = final_receipt["locked_thresholds"]
    s1 = score_tfidf(run_dir / "final_models/A1_TFIDF_LR/tfidf_lr.joblib", data)
    s2 = score_transformer_ensemble(freeze, run_dir / "final_models/A2_MDISTILBERT", data)
    payload = {
        "schema": "PHISHKAN_RB_G1_TREC7_RETROSPECTIVE_V1",
        "status": "RETROSPECTIVE_DIAGNOSTIC_ONLY",
        "source": RETROSPECTIVE_SOURCE,
        "A1_TFIDF_LR": metrics(y, s1, float(thresholds["A1_TFIDF_LR"])),
        "A2_MDISTILBERT": metrics(y, s2, float(thresholds["A2_MDISTILBERT"])),
        "selection_effect": "NONE_ALLOWED",
        "spaPhish_accessed": False,
    }
    write_json(run_dir / "trec7_retrospective.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["audit", "fit-folds", "final-fit", "score-retrospective"], default="audit")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--training-authorization", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "audit":
        payload = audit()
        out = PAPER_ROOT / "8_Project_Management/PROJECT_REBASELINE_20261022/20_RB_G1_RUNNER_AUDIT.json"
        write_json(out, payload)
        print(json.dumps({"status": payload["status"], "out": str(out)}, indent=2))
        return
    if args.run_dir is None:
        raise SystemExit("--run-dir is required for empirical phases")
    run_dir = args.run_dir.resolve()
    if RUN_ROOT.resolve() not in run_dir.parents:
        raise SystemExit(f"--run-dir must be a child of {RUN_ROOT}")
    if args.phase in {"fit-folds", "final-fit"}:
        if args.training_authorization is None:
            raise SystemExit("--training-authorization is required for fit phases")
        if args.phase == "fit-folds":
            result = fit_folds(run_dir, args.training_authorization.resolve())
        else:
            result = final_fit(run_dir, args.training_authorization.resolve())
    else:
        result = retrospective_score(run_dir)
    print(json.dumps({"status": result["status"], "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
