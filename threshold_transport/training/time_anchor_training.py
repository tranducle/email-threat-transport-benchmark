"""Time the final-route training of the two anchor models (balanced LR, A1; mDistilBERT, A2).

The final-fit phase of train_anchor_models.py does not log training time. This script reruns
that training code (imported from train_anchor_models.py, not copied) on the same pooled
TREC5+TREC6 data, records wall-clock time per model, and checks the retrained models against
the saved final models:

  * A1: vocabulary and coefficients must match the saved bundle (determinism check);
  * A2: ensemble scores on a fixed 512-message sample are compared with the saved ensemble.

Retrained models go to a caller-supplied scratch directory and are deleted afterwards; the
saved final_models directory is only read. Output: results/anchor_training_timing.json.

Usage: python time_anchor_training.py --model-dir <scratch dir> [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("THRESHOLD_TRANSPORT_DATA", REPO_ROOT / "data"))
RUNNER = Path(__file__).resolve().parent / "train_anchor_models.py"
TRAIN_DATA = DATA_ROOT / "processed" / "meajor_preprocessed.parquet"
FROZEN = DATA_ROOT / "models" / "anchors" / "final_models"
OUT = DATA_ROOT / "results" / "anchor_training_timing.json"
SAMPLE_SEED = 20261113
SAMPLE_N = 512


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def load_runner():
    spec = importlib.util.spec_from_file_location("train_anchor_models", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    mod.DATA = TRAIN_DATA
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="A1 in full, A2 one seed on 256 rows; no JSON written")
    args = ap.parse_args()

    import sklearn
    import torch
    import transformers

    if not torch.cuda.is_available():
        print("CUDA not available; the reported A2 time is a CUDA time", file=sys.stderr)
        return 2
    if args.model_dir.exists() and any(args.model_dir.iterdir()):
        print(f"model dir not empty: {args.model_dir}", file=sys.stderr)
        return 2
    args.model_dir.mkdir(parents=True, exist_ok=True)

    mod = load_runner()
    config = mod.load_config()
    data = mod.load_meajor(sources=mod.ALLOWED_DEV_SOURCES)
    print(f"training rows: {len(data)} ({dict(data['source'].value_counts())})", flush=True)

    record: dict = {
        "schema": "anchor_training_timing",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "final-route training time of A1 and A2",
        "runner": rel(RUNNER, REPO_ROOT),
        "runner_sha256": sha256(RUNNER),
        "runner_sha256_matches_final_fit_receipt": None,
        "this_script_sha256": sha256(Path(__file__)),
        "train_data": rel(TRAIN_DATA, DATA_ROOT),
        "train_data_sha256": sha256(TRAIN_DATA),
        "train_rows": int(len(data)),
        "train_rows_by_source": {k: int(v) for k, v in data["source"].value_counts().items()},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "sklearn": sklearn.__version__,
            "transformers": transformers.__version__,
        },
        "timing_scope": "wall clock from the training call to its return; includes TF-IDF fitting (A1), tokenization in the data loader (A2) and saving the fitted model",
    }
    receipt = json.loads((FROZEN.parent / "final_fit_receipt.json").read_text(encoding="utf-8"))
    record["runner_sha256_matches_final_fit_receipt"] = receipt.get("script_sha256") == record["runner_sha256"]

    # A1: balanced LR on TF-IDF, exact runner function.
    a1_dir = args.model_dir / "A1_TFIDF_LR"
    t0 = time.perf_counter()
    mod.fit_tfidf_final(config, data, a1_dir)
    a1_s = time.perf_counter() - t0
    new = joblib.load(a1_dir / "tfidf_lr.joblib")
    old = joblib.load(FROZEN / "A1_TFIDF_LR" / "tfidf_lr.joblib")
    same_vocab = new["vectorizer"].vocabulary_ == old["vectorizer"].vocabulary_
    coef_diff = float(np.max(np.abs(new["model"].coef_ - old["model"].coef_))) if same_vocab else None
    icpt_diff = float(np.max(np.abs(new["model"].intercept_ - old["model"].intercept_)))
    record["A1"] = {
        "train_seconds": round(a1_s, 2),
        "device": "CPU",
        "vocabulary_identical_to_frozen": bool(same_vocab),
        "max_abs_coef_diff_vs_frozen": coef_diff,
        "max_abs_intercept_diff_vs_frozen": icpt_diff,
    }
    print(f"A1 trained in {a1_s:.2f} s; vocab identical {same_vocab}; max |dcoef| {coef_diff}", flush=True)

    # A1 components, repeated, with the runner's own factories (no saving): separates the
    # TF-IDF fit from the classifier fit.
    reps = []
    for _ in range(1 if args.smoke else 3):
        vec = mod.make_tfidf(config)
        t0 = time.perf_counter()
        x = vec.fit_transform(data["text_clean"].tolist())
        t1 = time.perf_counter()
        mod.make_lr(config).fit(x, data["label"].to_numpy())
        t2 = time.perf_counter()
        reps.append({"tfidf_fit_seconds": round(t1 - t0, 2), "lr_fit_seconds": round(t2 - t1, 2)})
    record["A1"]["component_repeats"] = reps
    record["A1"]["median_tfidf_fit_seconds"] = float(np.median([r["tfidf_fit_seconds"] for r in reps]))
    record["A1"]["median_lr_fit_seconds"] = float(np.median([r["lr_fit_seconds"] for r in reps]))
    record["A1"]["median_total_seconds"] = float(np.median([r["tfidf_fit_seconds"] + r["lr_fit_seconds"] for r in reps]))
    print(f"A1 components: {reps}", flush=True)

    # A2: three seeds, exact runner function.
    seeds = [int(s) for s in config["anchors"][1]["seeds"]]
    a2_data = data
    if args.smoke:
        seeds = seeds[:1]
        a2_data = data.sample(n=256, random_state=SAMPLE_SEED).reset_index(drop=True)
    per_seed = {}
    for seed in seeds:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mod.fit_transformer_final_once(config, a2_data, seed, args.model_dir / "A2_MDISTILBERT" / f"seed_{seed}")
        torch.cuda.synchronize()
        per_seed[str(seed)] = round(time.perf_counter() - t0, 1)
        print(f"A2 seed {seed} trained in {per_seed[str(seed)]} s", flush=True)

    if args.smoke:
        print("smoke OK; no JSON written", flush=True)
        shutil.rmtree(args.model_dir)
        return 0

    sample = data.sample(n=SAMPLE_N, random_state=SAMPLE_SEED).reset_index(drop=True)
    s_new = mod.score_transformer_ensemble(config, args.model_dir / "A2_MDISTILBERT", sample)
    s_old = mod.score_transformer_ensemble(config, FROZEN / "A2_MDISTILBERT", sample)
    record["A2"] = {
        "device": "CUDA",
        "seeds": seeds,
        "train_seconds_per_seed": per_seed,
        "train_seconds_three_seeds": round(sum(per_seed.values()), 1),
        "train_seconds_mean_per_seed": round(sum(per_seed.values()) / len(per_seed), 1),
        "ensemble_score_check": {
            "sample_rows": SAMPLE_N,
            "sample_random_state": SAMPLE_SEED,
            "max_abs_diff_vs_frozen": float(np.max(np.abs(s_new - s_old))),
            "mean_abs_diff_vs_frozen": float(np.mean(np.abs(s_new - s_old))),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.new")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    shutil.rmtree(args.model_dir)
    print(json.dumps(record, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
