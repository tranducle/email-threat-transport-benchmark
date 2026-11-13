#!/usr/bin/env python3
"""Deterministic normalization of the MeAJOR, CEAS-08 and Nazario_5 email corpora.

``normalize_dataset`` maps labels to 0/1, normalizes subject and body whitespace, derives a
SHA-256 canonical hash of the lowercased subject plus body, removes empty messages, removes
hash groups whose rows carry conflicting labels, and keeps the first row of each exact
duplicate group. Running this module on all three corpora also writes text-free split files
(identifiers, labels, sources and hashes only) and a preprocessing summary. The
threshold-transport experiments use only the MeAJOR output, which
``build_trec_development_set.py`` produces by calling ``normalize_dataset("meajor")``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from dataclasses import dataclass
from email.utils import parseaddr
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


SEED = 20260503
VAL_FRACTION = 0.15
RANDOM_TEST_FRACTION = 0.20
RANDOM_VAL_FRACTION = 0.10

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("THRESHOLD_TRANSPORT_DATA", REPO_ROOT / "data"))
RAW_DIR = DATA_ROOT / "raw"
PREPROCESSED_DIR = DATA_ROOT / "processed"
SPLIT_DIR = DATA_ROOT / "processed" / "splits"
REPORT_DIR = DATA_ROOT / "results" / "preprocessing"

URL_RE = re.compile(r"https?://[^\s,'\"<>\\)\\]]+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


@dataclass
class DatasetResult:
    name: str
    input_rows: int
    output_rows: int
    dropped_missing_label: int
    dropped_empty_text: int
    dropped_conflict_rows: int
    removed_duplicate_rows: int
    label_counts: dict[str, int]
    source_counts: dict[str, int]
    missing_sender_domain_rate: float
    missing_receiver_domain_rate: float
    output_path: str


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_space(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\x00", " ")
    return SPACE_RE.sub(" ", text).strip()


def normalize_for_hash(subject: Any, body: Any) -> str:
    return normalize_space(f"{normalize_space(subject)} {normalize_space(body)}").lower()


def normalize_label(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "phish", "phishing", "malicious", "spam", "fraud"}:
            return 1
        if lowered in {"0", "benign", "legitimate", "ham", "valid"}:
            return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric):
        return None
    return int(numeric > 0)


def clean_domain(domain: Any) -> str:
    if domain is None or pd.isna(domain):
        return ""
    value = str(domain).strip().lower()
    value = value.replace("mailto:", "")
    value = value.strip("<>()[]{}'\" ")
    value = value.rstrip(".")
    if "/" in value and not value.startswith("http"):
        value = value.split("/", 1)[0]
    if ":" in value and not value.startswith("http"):
        value = value.split(":", 1)[0]
    if "@" in value:
        value = value.rsplit("@", 1)[-1]
    return value


def domain_from_email(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    raw = str(value)
    parsed = parseaddr(raw)[1] or raw
    if "@" in parsed:
        return clean_domain(parsed.rsplit("@", 1)[-1])
    match = re.search(r"@([A-Za-z0-9_.-]+\.[A-Za-z]{2,})", raw)
    if match:
        return clean_domain(match.group(1))
    return clean_domain(raw)


def parse_urls(value: Any) -> tuple[list[str], int]:
    if value is None or pd.isna(value):
        return [], 0

    if isinstance(value, (int, np.integer, bool)):
        indicator = int(value) > 0
        return [], int(indicator)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return [], 0
        return [], int(value > 0)

    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "[]"}:
        return [], 0

    parsed_items: list[str] = []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            literal = ast.literal_eval(raw)
            if isinstance(literal, (list, tuple)):
                parsed_items = [str(item) for item in literal if item]
        except (SyntaxError, ValueError):
            parsed_items = []

    candidates = parsed_items or URL_RE.findall(raw)
    if not candidates:
        candidates = [part.strip() for part in raw.splitlines() if part.strip().startswith(("http://", "https://"))]

    urls = []
    for candidate in candidates:
        cleaned = candidate.strip().strip(",'\"<>[]()")
        if cleaned.startswith(("http://", "https://")):
            urls.append(cleaned)
    return urls, int(bool(urls))


def domains_from_urls(urls: list[str]) -> list[str]:
    domains: list[str] = []
    for url in urls:
        try:
            netloc = urlparse(url).netloc
        except ValueError:
            continue
        domain = clean_domain(netloc)
        if domain:
            domains.append(domain)
    return sorted(set(domains))


def defang_url(url: str) -> str:
    return url.replace("http://", "hxxp://").replace("https://", "hxxps://").replace(".", "[.]")


def load_raw_dataset(name: str) -> pd.DataFrame:
    if name == "meajor":
        path = RAW_DIR / "meajor_cleaned_preprocessed.parquet.gzip"
        return pd.read_parquet(path)
    if name == "ceas_08":
        path = RAW_DIR / "CEAS_08.csv"
        return pd.read_csv(path)
    if name == "nazario_5":
        path = RAW_DIR / "Nazario_5.csv"
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset: {name}")


def normalize_dataset(name: str) -> tuple[pd.DataFrame, DatasetResult]:
    raw = load_raw_dataset(name)
    input_rows = len(raw)
    df = raw.copy()
    df["dataset"] = name
    df["original_index"] = np.arange(len(df), dtype=np.int64)
    df["row_id"] = [f"{name}_{idx:07d}" for idx in df["original_index"]]

    df["label"] = df["label"].map(normalize_label)
    missing_label_mask = df["label"].isna()
    dropped_missing_label = int(missing_label_mask.sum())
    df = df.loc[~missing_label_mask].copy()
    df["label"] = df["label"].astype(int)

    if "source" not in df.columns:
        df["source"] = name
    df["source"] = df["source"].map(lambda value: normalize_space(value).lower())

    for column in ["sender", "receiver", "subject", "body", "urls"]:
        if column not in df.columns:
            df[column] = ""

    if "sender_domain" in df.columns:
        df["sender_domain_norm"] = df["sender_domain"].map(clean_domain)
        missing_sender = df["sender_domain_norm"].eq("")
        df.loc[missing_sender, "sender_domain_norm"] = df.loc[missing_sender, "sender"].map(domain_from_email)
    else:
        df["sender_domain_norm"] = df["sender"].map(domain_from_email)

    if "receiver_domain" in df.columns:
        df["receiver_domain_norm"] = df["receiver_domain"].map(clean_domain)
        missing_receiver = df["receiver_domain_norm"].eq("")
        df.loc[missing_receiver, "receiver_domain_norm"] = df.loc[missing_receiver, "receiver"].map(domain_from_email)
    else:
        df["receiver_domain_norm"] = df["receiver"].map(domain_from_email)

    df["subject_clean"] = df["subject"].map(normalize_space)
    df["body_clean"] = df["body"].map(normalize_space)
    df["text_clean"] = (df["subject_clean"] + " " + df["body_clean"]).map(normalize_space)
    df["canonical_text"] = [normalize_for_hash(s, b) for s, b in zip(df["subject_clean"], df["body_clean"])]
    df["canonical_hash"] = df["canonical_text"].map(sha256_text)
    df["text_length_chars"] = df["text_clean"].str.len().astype(int)
    df["token_count"] = df["text_clean"].map(lambda text: len(text.split())).astype(int)

    empty_text_mask = df["text_clean"].eq("")
    dropped_empty_text = int(empty_text_mask.sum())
    df = df.loc[~empty_text_mask].copy()

    url_parse = df["urls"].map(parse_urls)
    df["url_list_json"] = url_parse.map(lambda item: json.dumps([defang_url(url) for url in item[0]], ensure_ascii=True))
    df["url_domains_json"] = url_parse.map(lambda item: json.dumps(domains_from_urls(item[0]), ensure_ascii=True))
    df["url_indicator"] = url_parse.map(lambda item: item[1]).astype(int)
    df["url_count_parsed"] = url_parse.map(lambda item: len(item[0])).astype(int)

    for numeric_col, default in [
        ("url_count", 0),
        ("url_length_max", 0.0),
        ("url_length_avg", 0.0),
        ("url_subdom_max", 0.0),
        ("url_subdom_avg", 0.0),
        ("attachment_count", 0),
        ("has_attachments", 0),
    ]:
        if numeric_col not in df.columns:
            df[numeric_col] = default
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce").fillna(default)

    for optional_col in ["content_types", "attachment_types", "language"]:
        if optional_col not in df.columns:
            df[optional_col] = ""
        df[optional_col] = df[optional_col].map(normalize_space)

    conflict_hashes = (
        df.groupby("canonical_hash")["label"]
        .nunique()
        .loc[lambda series: series > 1]
        .index
    )
    conflict_mask = df["canonical_hash"].isin(conflict_hashes)
    dropped_conflict_rows = int(conflict_mask.sum())
    df = df.loc[~conflict_mask].copy()

    before_dedup = len(df)
    df = df.sort_values(["canonical_hash", "dataset", "original_index"]).drop_duplicates("canonical_hash", keep="first")
    removed_duplicate_rows = int(before_dedup - len(df))
    df = df.sort_values("row_id").reset_index(drop=True)

    columns = [
        "row_id",
        "dataset",
        "original_index",
        "source",
        "label",
        "canonical_hash",
        "sender_domain_norm",
        "receiver_domain_norm",
        "subject_clean",
        "body_clean",
        "text_clean",
        "text_length_chars",
        "token_count",
        "url_indicator",
        "url_count",
        "url_count_parsed",
        "url_length_max",
        "url_length_avg",
        "url_subdom_max",
        "url_subdom_avg",
        "url_list_json",
        "url_domains_json",
        "attachment_count",
        "has_attachments",
        "attachment_types",
        "content_types",
        "language",
    ]
    df = df[columns]

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREPROCESSED_DIR / f"{name}_preprocessed.parquet"
    df.to_parquet(output_path, index=False)

    result = DatasetResult(
        name=name,
        input_rows=input_rows,
        output_rows=len(df),
        dropped_missing_label=dropped_missing_label,
        dropped_empty_text=dropped_empty_text,
        dropped_conflict_rows=dropped_conflict_rows,
        removed_duplicate_rows=removed_duplicate_rows,
        label_counts={str(k): int(v) for k, v in df["label"].value_counts().sort_index().to_dict().items()},
        source_counts={str(k): int(v) for k, v in df["source"].value_counts(dropna=False).sort_index().to_dict().items()},
        missing_sender_domain_rate=float(df["sender_domain_norm"].eq("").mean()),
        missing_receiver_domain_rate=float(df["receiver_domain_norm"].eq("").mean()),
        output_path=str(output_path.relative_to(DATA_ROOT)),
    )
    return df, result


def split_frame(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return df[
        [
            "row_id",
            "dataset",
            "original_index",
            "source",
            "label",
            "canonical_hash",
        ]
    ].assign(split=split_name)


def write_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def create_meajor_source_split(df: pd.DataFrame) -> dict[str, Path]:
    split_root = SPLIT_DIR / "meajor_source_heldout"
    usable = df[df["source"].isin(["trec5", "trec6", "trec7"])].copy()
    train_pool = usable[usable["source"].isin(["trec5", "trec6"])].copy()
    test = usable[usable["source"].eq("trec7")].copy()

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    train_idx, val_idx = next(splitter.split(train_pool, train_pool["label"]))
    train = train_pool.iloc[train_idx].copy()
    val = train_pool.iloc[val_idx].copy()

    outputs = {
        "train": split_root / "train.csv",
        "val": split_root / "val.csv",
        "test": split_root / "test.csv",
    }
    write_split(split_frame(train, "train"), outputs["train"])
    write_split(split_frame(val, "val"), outputs["val"])
    write_split(split_frame(test, "test"), outputs["test"])
    return outputs


def create_meajor_random_split(df: pd.DataFrame) -> dict[str, Path]:
    split_root = SPLIT_DIR / "meajor_random_70_10_20"
    data = df.copy()

    first = StratifiedShuffleSplit(n_splits=1, test_size=RANDOM_TEST_FRACTION, random_state=SEED)
    train_val_idx, test_idx = next(first.split(data, data["label"]))
    train_val = data.iloc[train_val_idx].copy()
    test = data.iloc[test_idx].copy()

    val_fraction_within_train_val = RANDOM_VAL_FRACTION / (1.0 - RANDOM_TEST_FRACTION)
    second = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction_within_train_val, random_state=SEED)
    train_idx, val_idx = next(second.split(train_val, train_val["label"]))
    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    outputs = {
        "train": split_root / "train.csv",
        "val": split_root / "val.csv",
        "test": split_root / "test.csv",
    }
    write_split(split_frame(train, "train"), outputs["train"])
    write_split(split_frame(val, "val"), outputs["val"])
    write_split(split_frame(test, "test"), outputs["test"])
    return outputs


def summarize_split(path_map: dict[str, Path]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for split, path in path_map.items():
        frame = pd.read_csv(path)
        summary[split] = {
            "rows": int(len(frame)),
            "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().to_dict().items()},
            "source_counts": {str(k): int(v) for k, v in frame["source"].value_counts().sort_index().to_dict().items()},
            "path": str(path.relative_to(DATA_ROOT)),
            "sha256": sha256_file(path),
        }
    return summary


def compute_hash_overlaps(path_map: dict[str, Path]) -> dict[str, int]:
    frames = {split: pd.read_csv(path) for split, path in path_map.items()}
    overlaps: dict[str, int] = {}
    splits = sorted(frames)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            left_hashes = set(frames[left]["canonical_hash"])
            right_hashes = set(frames[right]["canonical_hash"])
            overlaps[f"{left}_vs_{right}"] = len(left_hashes.intersection(right_hashes))
    return overlaps


def markdown_counts(title: str, counts: dict[str, int]) -> list[str]:
    lines = [f"### {title}", "", "| Value | Count |", "|---|---:|"]
    for key, value in counts.items():
        lines.append(f"| `{key}` | {value:,} |")
    lines.append("")
    return lines


def write_reports(
    dataset_results: list[DatasetResult],
    source_split_summary: dict[str, dict[str, Any]],
    random_split_summary: dict[str, dict[str, Any]],
    source_hash_overlaps: dict[str, int],
    random_hash_overlaps: dict[str, int],
    manifest: dict[str, Any],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = REPORT_DIR / "preprocessing_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": SEED,
                "val_fraction": VAL_FRACTION,
                "random_split": {
                    "train": 0.70,
                    "val": 0.10,
                    "test": 0.20,
                },
                "datasets": [result.__dict__ for result in dataset_results],
                "source_split": source_split_summary,
                "random_split": random_split_summary,
                "source_split_hash_overlaps": source_hash_overlaps,
                "random_split_hash_overlaps": random_hash_overlaps,
                "manifest": manifest,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    lines = [
        "# Preprocessing Report",
        "",
        "## Scope",
        "",
        "This report records the preprocessing outputs. It does not include raw email examples, raw URLs, or message bodies.",
        "",
        "## Leakage Controls Applied",
        "",
        "- Labels were normalized to integer `0` and `1`.",
        "- Empty subject/body records were removed.",
        "- Exact duplicates were removed using canonical subject+body SHA-256 hashes before split generation.",
        "- Hash groups with conflicting labels were removed instead of resolved heuristically.",
        "- MeAJOR source-held-out split uses `trec5` and `trec6` for train/validation and `trec7` for test.",
        "- Split files contain row identifiers, labels, source values, and hashes, but no raw email text.",
        "- `source` is retained only in split metadata and diagnostics, not as a predictive feature.",
        "",
        "## Dataset Outputs",
        "",
        "| Dataset | Input Rows | Output Rows | Missing Label Dropped | Empty Text Dropped | Conflict Rows Dropped | Duplicate Rows Removed | Output Path |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for result in dataset_results:
        lines.append(
            f"| `{result.name}` | {result.input_rows:,} | {result.output_rows:,} | "
            f"{result.dropped_missing_label:,} | {result.dropped_empty_text:,} | "
            f"{result.dropped_conflict_rows:,} | {result.removed_duplicate_rows:,} | "
            f"`{result.output_path}` |"
        )
    lines.append("")

    for result in dataset_results:
        lines.extend(markdown_counts(f"{result.name} label counts", result.label_counts))
        lines.extend(markdown_counts(f"{result.name} source counts", result.source_counts))
        lines.append(
            f"- `{result.name}` missing sender-domain rate: {result.missing_sender_domain_rate:.4f}"
        )
        lines.append(
            f"- `{result.name}` missing receiver-domain rate: {result.missing_receiver_domain_rate:.4f}"
        )
        lines.append("")

    lines.extend(["## MeAJOR Source-Held-Out Split", "", "| Split | Rows | Label Counts | Source Counts | File | SHA-256 |", "|---|---:|---|---|---|---|"])
    for split, info in source_split_summary.items():
        lines.append(
            f"| `{split}` | {info['rows']:,} | `{info['label_counts']}` | "
            f"`{info['source_counts']}` | `{info['path']}` | `{info['sha256']}` |"
        )
    lines.append("")
    lines.extend(["Canonical hash overlap check:", ""])
    for pair, count in source_hash_overlaps.items():
        lines.append(f"- `{pair}`: {count}")
    lines.append("")

    lines.extend(["## MeAJOR Random Diagnostic Split", "", "| Split | Rows | Label Counts | Source Counts | File | SHA-256 |", "|---|---:|---|---|---|---|"])
    for split, info in random_split_summary.items():
        lines.append(
            f"| `{split}` | {info['rows']:,} | `{info['label_counts']}` | "
            f"`{info['source_counts']}` | `{info['path']}` | `{info['sha256']}` |"
        )
    lines.append("")
    lines.extend(["Canonical hash overlap check:", ""])
    for pair, count in random_hash_overlaps.items():
        lines.append(f"- `{pair}`: {count}")
    lines.append("")

    lines.extend(["## Artifact Manifest", "", "| Artifact | SHA-256 |", "|---|---|"])
    for artifact, digest in manifest.items():
        lines.append(f"| `{artifact}` | `{digest}` |")
    lines.append("")

    report_path = REPORT_DIR / "preprocessing_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    dataset_results: list[DatasetResult] = []
    normalized: dict[str, pd.DataFrame] = {}
    for name in ["meajor", "ceas_08", "nazario_5"]:
        frame, result = normalize_dataset(name)
        normalized[name] = frame
        dataset_results.append(result)

    source_paths = create_meajor_source_split(normalized["meajor"])
    random_paths = create_meajor_random_split(normalized["meajor"])
    source_summary = summarize_split(source_paths)
    random_summary = summarize_split(random_paths)
    source_hash_overlaps = compute_hash_overlaps(source_paths)
    random_hash_overlaps = compute_hash_overlaps(random_paths)

    artifacts = list(PREPROCESSED_DIR.glob("*.parquet"))
    artifacts.extend(path for path in source_paths.values())
    artifacts.extend(path for path in random_paths.values())

    manifest = {
        str(path.relative_to(DATA_ROOT)): sha256_file(path)
        for path in sorted(artifacts)
    }
    write_reports(dataset_results, source_summary, random_summary, source_hash_overlaps, random_hash_overlaps, manifest)

    print("Preprocessing complete.")
    print(f"Report: {REPORT_DIR / 'preprocessing_report.md'}")
    print(f"Summary: {REPORT_DIR / 'preprocessing_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize the MeAJOR, CEAS-08 and Nazario_5 email corpora and write split files.")
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Split seed. Only the fixed value 20260503 is accepted.",
    )
    args = parser.parse_args()
    if args.seed != SEED:
        raise SystemExit(f"This protocol currently requires the fixed seed {SEED}; got {args.seed}.")
    run()


if __name__ == "__main__":
    main()
