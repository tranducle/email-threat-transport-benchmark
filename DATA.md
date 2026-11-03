# Dataset Specification and Preparation (`DATA.md`)

## 1. Benchmark Source

All experiments use the public **MeAJOR** multi-source email benchmark. Raw email payloads and full-row parquet files are not redistributed in this repository to respect upstream corpus licensing and repository size limits.

## 2. Expected Local Layout

Place the preprocessed MeAJOR parquet table at `data/meajor_preprocessed.parquet`:

```text
data/
└── meajor_preprocessed.parquet
```

### Active Reciprocal Development Sources (`N = 61,325`)

- `trec6`: `N = 14,685` messages (`3,503` positives, `11,182` negatives, positive prevalence `23.85%`).
- `trec5`: `N = 46,640` messages (`19,235` positives, `27,405` negatives, positive prevalence `41.24%`).

### Quarantined / Sealed Sources

As documented in `configs/semantic_anchor_protocol.json`:
- `spaphish`: Sealed cross-language source (`100%` positive class, zero negatives, making `FPR` undefined); never accessed during Stage 1 or Stage 2 fitting or selection.
- `trec7`: Retrospective diagnostic source only; never used for model training, hyperparameter tuning, or threshold selection.

## 3. Required Schema Columns

`data/meajor_preprocessed.parquet` must contain:
- `row_id`, `source`, `label` (`0` = benign, `1` = phishing/malicious), `text_clean`, `canonical_hash`
- `sender_domain_norm`, `receiver_domain_norm`, `url_domains_json`
- `url_indicator`, `url_count`, `url_count_parsed`, `url_length_max`, `url_length_avg`, `url_subdom_max`, `url_subdom_avg`
- `text_length_chars`, `token_count`, `attachment_count`, `has_attachments`, `content_types`, `attachment_types`

Verify your local file with:

```bash
python scripts/prepare_meajor_corpus.py --data-path data/meajor_preprocessed.parquet
```
