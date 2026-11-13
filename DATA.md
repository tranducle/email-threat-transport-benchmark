# Dataset Specification and Preparation (`DATA.md`)

## 1. Benchmark Source

The experiments in `experiments/` use the public **MeAJOR** multi-source email benchmark (Zenodo, version 2.0, DOI [10.5281/zenodo.18471483](https://doi.org/10.5281/zenodo.18471483), CC BY 4.0). The studies in `threshold_transport/` and `extension/` also use SpaPhish v5 and CEAS-08; see section 4. Raw email payloads and full-row parquet files are not redistributed in this repository to respect upstream corpus licensing and repository size limits.

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

## 4. Datasets for the Threshold Transport Study (`threshold_transport/`)

| Corpus | Record | File | Size and checksum | License |
| :- | :- | :- | :- | :- |
| MeAJOR | Zenodo, "MeAJOR: Merged email Assets from Joint Open-source Repositories", version 2.0, DOI [10.5281/zenodo.18471483](https://doi.org/10.5281/zenodo.18471483) | `meajor_cleaned_preprocessed.parquet.gzip` | 86,278,230 bytes, MD5 `78e397ad8447bcdba5a98097921ba8bd` | CC BY 4.0 |
| SpaPhish v5 | Mendeley Data, "SpaPhish: A Spanish Dataset for Phishing and Psychological Pattern Detection", version 5, DOI [10.17632/hz2d6gz7pc.5](https://doi.org/10.17632/hz2d6gz7pc.5) | `Spaphish dataset - DiB.csv` | 5,098,917 bytes, SHA-256 `656b2245d58da72d640680e5c2a168673a130b38607f2a427c773bbb167e995e` | CC BY 4.0 |
| CEAS-08 | Zenodo, "Phishing Email Curated Datasets", DOI [10.5281/zenodo.8339691](https://doi.org/10.5281/zenodo.8339691) | `CEAS_08.csv` | 67,902,056 bytes, MD5 `1f0d59191eec892a709995d48cd8decb` | CC BY 4.0 |

TREC5 and TREC6 are the TREC 2005 and 2006 Spam Track corpora as distributed inside MeAJOR ([2005 overview](https://trec.nist.gov/pubs/trec14/papers/SPAM.OVERVIEW.pdf), [2006 overview](https://trec.nist.gov/pubs/trec15/papers/SPAM06.OVERVIEW.pdf)). SpaPhish v5 above is the standalone Spanish corpus; it is not the `spaphish` source inside MeAJOR described in section 2.

After preprocessing, the study uses 46,640 TREC5 and 14,685 TREC6 messages for development, 1,395 SpaPhish v5 messages and 38,920 CEAS-08 messages for external evaluation. The scripts check these counts and the checksums above and stop when they differ. The expected folder layout and run order are in [threshold_transport/README.md](threshold_transport/README.md).
