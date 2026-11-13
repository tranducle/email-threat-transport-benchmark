# Email Threat Transport Benchmark

Companion code, frozen experimental protocols, and verification suite for the manuscript:

> **Ranking Transportability, Common-Threshold Compatibility, and Residual Structural Utility in Email Threat Detection: A Cross-Corpus Measurement Study**

The repository also holds two follow-up studies:

- [`threshold_transport/`](threshold_transport/README.md): code for **A Good Ranking Is Not a Transferable Cutoff: Multi-Family False-Positive Control of Email Filters Across Sources, Priors, and Time**. It covers six classifiers from three model families, prior-shift recalibration, Clopper-Pearson certified cutoffs, forward chronological certification on SpaPhish v5 and a sender-domain cluster bootstrap on CEAS-08. The pinned environment is [`environment.lock.json`](environment.lock.json).
- [`extension/`](extension/README.md): an earlier version of that study (CEAS-08 external scoring, a label-budget analysis, shared versus same-model cutoffs), with frozen score files, label-only tables and result tables.

## Datasets

No email text is stored in this repository. All corpora are public; download them from the records below.

| Corpus | Used by | Record | License |
| :- | :- | :- | :- |
| MeAJOR (TREC 2005, 2006 and 2007 public spam corpora, merged and preprocessed) | both studies | Zenodo, version 2.0, DOI [10.5281/zenodo.18471483](https://doi.org/10.5281/zenodo.18471483) | CC BY 4.0 |
| SpaPhish v5 (Spanish phishing and legitimate email) | `threshold_transport/` | Mendeley Data, DOI [10.17632/hz2d6gz7pc.5](https://doi.org/10.17632/hz2d6gz7pc.5) | CC BY 4.0 |
| CEAS-08 (from "Phishing Email Curated Datasets") | `threshold_transport/`, `extension/` | Zenodo, DOI [10.5281/zenodo.8339691](https://doi.org/10.5281/zenodo.8339691) | CC BY 4.0 |

The TREC corpora inside MeAJOR come from the TREC 2005 and 2006 Spam Tracks ([2005 overview](https://trec.nist.gov/pubs/trec14/papers/SPAM.OVERVIEW.pdf), [2006 overview](https://trec.nist.gov/pubs/trec15/papers/SPAM06.OVERVIEW.pdf)). File names, sizes, checksums and the expected local layout are in [DATA.md](DATA.md).

## Overview

Cross-corpus email threat detection benchmarks frequently conflate ranking discrimination (`AUPRC`) with fixed-threshold operating compatibility (`Recall@FPR <= 1%`). This repository provides the three-stage measurement framework and frozen evaluation artifacts used to audit:

1. **Stage 1 (Semantic Anchor Transportability)**: Reciprocal source-held-out cross-corpus evaluation between `trec5` (`N = 46,640`, `41.24%` positive) and `trec6` (`N = 14,685`, `23.85%` positive; pooled `N = 61,325`), comparing sparse lexical TF-IDF + Logistic Regression (`A1_TFIDF_LR`) against a 3-seed multilingual transformer ensemble (`A2_MDISTILBERT`, `distilbert-base-multilingual-cased`) under a conservative source-locked threshold rule (`tau* = max(tau_trec5, tau_trec6)` at `FPR <= 1%`).
2. **Stage 2 (De-Identified Residual Structural Utility)**: Nested cross-fitted residual ridge probes (`alpha = 1.0`, logit-additive fusion weight `c in {0.0, 0.25, 0.5, 1.0}`) testing whether 17 sender-agnostic structural features across URL structure (`U`, 7 features), domain shape without identity (`D`, 4 features), and message/MIME layout (`M`, 6 features) improve cross-corpus generalization above `A2_MDISTILBERT`.
3. **Stage 3 (Paired Stratified Bootstrap Inference)**: Non-parametric paired class-stratified row bootstrap (`B = 1,000`, `seed = 20261024`) characterizing within-source sampling uncertainty for all primary contrasts.

## Key Empirical Findings

| Stage | Model / Block | `trec6` (`N = 14,685`) | `trec5` (`N = 46,640`) | Principal Finding |
| :- | :- | :- | :- | :- |
| **Stage 1** | `A1_TFIDF_LR` (`tau* = 0.8367`) | `AUPRC = 0.9281`, `Recall = 0.6957`, `FPR = 0.0099` | `AUPRC = 0.9592`, `Recall = 0.6337`, `FPR = 0.0090` | Moderate ranking with non-zero recall at the shared threshold on both corpora |
| **Stage 1** | `A2_MDISTILBERT` (`tau* = 0.9996`) | `AUPRC = 0.9661`, `Recall = 0.8553`, `FPR = 0.0099` | `AUPRC = 0.9714`, `Recall = 0.0000`, `FPR = 0.0000` | Superior ranking and Brier/ECE-15 on both sources, but severe cross-source score-scale shift collapses `trec5` recall to `0.0000` (`TP = 0 / 19,235`) at the shared threshold |
| **Stage 2** | `U_URL_STRUCTURE` (`c = 1.0`) | `dAUPRC = +0.000066`, `dLogLoss = +0.000143` | `dAUPRC = +0.000052`, `dLogLoss = -0.000034` | Fails cross-source log-loss consistency (`dLogLoss > 0` on `trec6`) |
| **Stage 2** | `D_DOMAIN_SHAPE_NO_IDENTITY` (`c = 1.0`) | `dAUPRC = +0.000022`, `dLogLoss = +0.000226` | `dAUPRC = -0.000005`, `dLogLoss = +0.000136` | Worsens log-loss on both sources; negative `dAUPRC` on `trec5` |
| **Stage 2** | `M_MESSAGE_STRUCTURE` (`c = 1.0`) | `dAUPRC = +0.000042`, `dLogLoss = +0.000151` | `dAUPRC = +0.000010`, `dLogLoss = +0.000107` | Worsens log-loss on both sources |

## Repository Structure

```text
email-threat-transport-benchmark/
├── README.md                              # Study overview, findings, and quickstart
├── DATA.md                                # MeAJOR dataset schema and preparation guide
├── REPRODUCIBILITY.md                     # Step-by-step verification and retraining guide
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
├── configs/
│   ├── semantic_anchor_protocol.json      # Frozen Stage 1 specification
│   ├── residual_utility_protocol.json     # Frozen Stage 2 specification
│   └── published_results_summary.json     # Complete numerical results for Tables I to III
├── src/
│   ├── metrics_and_thresholds.py          # Threshold selection, ECE-15, and binary metrics
│   ├── structural_features.py             # Definitions and extraction for blocks U, D, M
│   └── bootstrap_inference.py             # Paired class-stratified bootstrap engine
├── experiments/
│   ├── semantic_anchor_study.py           # Stage 1 evaluation entrypoint
│   ├── residual_utility_study.py          # Stage 2 residual probe entrypoint
│   └── uncertainty_analysis.py            # Stage 3 bootstrap interval entrypoint
├── scripts/
│   ├── prepare_meajor_corpus.py           # Verify local MeAJOR preprocessed parquet table
│   └── verify_published_results.py        # Deterministic check of all manuscript claims
├── tests/
│   └── test_protocol_invariants.py        # Automated test suite
├── docs/
│   └── PROTOCOL_NOTES.md                  # Source quarantine rules and calibration details
├── extension/                             # Earlier follow-up: CEAS-08 scoring, label budget,
│                                          # shared versus same-model cutoffs (see its README)
├── threshold_transport/                   # Multi-family threshold transport study (see its README)
└── environment.lock.json                  # Pinned environment of the threshold transport runs
```

## Quick Verification (No Dataset Download Required)

To verify the protocol invariants, unit tests, and all published numerical metrics from the frozen evaluation summaries:

```bash
pip install -r requirements.txt
pytest -q
python scripts/verify_published_results.py
python experiments/semantic_anchor_study.py --verify-frozen
python experiments/residual_utility_study.py --verify-frozen
python experiments/uncertainty_analysis.py --verify-frozen
```

See [DATA.md](DATA.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for full retraining instructions on the public MeAJOR benchmark.

## Threshold transport study

The `threshold_transport/` folder holds the code for the manuscript "A Good Ranking Is Not a Transferable Cutoff: Multi-Family False-Positive Control of Email Filters Across Sources, Priors, and Time": deterministic preprocessing, anchor and multi-family training, prior-shift recalibration, chronological certification and the sender-domain cluster bootstrap. The pinned environment of those runs is `environment.lock.json` at the repository root. Data sources, data layout, run order and the mapping from scripts to tables are in [threshold_transport/README.md](threshold_transport/README.md).
