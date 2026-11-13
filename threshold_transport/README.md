# Threshold transport across email corpora, priors and time

This folder contains the code for the manuscript "A Good Ranking Is Not a Transferable Cutoff: False-Positive Control of Spam and Phishing Filters Under Dataset Shift". The study fixes a decision cutoff on two development sources (TREC5 and TREC6 from the MeAJOR collection) at a target false-positive rate, freezes it, and measures what happens to that cutoff on SpaPhish v5 and CEAS-08, under prior shift, and forward in time. It covers six classifiers in three model families, threshold policies that use increasing amounts of target information (from the frozen development cutoff to prior-shift recalibration and a Clopper-Pearson certified cutoff), exchangeable label-budget simulations, forward chronological certification on SpaPhish, a sender-domain cluster bootstrap on CEAS-08, and cross-fitted residual probes over the mDistilBERT scores.

The rest of this repository belongs to an earlier study and is not needed here.

## Requirements

Python 3.12 with the versions in `requirements.txt` in this folder. The full environment of the original runs (all installed packages, GPU, CUDA and platform) is in `environment.lock.json` at the repository root.

```bash
pip install -r threshold_transport/requirements.txt
```

`torch` must be a CUDA build for the steps marked GPU below (the original runs used `torch 2.6.0+cu124`). The first A2 run downloads `distilbert-base-multilingual-cased` from the Hugging Face Hub.

## Data

No email data is included. Download the following public files and place them under the data root (layout below).

| Corpus | Source | File | Check |
| :- | :- | :- | :- |
| MeAJOR (TREC5, TREC6; TREC7 only for an optional diagnostic) | Zenodo, "MeAJOR: Merged email Assets from Joint Open-source Repositories", version 2.0, DOI [10.5281/zenodo.18471483](https://doi.org/10.5281/zenodo.18471483) (concept DOI [10.5281/zenodo.18471482](https://doi.org/10.5281/zenodo.18471482)), CC BY 4.0 | `meajor_cleaned_preprocessed.parquet.gzip` | 86,278,230 bytes; MD5 `78e397ad8447bcdba5a98097921ba8bd` (Zenodo listing); SHA-256 of the copy used in the study `5a9de6c207634f068928ed0cc68d78ee2fd594ed9a587daf4048974c95715c1a` |
| SpaPhish v5 | Mendeley Data, version 5, DOI [10.17632/hz2d6gz7pc.5](https://doi.org/10.17632/hz2d6gz7pc.5), CC BY 4.0 | `Spaphish dataset - DiB.csv` | 5,098,917 bytes; SHA-256 `656b2245d58da72d640680e5c2a168673a130b38607f2a427c773bbb167e995e` (checked by the adapter) |
| CEAS-08 | Zenodo, "Phishing Email Curated Datasets", DOI [10.5281/zenodo.8339691](https://doi.org/10.5281/zenodo.8339691) (concept DOI [10.5281/zenodo.8339690](https://doi.org/10.5281/zenodo.8339690)), CC BY 4.0 | `CEAS_08.csv` | 67,902,056 bytes; MD5 `1f0d59191eec892a709995d48cd8decb`; 39,154 rows (21,842 spam, 17,312 ham), columns `sender, receiver, date, subject, body, label, urls` (checked by the adapter) |
| Nazario_5 (only for the full `preprocess_corpora.py` run, not used by the experiments) | same Zenodo record as CEAS-08 | `Nazario_5.csv` | 11,755,517 bytes; MD5 `45db8330ea4aabbf72f5199949ae03e5` |

After normalization and contamination screening the experiments use:

- TREC5: 46,640 messages (19,235 positive, 27,405 negative); TREC6: 14,685 messages (3,503 positive, 11,182 negative). The scripts stop if these source counts differ.
- SpaPhish v5: 1,395 messages (731 positive, 664 negative), no contamination match. Chronological split: `pre_2023` 438 messages (400 negative, 38 positive), `2023_to_2025` 933 messages (264 negative, 669 positive), 24 undated messages.
- CEAS-08: 38,920 messages (21,674 positive, 17,246 negative) after removing 191 internal exact duplicates and 43 messages at or above 0.95 character 5-gram similarity to TREC5/TREC6. The retained messages form 16,729 sender-domain clusters (16,728 parsed domains plus one pooled cluster of the 925 messages without a domain).

## Data root and layout

Every script resolves its inputs and outputs from one data root: the environment variable `THRESHOLD_TRANSPORT_DATA` if set, otherwise `data/` at the repository root (ignored by Git).

```text
<data root>/
  raw/
    meajor_cleaned_preprocessed.parquet.gzip
    CEAS_08.csv
    Nazario_5.csv                          (optional)
    SpaPhish_v5/Spaphish dataset - DiB.csv
  processed/
    meajor_preprocessed.parquet            normalized MeAJOR table (all sources)
    meajor_trec56_reconstructed.parquet    TREC5+TREC6 development set
    spaphish_confirmation.parquet          SpaPhish adapter table (+ _split.csv)
    spaphish_confirmation_urlfixed.parquet SpaPhish table with rebuilt URL features
    ceas08_confirmation.parquet            CEAS-08 adapter table (+ _split.csv)
  models/
    anchors/                               A1 and A2: fit_trec5__val_trec6/, fit_trec6__val_trec5/,
                                           final_models/, fold and final-fit receipts
    crossfit/                              cross-fitted A2 scores (crossfit/<source>_a2_crossfit.parquet)
  scores/
    spaphish_frozen_scores.parquet, ceas08_frozen_scores.parquet
    a2_per_seed_<corpus>.parquet, exp2_all_families_<corpus>.parquet
  results/
    exp1_*.csv, exp2_*.csv, exp3_*.csv, anchor_training_timing.json,
    spaphish_url_residual_urlfixed.json
    receipts/                              JSON receipts with SHA-256 digests of inputs and outputs
```

Result files keep the names used for the paper. The prefixes `exp1_`, `exp2_` and `exp3_` name the experiment script that writes them (prior shift and seed ceilings, multi-family training and probes, chronological and cluster certification).

## Run order

Run from the repository root. Steps marked GPU need CUDA.

```bash
# 1. Normalize MeAJOR and build the TREC5+TREC6 development set (CPU, under a minute)
python threshold_transport/preprocessing/build_trec_development_set.py

# 2. Anchor models: data check, source-held-out folds, final fit (GPU for A2)
python threshold_transport/training/train_anchor_models.py --phase audit
python threshold_transport/training/train_anchor_models.py --phase fit-folds
python threshold_transport/training/train_anchor_models.py --phase final-fit

# 3. Cross-fitted A2 scores on the development sources (GPU); the residual-fit phase is an
#    optional linear probe summary (the probe results in the paper come from step 7)
python threshold_transport/training/crossfit_anchor_scores.py --phase crossfit
python threshold_transport/training/crossfit_anchor_scores.py --phase residual-fit

# 4. Rerun step 1: it now also checks row order and labels against the cross-fitted scores
python threshold_transport/preprocessing/build_trec_development_set.py

# 5. External corpora: adapters (CPU) and scoring with the frozen anchors (GPU)
python threshold_transport/external_corpora/spaphish_adapter_and_scoring.py prepare
python threshold_transport/external_corpora/spaphish_adapter_and_scoring.py score
python threshold_transport/external_corpora/rebuild_spaphish_url_features.py
python threshold_transport/external_corpora/ceas08_adapter_and_scoring.py prepare
python threshold_transport/external_corpora/ceas08_adapter_and_scoring.py score

# 6. Training time of A1 and A2 (GPU, about 45 min; retrains in a scratch folder, then deletes it).
#    Writes results/anchor_training_timing.json, which step 7 reads for the A1/A2 train_seconds
#    of exp2_latency_cost.csv (written as NaN, with a message, if the file is missing).
python threshold_transport/training/time_anchor_training.py --model-dir <empty scratch folder>

# 7. Experiments
python threshold_transport/experiments/prior_shift_and_seed_ceilings.py           # GPU, about 14 min
python threshold_transport/experiments/multifamily_training_and_probes.py         # GPU, about 60 min
python threshold_transport/experiments/chronological_and_cluster_certification.py # CPU, about 20 min

# Unit tests for the URL parser
python -m unittest discover -s threshold_transport/tests
```

The three experiment scripts accept `--smoke` for a short check. The smoke check of `prior_shift_and_seed_ceilings.py` needs CUDA. `time_anchor_training.py` requires CUDA; with `--smoke` it trains one A2 seed on 256 rows and writes no result. The run times above are those of the original runs on one RTX 3090.

`spaphish_adapter_and_scoring.py prepare` keeps the URL parser used when the SpaPhish table was first built. That parser returns no URL for SpaPhish's `[url1],[url2]` format, so the URL features in `spaphish_confirmation.parquet` are all zero. `rebuild_spaphish_url_features.py` rebuilds them with `url_parser.py`, leaves every other column unchanged, and the experiments read the rebuilt table.

`train_anchor_models.py --phase score-retrospective` scores TREC7 with the final anchors; it is a diagnostic and no result in the paper depends on it. `preprocess_corpora.py`, run on its own, normalizes MeAJOR, CEAS-08 and Nazario_5 and writes text-free split files that the experiments do not use.

## Hardware

All runs used one workstation with an NVIDIA GeForce RTX 3090 (24 GB, CUDA 12.4) under Windows 11, Python 3.12.10. A2 fine-tuning, A2 scoring and the per-seed rescoring need a CUDA GPU. TF-IDF models, the adapters, the residual probes and the chronological and cluster analyses run on CPU. cuDNN runs in deterministic mode; rescoring the saved A2 checkpoints reproduces the saved ensemble scores to within 1e-5. Retraining is a different matter: rerunning A1 training reproduces the saved model exactly, but rerunning A2 fine-tuning with the same seeds on the same GPU gave ensemble scores that differ from the saved ones by 0.0008 on average and by up to 0.31 on 512 development messages. Use the saved A2 checkpoints to reproduce the reported A2 numbers.

## Seeds and fixed settings

- Classifier training (A2 seeds, per-family model seeds), with the A2 ensemble as the mean of the three seed probabilities: 20260503, 20260504, 20260505.
- Residual-probe cross-fitting split: 20261024. Probe coefficient selection split: 20261025.
- Row-level and sender-domain cluster bootstrap resamples and the 10,000-draw label-budget simulation: 20261112.
- SpaPhish URL residual bootstrap: 20261105. Sample for the A2 timing check: 20261113.
- FPR budgets alpha in {0.005, 0.01, 0.02}; the shared development cutoff of each model is the larger of its two source-wise cutoffs.

Anchor settings are in `configs/anchor_models.json` and the residual-probe settings in `configs/residual_probe.json`.

## Which script produces which table or figure

| Paper item (by content) | Script | Output |
| :- | :- | :- |
| Corpora after normalization and contamination screening, SpaPhish chronological split | `build_trec_development_set.py`, both adapter scripts (`prepare`) | `results/receipts/trec_development_set.json`, `spaphish_adapter.json`, `ceas08_adapter.json` |
| Ranking, probability quality and frozen-cutoff performance of the six classifiers; SpaPhish decoupling figure | `multifamily_training_and_probes.py` (with `prior_shift_and_seed_ceilings.py` for the A1/A2 policy rows of the figure) | `exp2_multifamily_metrics.csv`, `exp2_multifamily_contrasts_holm.csv`, `exp2_multifamily_prior_shift.csv`, `exp1_prior_shift_table.csv` |
| Row-level versus sender-domain cluster bootstrap on CEAS-08 | `chronological_and_cluster_certification.py` | `exp3_ceas08_cluster_bootstrap.csv` |
| Convex probability blends | `multifamily_training_and_probes.py` | `exp2_blend_sensitivity.csv` |
| Computational footprint | `time_anchor_training.py` (A1 and A2 training times), then `multifamily_training_and_probes.py` | `anchor_training_timing.json`, `exp2_latency_cost.csv` |
| Per-seed score ceilings and reciprocal cutoff feasibility; cutoff feasibility and repair figure | `prior_shift_and_seed_ceilings.py`, `multifamily_training_and_probes.py` | `exp1_seed_ceiling_audit.csv`, `exp1_reciprocal_feasibility.csv`, `exp2_multifamily_reciprocal_feasibility.csv`, `exp1_prior_shift_table.csv` |
| Realized FPR under the threshold policies on the external corpora (prior-shift table) | `prior_shift_and_seed_ceilings.py`, `multifamily_training_and_probes.py` | `exp1_prior_shift_table.csv`, `exp1_prior_shift_bootstrap_contrasts.csv`, `exp2_multifamily_prior_shift.csv` |
| Cross-fitted residual probe audit and residual probe figure | `multifamily_training_and_probes.py` | `exp2_nonlinear_residuals.csv` |
| Certified cutoff under exchangeable label-budget draws; label-budget figure | `chronological_and_cluster_certification.py` | `exp3_exchangeable_e1_summary.csv` |
| Empirical versus certified cutoffs; forward chronological certification across routes and FPR caps | `chronological_and_cluster_certification.py` | `exp3_exchangeable_e1_summary.csv`, `exp3_spaphish_chronological.csv` (the route-by-cap table uses the rows with `certifiable == True`, which drops the 54 rows at alpha 0.005, a cap that is not certifiable on this split) |

Model specifications come from `configs/anchor_models.json` and the model definitions in `multifamily_training_and_probes.py`.
