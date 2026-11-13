# Reproducibility

## Environment

The original runs used Python 3.12.10 on one NVIDIA GeForce RTX 3090 (24 GB, CUDA 12.4) under Windows 11. `requirements.txt` installs the pinned package versions; `environment.lock.json` records the full environment (all installed packages, GPU, CUDA and platform).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`torch` must be a CUDA build for the DistilBERT steps (the original runs used `torch 2.6.0+cu124`). The first DistilBERT run downloads `distilbert-base-multilingual-cased` from the Hugging Face Hub.

## Checks without data

```bash
python -m unittest discover -s threshold_transport/tests
```

The three experiment scripts also accept `--smoke` for a short run; the smoke run of `prior_shift_and_seed_ceilings.py` needs CUDA.

## Full run

Place the corpora as described in [DATA.md](DATA.md), then run from the repository root in this order. The exact commands, the flags of each step and the output files are in [threshold_transport/README.md](threshold_transport/README.md#run-order).

| Step | Script | Hardware | Time on the original machine |
| :- | :- | :- | :- |
| 1 | `preprocessing/build_trec_development_set.py` | CPU | under a minute |
| 2 | `training/train_anchor_models.py` (`audit`, `fit-folds`, `final-fit`) | GPU for DistilBERT | |
| 3 | `training/crossfit_anchor_scores.py` (`crossfit`, optional `residual-fit`) | GPU | |
| 4 | `preprocessing/build_trec_development_set.py` again (checks row order and labels against the cross-fitted scores) | CPU | under a minute |
| 5 | `external_corpora/spaphish_adapter_and_scoring.py`, `rebuild_spaphish_url_features.py`, `ceas08_adapter_and_scoring.py` (`prepare`, then `score`) | CPU, then GPU | |
| 6 | `training/time_anchor_training.py` | GPU | about 45 min |
| 7 | `experiments/prior_shift_and_seed_ceilings.py` | GPU | about 14 min |
| 7 | `experiments/multifamily_training_and_probes.py` | GPU | about 60 min |
| 7 | `experiments/chronological_and_cluster_certification.py` | CPU | about 20 min |

## Determinism

cuDNN runs in deterministic mode, and the scripts write JSON receipts with SHA-256 digests of their inputs and outputs under `results/receipts/` in the data root. Retraining the balanced logistic regression anchor (A1) reproduces the saved model exactly, and rescoring saved DistilBERT checkpoints reproduces the saved ensemble scores to within 1e-5. Retraining DistilBERT with the same seeds on the same GPU does not: the ensemble scores differed from the saved ones by 0.0008 on average and by up to 0.31 on 512 development messages. The DistilBERT numbers in the manuscript come from the saved checkpoints.

Seeds and fixed settings are listed in [threshold_transport/README.md](threshold_transport/README.md#seeds-and-fixed-settings).
