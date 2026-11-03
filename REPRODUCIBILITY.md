# Reproducibility Guide (`REPRODUCIBILITY.md`)

## 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Tier 1: Deterministic Verification of Published Claims (CPU, < 10 seconds)

This tier verifies the mathematical invariants, threshold selection rules, feature block schemas, and every number reported in Tables I, II, III and Figures 2 and 3 of the manuscript:

```bash
pytest -q
python scripts/verify_published_results.py
python experiments/semantic_anchor_study.py --verify-frozen
python experiments/residual_utility_study.py --verify-frozen
python experiments/uncertainty_analysis.py --verify-frozen
```

## 3. Tier 2: Full Cross-Corpus Retraining from MeAJOR Preprocessed Data

1. Place `data/meajor_preprocessed.parquet` following [DATA.md](DATA.md) and verify partition counts:
   ```bash
   python scripts/prepare_meajor_corpus.py --data-path data/meajor_preprocessed.parquet
   ```
2. Run the Stage 1 lexical anchor reciprocal folds (`fit_trec5__val_trec6` and `fit_trec6__val_trec5`):
   ```bash
   python experiments/semantic_anchor_study.py --data-path data/meajor_preprocessed.parquet --output-dir outputs/stage1
   ```
3. Frozen protocol hyperparameters for `A2_MDISTILBERT` (`distilbert-base-multilingual-cased`, `max_length = 192`, `epochs = 3`, `batch_size = 16`, `lr = 2e-5`, `seeds = [20260503, 20260504, 20260505]`) and Stage 2 residual ridge probes (`alpha = 1.0`, `coefficient_grid = [0.0, 0.25, 0.5, 1.0]`, `selection_seed = 20261025`) are locked in `configs/semantic_anchor_protocol.json` and `configs/residual_utility_protocol.json`.
