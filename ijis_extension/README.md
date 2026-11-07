# Extension: CEAS-08 external scoring and label budget for restoring a false-positive budget

This folder holds the code, frozen score files, label tables, and result tables for the two
analyses added in the extended version of the study:

> **A Good Ranking Is Not a Transferable Cutoff: Source Dependence of False-Positive Control in Email Filters**

* **E2, CEAS-08 external scoring.** The frozen A1 (TF-IDF + logistic regression) and A2
  (3-seed multilingual DistilBERT ensemble) models from the three training routes (`final`,
  `fit_trec5`, `fit_trec6`) score CEAS-08 (38,920 messages after removing 191 duplicate and 43
  development-contaminated messages from 39,154 rows). No model is retrained, re-seeded, or
  re-tuned, and no CEAS-08 label is used for fitting, calibration, or cutoff selection.
* **E1, label budget.** For each target stream (TREC6, TREC5, SpaPhish v5, CEAS-08), five cutoff
  policies are compared at false-positive budgets of 0.5%, 1%, and 2% with k in
  {50, 100, 200, 500, 1000, 2000} labeled target messages and 1,000 draws per cell (seed
  20261107): P0 frozen development cutoff, P1 empirical target cutoff, P2 one-sided
  Clopper-Pearson certified cutoff (95%), P3 rank-quantile transfer, and P4 Platt scaling.

## Layout

The folder mirrors the project workspace, so the scripts' relative paths resolve unchanged:

```text
ijis_extension/
  ijis_e1e2_20261106/
    code/          e2_ceas08.py, e1_label_budget.py, e1_tables.py, verify_e1e2.py, job_wrapper.py
    data/          ceas08_confirmation.parquet (label-only), ceas08_confirmation_split.csv
    predictions/   ceas08_frozen_scores.parquet (row_id and six frozen score columns)
    receipts/      adapter freeze and scoring receipts (hashes of models, data, and code)
    results/       E1 and E2 result tables and summaries, verification output
  rebaseline_rb_g1/rb_g1_semantic_anchor.py  development training, scoring, and cutoff rule
  rebaseline_rb_g1/runs/RB_G1_20261023_V1/
    final_fit_receipt.json                 locked development cutoffs
    fit_*__val_*/A*/..._validation.parquet  development validation scores (row_id, label, score)
  top_q1_external_confirmation_20261105/
    data/          spaphish_confirmation.parquet (label-only)
    predictions/   spaphish_frozen_scores.parquet
  LABEL_ONLY_DERIVATION.json   source hash, kept columns, and row count of each label-only table
  SHA256SUMS.txt               SHA-256 of every file in this folder
```

## What is and is not released

No email text is released. Score files contain only `row_id` and model scores. The two
label-only tables keep `row_id`, `label`, the contamination flag (CEAS-08), and a SHA-256 hash of
the canonicalized message; they were derived from the full adapter tables, and each kept column
was checked to be identical to its source column. CEAS-08 is redistributed by the Phishing Email
Curated Datasets on Zenodo (doi:10.5281/zenodo.8339691) and SpaPhish v5 by Mendeley Data
(doi:10.17632/hz2d6gz7pc.5), both under CC BY 4.0. The raw corpora, full adapter tables, and model
weights are not redistributed. Absolute workstation paths in receipts were rewritten as
`<workspace>/...`; the recorded SHA-256 values refer to the original files.

## Running the analyses

```bash
pip install numpy pandas pyarrow scipy scikit-learn joblib
cd ijis_extension
python ijis_e1e2_20261106/code/e1_label_budget.py    # E1, CPU only, under a minute on one core
python ijis_e1e2_20261106/code/e1_tables.py          # E1 tables from the long results file
python ijis_e1e2_20261106/code/e2_ceas08.py analyze  # E2 metrics and paired bootstrap from frozen scores
```

E1 reads only the score files and the `row_id`, `label`, and contamination columns of the label
tables, all of which are included. The E2 `analyze` action reads the frozen scores, the CEAS-08
label table, the locked cutoffs, and the development validation scores, and imports the cutoff
rule from `rebaseline_rb_g1/rb_g1_semantic_anchor.py` (the development training and scoring
script, included; it also needs `joblib`). These commands are listed from the scripts' input
paths; they were not re-executed on this released copy. The E2 `prepare` and
`score` actions and `verify_e1e2.py` additionally need the raw CEAS-08 file, the MeAJOR
development data, and the frozen model bundles, so they run only inside the full workspace. The
reference outputs are in `results/`; `e1_summary.json` records the SHA-256 of
`e1_results_long.csv` and of the per-draw file `e1_draws.parquet` (15 MB, not included because
`e1_label_budget.py` regenerates it).
