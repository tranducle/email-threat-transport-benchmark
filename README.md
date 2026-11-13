# Email Threat Transport Benchmark

Code for the manuscript:

> **A Good Ranking Is Not a Transferable Cutoff: False-Positive Control of Spam and Phishing Filters Under Dataset Shift**

The study fixes a spam or phishing decision cutoff on two development sources (TREC5 and TREC6 from the MeAJOR collection) at a target false-positive rate, freezes it, and measures what happens to that cutoff on two external corpora (SpaPhish v5 and CEAS-08), under prior shift, and forward in time. It reports three properties of each classifier separately: how well it ranks messages, how good its probabilities are, and whether its cutoff keeps the false-positive budget on new mail.

The code covers:

- six classifiers in three model families: balanced and unweighted TF-IDF logistic regression, a three-seed multilingual DistilBERT ensemble, and a B-spline Kolmogorov-Arnold network, an MLP and histogram gradient-boosted trees on truncated SVD projections;
- threshold policies ordered by the target information they use, from the frozen development cutoff through prior-shift recalibration (EM, BBSE, oracle prior) to an exact Clopper-Pearson certified cutoff;
- per-seed score-ceiling and reciprocal cutoff feasibility checks;
- exchangeable label-budget simulations and forward chronological certification on SpaPhish v5;
- a sender-domain cluster bootstrap on CEAS-08;
- cross-fitted residual probes of static URL, domain and message features over the DistilBERT scores.

## Repository layout

```text
README.md                 this overview
DATA.md                   data sources, checksums and expected message counts
REPRODUCIBILITY.md        environment, checks and run order
requirements.txt          installs the pinned versions in threshold_transport/requirements.txt
environment.lock.json     full environment of the original runs (packages, GPU, CUDA, platform)
RELEASE_SHA256.txt        SHA-256 of every committed file
LICENSE                   MIT License
threshold_transport/
    preprocessing/        MeAJOR normalization and the TREC5+TREC6 development set
    training/             anchor models, cross-fitted DistilBERT scores, training-time measurement
    external_corpora/     SpaPhish v5 and CEAS-08 adapters and scoring, SpaPhish URL feature rebuild
    experiments/          prior shift and seed ceilings; multi-family training and residual probes;
                          chronological and cluster certification
    configs/              anchor-model and residual-probe settings
    tests/                regression tests for the SpaPhish URL parser
    README.md             data layout, run order, seeds and the script behind each table and figure
```

## Data

No email text is stored in this repository. All three corpora are public under CC BY 4.0; [DATA.md](DATA.md) lists the records, file names, checksums and the message counts the scripts check before running.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s threshold_transport/tests
```

The unit tests need no data. The full pipeline needs the three corpora and a CUDA GPU for the DistilBERT steps; [REPRODUCIBILITY.md](REPRODUCIBILITY.md) gives the run order and run times, and [threshold_transport/README.md](threshold_transport/README.md) maps every table and figure of the manuscript to the script and output file that produce it.

## License

MIT; see [LICENSE](LICENSE). The corpora keep their own licenses (see [DATA.md](DATA.md)).
