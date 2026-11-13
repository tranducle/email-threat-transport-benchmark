# Data

No email text, score file or model checkpoint is redistributed here. Download the public files below and place them under the data root described in [threshold_transport/README.md](threshold_transport/README.md#data-root-and-layout): the directory named by the environment variable `THRESHOLD_TRANSPORT_DATA`, or `data/` at the repository root (ignored by Git).

## Sources

| Corpus | Record | File | Size and checksum | License |
| :- | :- | :- | :- | :- |
| MeAJOR (TREC5 and TREC6; TREC7 only for an optional diagnostic) | Zenodo, "MeAJOR: Merged email Assets from Joint Open-source Repositories", version 2.0, DOI [10.5281/zenodo.18471483](https://doi.org/10.5281/zenodo.18471483) | `meajor_cleaned_preprocessed.parquet.gzip` | 86,278,230 bytes; MD5 `78e397ad8447bcdba5a98097921ba8bd`; SHA-256 of the copy used in the study `5a9de6c207634f068928ed0cc68d78ee2fd594ed9a587daf4048974c95715c1a` | CC BY 4.0 |
| SpaPhish v5 | Mendeley Data, "SpaPhish: A Spanish Dataset for Phishing and Psychological Pattern Detection", version 5, DOI [10.17632/hz2d6gz7pc.5](https://doi.org/10.17632/hz2d6gz7pc.5) | `Spaphish dataset - DiB.csv` | 5,098,917 bytes; SHA-256 `656b2245d58da72d640680e5c2a168673a130b38607f2a427c773bbb167e995e` | CC BY 4.0 |
| CEAS-08 | Zenodo, "Phishing Email Curated Datasets", DOI [10.5281/zenodo.8339691](https://doi.org/10.5281/zenodo.8339691) | `CEAS_08.csv` | 67,902,056 bytes; MD5 `1f0d59191eec892a709995d48cd8decb` | CC BY 4.0 |

TREC5 and TREC6 are the TREC 2005 and 2006 Spam Track corpora as distributed inside MeAJOR ([2005 overview](https://trec.nist.gov/pubs/trec14/papers/SPAM.OVERVIEW.pdf), [2006 overview](https://trec.nist.gov/pubs/trec15/papers/SPAM06.OVERVIEW.pdf)). SpaPhish v5 is the standalone Spanish corpus, not the `spaphish` source inside MeAJOR. `Nazario_5.csv` from the CEAS-08 record is read only by a standalone run of `preprocess_corpora.py` and is not used by the experiments.

## Expected counts

After normalization and contamination screening the experiments use:

| Corpus | Messages | Positive | Negative | Role |
| :- | -: | -: | -: | :- |
| TREC5 | 46,640 | 19,235 | 27,405 | development (reciprocal fit and test) |
| TREC6 | 14,685 | 3,503 | 11,182 | development (reciprocal fit and test) |
| SpaPhish v5 | 1,395 | 731 | 664 | external and chronological test |
| CEAS-08 | 38,920 | 21,674 | 17,246 | external and sender-domain cluster test |

SpaPhish v5 is split by date into `pre_2023` (438 messages: 400 negative, 38 positive) for calibration and `2023_to_2025` (933 messages: 264 negative, 669 positive) for testing; 24 undated messages are set aside. CEAS-08 loses 191 internal exact duplicates and 43 messages at or above 0.95 character 5-gram similarity to TREC5 or TREC6; the retained messages form 16,729 sender-domain clusters (16,728 parsed domains plus one pooled cluster of the 925 messages without a domain).

The scripts check these counts and the checksums above and stop when they differ.
