# Protocol Notes and Anti-Leakage Guarantees

## 1. Reciprocal Source-Held-Out Design and Conservative Threshold Locking

- **Reciprocal Folds**: Fold 1 trains on `trec5` (`N = 46,640`) and evaluates out-of-source on `trec6` (`N = 14,685`). Fold 2 trains on `trec6` and evaluates out-of-source on `trec5`.
- **Conservative Common Threshold (`tau*`)**: On each validation fold, `threshold_at_max_fpr` selects the lowest score threshold achieving empirical `FPR <= 0.01`. The common operational threshold is locked as `tau* = max(tau_trec5, tau_trec6)` so that the declared operating point is conservative across both development sources without post-hoc target tuning.

## 2. Why Ranking Transportability Does Not Imply Common-Threshold Compatibility

For `A2_MDISTILBERT`, fold-specific `1%` FPR thresholds differ sharply across sources (`tau_trec5 = 0.9634` when trained on `trec6` vs `tau_trec6 = 0.9996` when trained on `trec5`). When the conservative shared threshold `tau* = 0.9996` is enforced without target-corpus recalibration, `A2_MDISTILBERT` achieves `Recall = 0.8553` at `FPR = 0.0099` on `trec6`, but `Recall = 0.0000` (`TP = 0 / 19,235`) on `trec5` despite achieving `AUPRC = 0.9714` on `trec5`. This demonstrates that cross-corpus ranking transportability and fixed-threshold compatibility must be evaluated as distinct deployment properties.
