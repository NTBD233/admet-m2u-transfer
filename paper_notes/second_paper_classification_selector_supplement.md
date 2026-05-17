# Classification Selector Supplementary Diagnostics

## Scope

This note summarizes RF selector diagnostics for five classification endpoints:
`bbb_martins`, `hia_hou`, `pgp_broccatelli`, `bioavailability_ma`, and `herg`.
Each endpoint uses train ratios 10/20/50 and seeds 42/123/3407, for 45 settings.

The selector is the same `rf_crossfit_train_pseudo_oracle` model used in the
regression main experiments, with RF teachers `ECFP4_RF`, `Desc_RF`, and
`ECFP4_Desc_RF`.

## Overall Result

| valid_accuracy | test_accuracy | valid_majority_accuracy | test_majority_accuracy | valid_macro_f1 | test_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| 0.1906 | 0.2162 | 0.2620 | 0.2958 | 0.1511 | 0.1766 |

## By Dataset

| dataset | valid_accuracy | test_accuracy | valid_majority_accuracy | test_majority_accuracy | valid_macro_f1 | test_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| bbb_martins | 0.1651 | 0.1554 | 0.3337 | 0.3352 | 0.1435 | 0.1387 |
| bioavailability_ma | 0.2429 | 0.2708 | 0.3715 | 0.3854 | 0.1783 | 0.2077 |
| herg | 0.2402 | 0.2567 | 0.2180 | 0.2163 | 0.1755 | 0.1959 |
| hia_hou | 0.1304 | 0.2127 | 0.2017 | 0.2650 | 0.1082 | 0.1744 |
| pgp_broccatelli | 0.1741 | 0.1850 | 0.1850 | 0.2771 | 0.1498 | 0.1662 |

## By Train Ratio

| train_ratio_tag | valid_accuracy | test_accuracy | valid_majority_accuracy | test_majority_accuracy | valid_macro_f1 | test_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.2032 | 0.2123 | 0.2531 | 0.2966 | 0.1423 | 0.1608 |
| 20 | 0.1984 | 0.2383 | 0.2498 | 0.2954 | 0.1667 | 0.1834 |
| 50 | 0.1701 | 0.1978 | 0.2830 | 0.2954 | 0.1442 | 0.1856 |

## Interpretation

Classification selector transfer is not ready for the main claim. The RF selector
overfits the cross-fit train labels but generalizes poorly: mean test selector
accuracy is below the train-majority baseline. This supports keeping classification
outside the main regression claim and treating it as a future calibration problem.

The likely issue is not only teacher availability. Classification pseudo-oracle
labels are based on probability-space absolute error, while downstream ADMET
classification is evaluated by ranking metrics such as ROC-AUC. That mismatch
can make sample-level teacher labels noisy and poorly aligned with student-level
classification gains.

## Paper Action

- Do not promote classification routing to a main result.
- Mention classification as supplementary diagnostics or limitation only.
- If classification is revisited, redesign pseudo-oracle labels around
  classification-calibrated criteria instead of copying the regression protocol.
