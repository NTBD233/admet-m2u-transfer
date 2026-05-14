# Stage 2 Full Regression Summary

## Scope

- Student: `ECFP4_MLP_DescAdapterFusion`
- Teachers: `ECFP4_RF`, `Desc_RF`, `ECFP4_Desc_RF`
- Distillation strength: `lambda_distill = 1.0`
- Datasets: `caco2_wang`, `lipophilicity_astrazeneca`, `solubility_aqsoldb`, `vdss_lombardo`, `ppbr_az`
- Train ratios: `10`, `20`, `50`
- Seeds: `42`, `123`, `3407`

This stage evaluates setting-level multi-teacher selection baselines before moving to a sample-level reliability gate.

## Compared Baselines

1. `uniform`
2. `validation_weighted`
3. `top1_validation`

Reference lines:

- base AdapterFusion
- fixed single-teacher distillation with `ECFP4_Desc_RF` and `lambda_distill = 1.0`

## Main Results

### Overall mean test RMSE

| Method | Mean test RMSE |
| --- | ---: |
| Base AdapterFusion | `5.1841` |
| Fixed single-teacher (`ECFP4_Desc_RF`) | `5.1186` |
| Uniform multi-teacher | `5.1403` |
| Validation-weighted multi-teacher | `5.0969` |
| Top-1 validation multi-teacher | `5.0792` |

### Number of settings beating base AdapterFusion

Out of `15` dataset x ratio settings:

| Method | Better than base |
| --- | ---: |
| Uniform multi-teacher | `9 / 15` |
| Validation-weighted multi-teacher | `11 / 15` |
| Top-1 validation multi-teacher | `12 / 15` |

### Number of settings beating fixed single-teacher distillation

Out of `15` dataset x ratio settings:

| Method | Better than fixed single-teacher |
| --- | ---: |
| Uniform multi-teacher | `8 / 15` |
| Validation-weighted multi-teacher | `8 / 15` |
| Top-1 validation multi-teacher | `7 / 15` |

### Mean test RMSE delta

Negative is better.

| Method | vs base | vs fixed single-teacher |
| --- | ---: | ---: |
| Uniform multi-teacher | `-0.0438` | `+0.0217` |
| Validation-weighted multi-teacher | `-0.0872` | `-0.0217` |
| Top-1 validation multi-teacher | `-0.1049` | `-0.0393` |

## Setting-Level Winners

Best multi-teacher strategy by mean test RMSE for each dataset x ratio setting:

| Dataset | 10 | 20 | 50 |
| --- | --- | --- | --- |
| `caco2_wang` | `top1_validation` | `top1_validation` | `uniform` |
| `lipophilicity_astrazeneca` | `uniform` | `validation_weighted` | `uniform` |
| `solubility_aqsoldb` | `top1_validation` | `uniform` | `uniform` |
| `vdss_lombardo` | `validation_weighted` | `validation_weighted` | `top1_validation` |
| `ppbr_az` | `uniform` | `top1_validation` | `validation_weighted` |

Win counts across the `15` settings:

- `uniform`: `6`
- `top1_validation`: `5`
- `validation_weighted`: `4`

This means no single setting-level strategy dominates every regime. However, `top1_validation` gives the best overall average and the strongest improvement over the base student.

## Interpretation

The full Stage 2 result supports three claims:

1. **Multi-teacher selection helps, but naive averaging is not enough.**  
   Uniform averaging improves over the base student on average, but it is weaker than the better selection strategies.

2. **Teacher selection should depend on the endpoint and train ratio.**  
   The best setting-level strategy changes across datasets and data regimes.

3. **The strongest setting-level baseline is still coarse.**  
   `top1_validation` only chooses one teacher for the entire setting. It cannot adapt at the sample level, and it does not model within-setting teacher conflict.

This is exactly the gap needed for Stage 3:

> move from setting-level teacher selection to sample-level reliability gating.

## Consequence for Stage 3

The main Stage 3 baseline to beat should be:

- `top1_validation` as the strongest overall setting-level multi-teacher baseline

Secondary controls should still include:

- fixed single-teacher distillation with `ECFP4_Desc_RF`
- validation-weighted multi-teacher distillation

## Implementation Direction

Stage 3 should start with the smallest upgrade over Stage 2:

1. compute per-sample reliability scores for each teacher
2. convert them into normalized teacher weights
3. replace global setting-level weights with sample-level weights inside the distillation loss

The first version does not need a learned gate. A rule-based sample gate is enough to test whether sample-level reliability is real.
