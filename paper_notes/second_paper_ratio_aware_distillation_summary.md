# Ratio-Aware Selector Distillation Summary

## Motivation

The selector route audit suggested that the remaining failure mode is not only
teacher choice. In several higher-ratio settings, the routed teacher target can
still over-constrain the student even when the selected teacher is reasonable.

This follow-up tested whether distillation strength should depend on the train
ratio for `pretrained_selector_top1`.

## Implementation

The training script now supports a ratio schedule for `lambda_distill`:

- `none`: no scaling.
- `low_resource_decay`: `10 -> 1.0`, `20 -> 0.7`, `50 -> 0.3`.
- `high_resource_decay`: `10 -> 1.0`, `20 -> 1.0`, `50 -> 0.3`.
- `sqrt_10_over_ratio`: continuous heuristic, `min(1.0, sqrt(10 / ratio))`.

The effective value is stored as:

- `lambda_distill_ratio_schedule`
- `lambda_distill_ratio_factor`
- `lambda_distill_scheduled`

## Experimental Scope

Partial regression subset:

- datasets: `caco2_wang`, `ppbr_az`
- train ratios: `10 / 20 / 50`
- seeds: `42 / 123 / 3407`
- total settings: `18`
- student: `ECFP4_MLP_DescAdapterFusion`
- selector: `rf_crossfit_train_pseudo_oracle`
- routing: `pretrained_selector_top1`
- base lambda: `lambda_distill=1.0`

Result roots:

- `results_pretrained_selector_top1_ratio_lambda_partial_regression`
- `results_pretrained_selector_top1_high_resource_lambda_partial_regression`

## Aggregate Results

Mean test RMSE, lower is better:

| method | mean test RMSE |
| --- | ---: |
| base AdapterFusion | 8.8510 |
| fixed `ECFP4_Desc_RF` distillation | 8.7949 |
| `top1_validation` | 8.5725 |
| plain `pretrained_selector_top1` | 8.6091 |
| `low_resource_decay` | 8.6666 |
| `high_resource_decay` | 8.5852 |

`high_resource_decay` wins:

- vs base AdapterFusion: `15/18`, mean delta `-0.2658`
- vs fixed `ECFP4_Desc_RF`: `13/18`, mean delta `-0.2097`
- vs `top1_validation`: `11/18`, mean delta `+0.0126`
- vs plain selector: `5/18`, mean delta `-0.0239`
- vs `low_resource_decay`: `4/18`, mean delta `-0.0814`

`low_resource_decay` wins:

- vs base AdapterFusion: `14/18`, mean delta `-0.1844`
- vs fixed `ECFP4_Desc_RF`: `10/18`, mean delta `-0.1283`
- vs `top1_validation`: `8/18`, mean delta `+0.0940`
- vs plain selector: `7/18`, mean delta `+0.0575`

## Ratio-Wise Behavior

Mean test RMSE by train ratio:

| ratio | base | fixed | top1 validation | plain selector | low decay | high decay |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 9.9492 | 9.9172 | 9.7479 | 9.8161 | 9.8161 | 9.8161 |
| 20 | 8.8121 | 8.8479 | 8.3772 | 8.4524 | 8.6966 | 8.4524 |
| 50 | 7.7916 | 7.6195 | 7.5925 | 7.5589 | 7.4871 | 7.4871 |

Interpretation:

- `10%`: both schedules equal plain selector because factor is `1.0`.
- `20%`: reducing lambda to `0.7` is harmful on aggregate.
- `50%`: reducing lambda to `0.3` improves over plain selector and
  `top1_validation` on aggregate.

## Dataset-Wise Behavior

Mean test RMSE by dataset:

| dataset | base | fixed | top1 validation | plain selector | low decay | high decay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `caco2_wang` | 1.1339 | 1.0981 | 1.1044 | 1.1048 | 1.0969 | 1.0912 |
| `ppbr_az` | 16.5680 | 16.4917 | 16.0407 | 16.1134 | 16.2363 | 16.0792 |

`high_resource_decay` improves caco2 more clearly. On ppbr it recovers most of
the damage caused by `low_resource_decay`, but still does not beat
`top1_validation` on mean RMSE.

## Current Conclusion

The result supports a restrained claim:

> selector routing should be paired with train-ratio-aware distillation
> strength, but only after the resource regime is high enough for teacher
> forcing to become a liability.

This should not replace the paper's main innovation. The main method remains
pretrained teacher selection with frozen routing. Ratio-aware lambda is a
secondary calibration component or ablation:

- use `high_resource_decay` as the preferred schedule if included;
- demote `low_resource_decay` to a negative ablation showing that early
  distillation weakening can hurt medium-resource settings;
- do not promote hand-designed lambda schedules above the selector supervision
  story unless the full 45-setting matrix confirms the effect.

## Recommended Next Step

Run `high_resource_decay` on the full five-dataset regression matrix only if
the paper needs an additional method column. Otherwise, keep it as a focused
analysis that explains why selector routing still needs student-side
distillation calibration.
