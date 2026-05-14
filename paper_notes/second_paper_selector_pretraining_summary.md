# Second Paper Selector Pretraining Summary

## Status

The formal RF selector pretraining stage has been expanded from smoke to the
full regression matrix.

Completed setting count:

- datasets: `caco2_wang`, `lipophilicity_astrazeneca`,
  `solubility_aqsoldb`, `vdss_lombardo`, `ppbr_az`
- train ratios: `10`, `20`, `50`
- seeds: `42`, `123`, `3407`
- total selector settings: `45/45`

Current selector configuration:

- teachers: `ECFP4_RF`, `Desc_RF`, `ECFP4_Desc_RF`
- selector model: `RF`
- label source: `crossfit_train_pseudo_oracle`

## Aggregate Selector Quality

Across all 45 regression settings:

| metric | value |
| --- | ---: |
| mean validation accuracy | `0.5178` |
| mean test accuracy | `0.5425` |
| mean validation macro-F1 | `0.4269` |
| mean test macro-F1 | `0.4396` |

Interpretation:

1. the selector signal remains stable after moving from a single smoke setting
   to the full regression matrix
2. selector prediction is noisy but consistently above a weak-chance regime for
   three-way teacher selection
3. the selector is strong enough to justify full student-level frozen-routing
   experiments

## Dataset-Level Mean Accuracy

| dataset | validation accuracy | test accuracy |
| --- | ---: | ---: |
| `caco2_wang` | `0.4924` | `0.5354` |
| `lipophilicity_astrazeneca` | `0.5035` | `0.5083` |
| `ppbr_az` | `0.5566` | `0.5450` |
| `solubility_aqsoldb` | `0.5380` | `0.5692` |
| `vdss_lombardo` | `0.4985` | `0.5546` |

Interpretation:

1. selector predictability varies by endpoint, which is consistent with the
   broader teacher-reliability story
2. `solubility_aqsoldb` and `ppbr_az` appear easier for teacher selection than
   `caco2_wang` and `lipophilicity_astrazeneca`
3. this supports reporting selector-quality diagnostics as part of the method
   analysis, not only student RMSE tables

## Current Decision

The selector-pretraining stage is now strong enough to support the next full
method step:

> expand `pretrained_selector_top1` to the full regression matrix before
> investing more effort in top-2 routing or selector-based filtering.

Current ordering of method promise:

1. `pretrained_selector_top1`
2. `selector_filtered_validation_weighted`
3. `pretrained_selector_top2`

This ordering is based on the current smoke results and should be re-checked on
the full student regression matrix.
