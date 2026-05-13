# Results Summary

## Main Regression Result

The regression paper table is in:

- `paper_tables/table1_regression_main_rmse.csv`
- `paper_tables/table1_regression_main_rmse.md`

Primary metric: test RMSE, lower is better.

Main findings:

- Descriptor-access RF remains the strongest practical baseline in most
  regression settings.
- Base AdapterFusion is consistently better than simple DescPred on regression
  tasks, supporting structured descriptor transfer.
- Fixed teacher distillation with `lambda_distill = 1.0` improves base
  AdapterFusion in 11/15 regression settings.
- The best fixed lambda per setting improves over base AdapterFusion in most
  settings, but the preferred lambda varies by dataset and train ratio.
- Adaptive validation-advantage weighting is not competitive with fixed
  distillation and should be reported as a negative ablation.

## Lambda Sweep

The lambda sweep tables are in:

- `paper_tables/table2_lambda_ablation_rmse.csv`
- `paper_tables/table3_lambda_summary.csv`

Aggregate result:

| lambda_distill | beats AdapterFusion | mean delta vs AdapterFusion |
| --- | --- | ---: |
| 0.01 | 8/15 | 0.0075 |
| 0.1 | 8/15 | -0.0221 |
| 0.3 | 10/15 | -0.0046 |
| 1.0 | 11/15 | -0.0656 |

Interpretation:

- `lambda_distill = 1.0` is the best global fixed setting.
- Smaller lambdas can be better on specific settings, especially `caco2_wang`
  and `vdss_lombardo`.
- This supports reporting both a fixed global method and a lambda sensitivity
  ablation.

## Adaptive Distillation

The adaptive distillation tables are in:

- `paper_tables/table4_adaptive_summary.csv`
- `paper_tables/table5_adaptive_vs_fixed.csv`

Aggregate result:

| comparison | result |
| --- | --- |
| adaptive beats base AdapterFusion | 9/15 |
| adaptive beats fixed `lambda_distill = 1.0` | 3/15 |
| adaptive beats best fixed lambda | 1/15 |
| mean delta vs fixed `lambda_distill = 1.0` | 0.0817 |

Interpretation:

- Validation teacher-advantage weighting is a negative ablation.
- It suppresses teacher guidance too aggressively on datasets where fixed
  strong distillation helps.
- The paper should not claim adaptive weighting as the method.

## Validation-Selected Lambda

The validation-selected lambda table is in:

- `paper_tables/table6_validation_selected_lambda.csv`
- `paper_tables/table6_validation_selected_lambda.md`

This table selects the fixed lambda with the lowest mean validation RMSE for
each dataset-ratio setting, then reports the corresponding test RMSE. No new
models are trained.

Aggregate result:

- Validation-selected lambda beats fixed `lambda_distill = 1.0` in 2/15
  settings.
- Validation-selected lambda beats base AdapterFusion in 12/15 settings.

Interpretation:

- Validation selection is useful as an analysis, but does not outperform the
  fixed global `lambda_distill = 1.0` setting overall.
- This reinforces the current decision to use fixed `lambda_distill = 1.0` as
  the main method and report lambda sensitivity as ablation.

## Classification Reference

The classification table is in:

- `paper_tables/tableS1_classification_main_roc_auc.csv`

Primary metric: test ROC-AUC, higher is better.

Interpretation:

- Classification results are less stable than regression.
- Descriptor-access controls are often strong.
- Current teacher distillation work should stay focused on regression.

## Current Paper Claim

Supported claim:

> Descriptor-teacher distillation improves ECFP-only AdapterFusion for
> low-resource regression ADMET, but does not replace descriptor-access RF.

Unsupported claim:

> The method is a new overall ADMET state-of-the-art predictor.

Recommended framing:

> A controlled study and lightweight method for descriptor-guided multi-to-uni
> transfer in low-resource regression ADMET.

## Draft Files

Initial manuscript draft sections are available in:

- `paper_notes/method_draft.md`
- `paper_notes/experiments_draft.md`
- `paper_notes/results_draft.md`
- `paper_notes/discussion_draft.md`

The consolidated manuscript draft is available in:

- `paper_notes/manuscript_draft.md`

Remaining writing task:

- Replace citation placeholders with complete references.
- Decide target venue formatting before converting to LaTeX or Word.
