# Results Draft

## Main Regression Results

The main regression results are reported in
`paper_tables/table1_regression_main_rmse.md`. RMSE is the primary metric, and
lower values indicate better performance.

Across the five regression datasets and three low-resource ratios, descriptor
knowledge is consistently useful. Descriptor-access RF baselines remain very
strong, showing that RDKit descriptors contain predictive information that is
not fully recovered by ECFP-only neural models.

AdapterFusion is a stronger ECFP-only transfer mechanism than simple descriptor
prediction. Before teacher distillation, AdapterFusion improves over DescPred
in all 15 regression dataset-ratio settings. This supports the central idea
that descriptor knowledge is better transferred through a structured
pseudo-descriptor representation and fusion gate than through a plain auxiliary
descriptor head alone.

## Descriptor-Teacher Distillation

Adding teacher prediction supervision further improves AdapterFusion. With
`lambda_distill = 1.0`, distilled AdapterFusion improves over base AdapterFusion
in 11/15 regression settings. The average RMSE delta versus base AdapterFusion
is -0.0656 across the regression settings.

The largest practical gains appear on settings where the descriptor-access
teacher is substantially stronger than the ECFP-only neural student. For
example, `ppbr_az` benefits from fixed strong distillation at 10%, 20%, and 50%
training ratios. However, the distilled student remains below the
descriptor-access RF teacher in most settings, so the method should be viewed
as partial descriptor knowledge transfer rather than a replacement for
descriptor-access models.

## Lambda Sensitivity

The lambda sweep is reported in
`paper_tables/table2_lambda_ablation_rmse.md` and summarized in
`paper_tables/table3_lambda_summary.md`.

Aggregate results are:

| lambda_distill | beats AdapterFusion | mean delta vs AdapterFusion |
| --- | --- | ---: |
| 0.01 | 8/15 | 0.0075 |
| 0.1 | 8/15 | -0.0221 |
| 0.3 | 10/15 | -0.0046 |
| 1.0 | 11/15 | -0.0656 |

The global best fixed setting is `lambda_distill = 1.0`. Smaller lambdas are
occasionally better for individual dataset-ratio settings, especially
`caco2_wang` and `vdss_lombardo`, but they do not improve the aggregate result.

## Validation-Selected Lambda

The validation-selected lambda analysis is reported in
`paper_tables/table6_validation_selected_lambda.md`. For each dataset-ratio
setting, the lambda with the lowest mean validation RMSE is selected and its
test RMSE is reported.

Validation-selected lambda beats fixed `lambda_distill = 1.0` in only 2/15
settings. It beats base AdapterFusion in 12/15 settings, but this is not enough
to justify replacing the fixed global lambda. This result suggests that
validation RMSE selection is useful diagnostically but not a clearly stronger
method under the current experimental setup.

## Adaptive Distillation Negative Ablation

The adaptive teacher-weighting ablation is reported in
`paper_tables/table4_adaptive_summary.md` and
`paper_tables/table5_adaptive_vs_fixed.md`.

The adaptive rule improves over base AdapterFusion in 9/15 settings, but it
beats fixed `lambda_distill = 1.0` in only 3/15 settings and beats the best
fixed lambda in only 1/15 settings. Its mean RMSE delta versus fixed
`lambda_distill = 1.0` is 0.0817, meaning it is worse on average.

This negative result indicates that raw validation teacher advantage is not a
sufficient reliability signal. The rule often down-weights teacher supervision
too aggressively on datasets where strong fixed distillation is beneficial.

## Classification Reference

Classification results are provided as supplementary reference in
`paper_tables/tableS1_classification_main_roc_auc.md`. The classification
results are less stable than regression results, so the main method claim is
restricted to regression ADMET prediction.
