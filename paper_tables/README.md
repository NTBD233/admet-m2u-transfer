# Paper Tables

- `table1_regression_main_rmse`: main regression comparison, RMSE lower is better.
- `table2_lambda_ablation_rmse`: fixed distillation lambda ablation.
- `table3_lambda_summary`: aggregate lambda sweep summary.
- `table4_adaptive_summary`: adaptive distillation aggregate negative ablation.
- `table5_adaptive_vs_fixed`: adaptive setting-level comparison.
- `table6_validation_selected_lambda`: lambda selected by validation RMSE, no retraining.
- `tableS1_classification_main_roc_auc`: classification reference table, ROC-AUC higher is better.

Primary method currently recommended for regression: fixed `lambda_distill = 1.0`.
Adaptive validation-advantage weighting should be reported as a negative ablation, not the main method.