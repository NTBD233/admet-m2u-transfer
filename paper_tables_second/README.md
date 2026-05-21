# Second Paper Tables

- `table1_main_aggregate_rmse`: full 45-run aggregate comparison.
- `table2_main_by_setting_rmse`: dataset x train-ratio RMSE table.
- `table3_selector_diagnostics`: selector predictability and gate-target diagnostics.
- `table4_selector_quality_by_dataset`: selector accuracy by endpoint.
- `table5_secondary_ablation_summary`: partial secondary/negative ablations.
- `table6_failure_mode_probe`: setting-level probes separating teacher-selection
  quality from student utilization of routed teacher supervision.
- `table7_selector_route_mix`: test-split selector routing proportions by
  dataset and train ratio.
- `table8_selector_conflict_win_summary`: teacher-conflict summaries grouped by
  whether selector variants beat the setting-level top-1 teacher baseline.

Lower RMSE is better. Deltas are target minus reference, so negative is better.
