# Experiment Log

## Completed

- Organized the original notebook into script-based project structure.
- Prepared `bbb_martins` and `caco2_wang`.
- Ran:
  - `ECFP4_MLP`
  - `ECFP4_MLP_DescPred`
  - `ECFP4_MLP_DescConcat`
  - `ECFP4_MLP_DescAdapterFusion`
- Ran lambda ablation for `ECFP4_MLP_DescPred`.

## Current Finding

`DescConcat` is the strongest descriptor-access control. `DescPred` only
partially transfers descriptor knowledge. `AdapterFusion` improves the caco2
regression result but not the BBB classification result.

## Distillation Update

- Completed regression teacher distillation for five regression datasets,
  three train ratios, and three seeds.
- Completed lambda sweep for `lambda_distill = 0.01, 0.1, 0.3, 1.0`.
- `lambda_distill = 1.0` is the best global setting so far: 11/15 regression
  settings improve over base AdapterFusion.
- Prediction-space diagnostics show that `lambda_distill = 1.0` also moves the
  student closer to the `ECFP4_Desc_RF` teacher on average.
- The result motivates adaptive teacher weighting rather than further fixed
  lambda sweeps.
- First adaptive teacher weighting by validation teacher advantage completed
  45/45 seed runs. It is weaker than fixed `lambda_distill = 1.0`, so it should
  be treated as a negative ablation rather than the next main method.
- Paper-ready result tables were generated under `paper_tables/`, including
  regression main results, lambda ablation, adaptive negative ablation, and a
  classification reference table.
- Validation-selected lambda analysis was added as
  `paper_tables/table6_validation_selected_lambda.md`; it beats fixed
  `lambda_distill = 1.0` in only 2/15 settings.
- Initial manuscript draft sections were generated for Method, Experiments,
  Results, and Discussion.
- A consolidated full manuscript draft was generated at
  `paper_notes/manuscript_draft.md`.

## Next Runs

1. Use `paper_tables/table1_regression_main_rmse.md` as the main Results table.
2. Use `paper_tables/table2_lambda_ablation_rmse.md` and
   `paper_tables/table3_lambda_summary.md` for ablation.
3. Use `paper_tables/table4_adaptive_summary.md` as a negative ablation.
4. Replace citation placeholders in `paper_notes/manuscript_draft.md` with
   complete BibTeX-ready references.
