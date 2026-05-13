# Submission Readiness Checklist

## Current Artifacts

- Submission-style LaTeX manuscript:
  `paper_notes/manuscript_submission.tex`
- Compiled PDF:
  `paper_notes/manuscript_submission.pdf`
- BibTeX references:
  `paper_notes/references.bib`
- Main figures:
  - `paper_figures/fig1_method_diagram.pdf`
  - `paper_figures/fig2_distillation_delta.pdf`
  - `paper_figures/fig3_teacher_gap_gain.pdf`
- Compact LaTeX tables:
  - `paper_tables/table1_regression_main_compact.tex`
  - `paper_tables/table3_lambda_summary.tex`
  - `paper_tables/table4_adaptive_summary_compact.tex`

## Completed Checks

- LaTeX build completed with XeLaTeX and BibTeX.
- References resolved against `references.bib`.
- Main table and ablation tables are included directly in the LaTeX manuscript.
- Main regression table now includes `DescPred`, `AdapterFusion`,
  `ECFP4+Desc RF`, and distilled AdapterFusion, so the AdapterFusion-vs-DescPred
  claim is directly checkable from the manuscript.
- Method equations are numbered and referenced through standard LaTeX equation environments.
- Figures are included as PDF assets.
- Main claim remains controlled:
  descriptor-teacher distillation improves ECFP-only AdapterFusion, but does
  not replace descriptor-access RF.

## Numeric Consistency Checks

- Fixed `lambda_distill = 1.0` improves AdapterFusion in 11/15 regression
  dataset-ratio settings.
- Lambda summary table reports mean delta vs AdapterFusion as `-0.0656`.
- Adaptive weighting improves over base AdapterFusion in 9/15 settings.
- Adaptive weighting beats fixed `lambda_distill = 1.0` in 3/15 settings.
- Adaptive weighting beats the best fixed lambda in 1/15 settings.
- Adaptive mean delta vs fixed `lambda_distill = 1.0` is `0.0817`.
- Validation-selected lambda beats fixed `lambda_distill = 1.0` in 2/15
  settings.
- Validation-selected lambda beats base AdapterFusion in 12/15 settings.

## Reproducibility Checks

- Descriptor scaler is fit only on the training split in `utils/features.py`
  and then applied to validation and test splits.
- All neural and RF runs use the same dataset-ratio-seed feature files.
- Distillation teacher predictions are generated from saved RF teacher runs.
- Available metrics files:
  - original results: 654 `metrics.json`
  - regression distillation: 45 `metrics.json`
  - lambda sweep distillation: 135 `metrics.json`
  - adaptive distillation: 45 `metrics.json`

## Remaining Risks Before Submission

- Author list and affiliation are placeholders and must be finalized.
- The paper is still in generic article format; venue-specific formatting is
  not yet applied.
- Classification is only supplementary because results are less stable.
- The method should not be described as state-of-the-art.
- The student still trails descriptor-access RF in most settings.
- Current XeLaTeX/BibTeX build has no undefined citation/reference warnings and
  no overfull/underfull warnings in the latest log.

## Recommended Next Step

Choose a target venue or workshop template, then port
`manuscript_submission.tex` into that template while preserving the current
claims, figures, tables, and limitations.
