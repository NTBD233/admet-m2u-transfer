# Advisor Review Pack

This folder contains the compact materials for advisor review.

## Recommended Reading Order

1. `one_page_summary_zh.md`
2. `manuscript_submission.pdf`
3. `figures/fig1_method_diagram.pdf`
4. `figures/fig2_distillation_delta.pdf`
5. `tables/table1_regression_main_rmse.md`
6. `submission_checklist.md`

## Core Message

Descriptor-teacher distillation improves an ECFP-only AdapterFusion student for
low-resource regression ADMET prediction, but descriptor-access RF remains a
stronger practical baseline. The paper should be framed as controlled
descriptor-guided multi-to-uni transfer, not as an ADMET SOTA claim.

## Main Claim

Fixed `lambda_distill = 1.0` improves AdapterFusion in 11/15 regression
dataset-ratio settings, with mean RMSE delta `-0.0656`.

## Main Caveat

The distilled student does not replace descriptor-access RF, and classification
results are not stable enough to support the primary claim.
