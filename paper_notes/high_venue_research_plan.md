# High-Venue Research Plan

## Working Title

**Selective Multi-Teacher Descriptor Distillation for Low-Resource ADMET Prediction**

## Motivation

The current manuscript establishes a pilot result: descriptor-teacher
distillation can improve an ECFP-only AdapterFusion student in low-resource
regression ADMET, but the effect is bounded and descriptor-access RF remains
stronger in most settings.

The high-venue version should not simply add more tables. It should answer a
more general question:

> When should an ECFP-only student trust descriptor-informed teachers, and
> which molecular knowledge source should guide each task or sample?

## Core Hypotheses

1. Descriptor knowledge is task-dependent: different ADMET endpoints benefit
   from different descriptor families or teacher types.
2. Teacher reliability is sample-dependent: a descriptor-access teacher should
   not be trusted uniformly across chemical space.
3. Selective multi-teacher distillation can improve ECFP-only students more
   reliably than fixed single-teacher distillation.

## Proposed Method

The expanded method should use multiple molecular expert teachers:

- **Descriptor teacher:** RF or XGBoost trained on ECFP4+RDKit descriptors.
- **Fingerprint teacher:** RF or XGBoost trained on ECFP4 only.
- **Graph/pretrained teacher:** Chemprop or another reproducible molecular
  encoder.
- **Reliability module:** estimates per-sample teacher weights from teacher
  uncertainty, teacher-student disagreement, descriptor-space coverage, and
  validation behavior.

The student remains ECFP-only at inference time. Training uses a weighted
distillation objective:

```text
task loss
+ descriptor reconstruction loss
+ weighted multi-teacher distillation loss
```

The central methodological contribution is not the existence of multiple
teachers, but selective teacher weighting by task and sample.

## Experimental Design

Use the current paper as the base benchmark:

- 5 regression ADMET datasets already completed.
- 10%, 20%, 50% train ratios.
- Seeds 42, 123, 3407.
- Primary metric: RMSE.

For the high-venue version, expand to:

- 8-12 ADMET datasets if compute allows.
- At least one strong pretrained or graph baseline, preferably Chemprop first.
- RF and XGBoost/LightGBM baselines with ECFP4, descriptor, and ECFP4+descriptor inputs.
- Classification kept as supplementary unless the new method stabilizes it.

## Required Ablations

- Single descriptor teacher vs multi-teacher.
- Fixed teacher weights vs learned reliability weights.
- Dataset-level weighting vs sample-level weighting.
- With and without descriptor reconstruction.
- With and without graph/pretrained teacher.
- Oracle teacher selection analysis for upper-bound interpretation.

## Mechanism Analysis

The high-venue paper needs analysis beyond aggregate RMSE:

- Which teacher is selected for which ADMET endpoint?
- How do teacher weights vary by train ratio?
- Does teacher-student disagreement predict downstream improvement?
- Does descriptor prediction quality correlate with final task gain?
- Are gains larger in low-resource settings?
- Which failure cases correspond to teacher conflict or out-of-domain samples?

## Expected Figures and Tables

- Method diagram: multi-teacher selective distillation.
- Main benchmark table: RMSE across datasets and train ratios.
- Teacher-weight heatmap by dataset and train ratio.
- Teacher disagreement vs student gain scatter.
- Low-resource sensitivity plot.
- Ablation table for teacher weighting variants.
- Failure-case table for settings where selective distillation hurts.

## Risks and Fallbacks

- If multi-teacher weighting does not improve over fixed single-teacher
  distillation, reframe as a systematic analysis of when descriptor teachers
  help or hurt.
- If Chemprop is hard to reproduce locally, use RF/XGBoost/LightGBM teachers
  first and add Chemprop later as an external baseline.
- If classification remains unstable, keep the high-venue claim restricted to
  regression ADMET.
- If gains are modest, emphasize selective reliability analysis and controlled
  inference constraints rather than SOTA performance.

## Immediate Next Steps

1. Freeze and commit the current pilot manuscript.
2. Share the advisor review pack and ask whether the short-paper framing is acceptable.
3. After advisor feedback, implement the first high-venue extension:
   RF + XGBoost multi-teacher distillation with sample-level reliability
   diagnostics.
4. Add Chemprop only after the simpler multi-teacher pipeline is reproducible.
