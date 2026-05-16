# Second Paper Writing Blueprint

## Working Title

**Pretrained Teacher Selection for Reliable Multi-Teacher Distillation in Low-Resource ADMET**

Shorter alternatives:

- **Selective Multi-Teacher Distillation for Low-Resource ADMET Prediction**
- **Supervised Teacher Selection for ECFP-Only ADMET Distillation**

## One-Sentence Thesis

Low-resource ADMET students should not uniformly imitate every molecular
teacher; an ECFP-only student benefits more reliably when teacher selection is
learned first and then used to route distillation into an ECFP-only student.

## Paper Type

This should be written as a **method paper**, not as a benchmark paper and not
as a simple extension of the pilot manuscript.

The pilot manuscript provides motivation:

- descriptor knowledge is useful;
- fixed descriptor-teacher distillation helps but is not uniformly reliable;
- different dataset-ratio settings prefer different distillation strengths;
- naive validation-advantage adaptive weighting fails.

The second paper's novelty is:

> pretrained teacher selection plus frozen routing distillation under
> ECFP-only inference.

## Core Research Question

Use this exact question to anchor the introduction:

> How can an ECFP-only ADMET student learn which molecular teacher to trust
> under low-resource supervision, and why does direct joint routing fail?

## Main Claim

Target claim after experiments:

> Pretrained teacher selection with frozen routing improves ECFP-only
> AdapterFusion more consistently than fixed single-teacher distillation,
> uniform multi-teacher averaging, task-level teacher weighting, and jointly
> learned routing gates on low-resource regression ADMET tasks.

Controlled limitation:

> Descriptor-access RF/XGB teachers remain strong controls; the contribution is
> selective training-time knowledge transfer into an ECFP-only student, not a
> claim that the student always replaces descriptor-access models.

Current refinement after the auto-reweight follow-up:

> Keep `pretrained_selector_top1` as the main method candidate. Treat
> confidence-based auto reweighting as a mechanism analysis, not as the main
> upgraded method. The partial regression audit shows that auto reweighting is
> reproducible and locally useful, but not strong enough to beat plain selector
> routing or `top1_validation` on aggregate.

Current refinement after the ratio-aware lambda follow-up:

> Treat train-ratio-aware distillation strength as a secondary calibration
> component. The `high_resource_decay` schedule is more defensible than early
> decay because it keeps low/medium-resource selector routing unchanged and only
> weakens teacher forcing at `50%` train ratio. The full 45-setting expansion
> improves plain selector mean RMSE (`5.0814` vs `5.0889`) and nearly matches
> `top1_validation` (`5.0792`), but does not decisively beat it. This supports
> the discussion point that teacher selection and teacher strength are related
> but distinct problems.

## Abstract Skeleton

Low-resource ADMET prediction benefits from inexpensive molecular knowledge
such as fingerprints and physicochemical descriptors, but a deployed predictor
may need to remain lightweight and ECFP-only. Prior descriptor-teacher
distillation improves ECFP-only students in some regression ADMET settings, yet
fixed distillation weights are unstable and naive adaptive or jointly learned
gating can suppress useful teacher supervision. We propose Pretrained Teacher
Selection for Multi-Teacher Distillation, which first trains a standalone
teacher selector from cross-fit pseudo-oracle labels and then freezes the
selector to route teacher supervision into an ECFP-only AdapterFusion student.
The selector uses teacher uncertainty, teacher consensus deviation,
validation-derived teacher priors, and descriptor-space coverage features to
predict which molecular expert should supervise each sample. Across
low-resource regression ADMET datasets, training ratios, and random seeds, the
method is evaluated against base AdapterFusion, fixed single-teacher
distillation, uniform multi-teacher distillation, task-level teacher selection,
and jointly learned routing gates. The intended result is a more reliable
ECFP-only transfer method and a sharper account of why teacher selection, not
generic gating, is the central missing piece.

Do not finalize the abstract until the diagnostics and main experiments are
complete.

## Section-by-Section Draft Plan

### 1. Introduction

Write the introduction in this order:

1. **Low-resource ADMET problem.**
   ADMET labels are scarce, noisy, and endpoint-specific. This makes auxiliary
   molecular knowledge valuable.

2. **Multiple molecular knowledge sources.**
   ECFP fingerprints, RDKit descriptors, tree ensembles, and graph/pretrained
   models capture different views of molecular structure and physicochemical
   behavior.

3. **Inference constraint.**
   Descriptor-access models are strong, but some workflows need a lightweight
   ECFP-only student at inference time.

4. **Pilot evidence and gap.**
   Descriptor-teacher distillation helps, but fixed weights are inconsistent;
   the best lambda varies by endpoint and train ratio, while naive adaptive
   validation weighting underperforms fixed distillation.

5. **New problem.**
   The central issue is not only teacher reliability, but how to obtain stable
   supervision for teacher selection under low-resource ADMET.

6. **Our method.**
   Introduce pretrained teacher selection plus frozen routing distillation.

7. **Contributions.**
   Use four contribution bullets:
   - formulate low-resource ECFP-only ADMET transfer as a teacher-selection problem;
   - show that oracle best-teacher labels are predictable while joint routing fails;
   - propose cross-fit selector pretraining plus frozen top-1 routing;
   - provide filtering/routing ablations and mechanism analyses, including
     confidence calibration and validation route audits that explain why naive
     selector-confidence reweighting is insufficient.
   - Include train-ratio-aware lambda as a secondary ablation if space allows:
     it helps explain why the selected teacher can still over-constrain the
     student when labeled data become less scarce.

### 2. Problem Formulation

Define:

- molecule \(x\);
- ECFP fingerprint \(f(x)\);
- descriptor vector \(d(x)\), available during training but not student
  inference;
- target \(y\);
- teachers \(T_1,\ldots,T_K\);
- ECFP-only student \(S_\theta\).

Core constraint:

\[
\hat{y}_{student}=S_\theta(f(x)),
\]

with no true descriptors or teacher models required at inference.

Training data can include teacher predictions:

\[
\hat{y}_{i}=T_i(x).
\]

The method learns not only from teacher predictions, but from teacher
reliability:

\[
w_i(x)=g_\phi(r_i(x), c(x)),
\]

where \(r_i(x)\) are teacher-specific reliability features and \(c(x)\) is a
teacher-conflict signal.

### 3. Method

Use these method subsections:

1. **ECFP-only AdapterFusion student**
   - Reuse the pilot architecture.
   - Keep descriptor reconstruction because it is already validated.
   - State that inference remains ECFP-only.

2. **Molecular expert teachers**
   - Descriptor-access: `ECFP4_Desc_RF`, `ECFP4_Desc_XGB`.
   - Descriptor-only: `Desc_RF`, `Desc_XGB`.
   - Fingerprint-only: `ECFP4_RF`, `ECFP4_XGB`.
   - Chemprop is future/optional, not first-round core.

3. **Reliability features**
   - Validation quality: task-level prior.
   - Uncertainty: RF per-tree std, XGB bootstrap/seed ensemble std.
   - Teacher agreement: whether other teachers support the prediction.
   - Teacher-student disagreement: how far the student is from a teacher.
   - Descriptor-space coverage: kNN distance or OOD score.

4. **Reliability gate**
   Start with the rule-based version:

   \[
   w_i(x)=\mathrm{softmax}_i(
   \alpha \tilde{q}_i
   -\beta \tilde{u}_i(x)
   +\eta \tilde{a}_i(x)
   -\gamma \tilde{\delta}_i(x)
   -\rho \tilde{o}(x)
   ).
   \]

   Explain that this is deliberately transparent and can be replaced by a
   learned gate in later ablations.

5. **Conflict-aware teacher selection**
   Define teacher conflict:

   \[
   \kappa(x)=\mathrm{Var}_i(\hat{y}_{T_i}(x)).
   \]

   If conflict is low, use soft weighted distillation from all teachers.
   If conflict is high, distill only from the top-k teachers by reliability.

6. **Training objective**

   \[
   \mathcal{L}=
   \mathcal{L}_{task}
   +\lambda_{desc}\mathcal{L}_{desc}
   +\lambda_{mt}\sum_i w_i(x)c(x)
   \mathrm{Huber}(\hat{y}_{student},\hat{y}_{T_i}).
   \]

   Default:
   - \(\lambda_{desc}=0.1\);
   - \(\lambda_{mt}=1.0\);
   - Huber teacher loss;
   - MSE teacher loss as ablation.

### 4. Experiments

Main experiment scope:

- regression ADMET only;
- datasets: `caco2_wang`, `lipophilicity_astrazeneca`,
  `solubility_aqsoldb`, `vdss_lombardo`, `ppbr_az`;
- train ratios: 10, 20, 50;
- seeds: 42, 123, 3407;
- primary metric: RMSE.

Main comparisons:

- base AdapterFusion;
- fixed single-teacher distillation with `ECFP4_Desc_RF`;
- best fixed lambda analysis from the pilot;
- uniform multi-teacher distillation;
- validation-quality weighted multi-teacher distillation;
- conflict-aware reliability-gated multi-teacher distillation.

Descriptor-access controls:

- `ECFP4_RF`, `Desc_RF`, `ECFP4_Desc_RF`;
- `ECFP4_XGB`, `Desc_XGB`, `ECFP4_Desc_XGB`.

Classification:

- supplementary only;
- do not use it to support the main claim unless results become stable.

### 5. Results

Write results around these questions:

1. Does selective multi-teacher distillation beat base AdapterFusion?
2. Does it beat fixed single-teacher distillation?
3. Does it beat uniform multi-teacher averaging?
4. Does conflict-aware top-k help when teachers disagree?
5. Which teachers are trusted for which endpoints and train ratios?

Do not overclaim if descriptor-access RF/XGB remains stronger.

### 6. Mechanism Analysis

Required analysis figures:

- teacher reliability heatmap by dataset and ratio;
- teacher disagreement vs gain scatter;
- OOD distance vs teacher error and gate weight;
- teacher uncertainty vs teacher error;
- failure-case table.

The goal is to show that the gate has interpretable behavior, not just a small
average RMSE improvement.

### 7. Discussion

Discussion should include:

- why fixed distillation is insufficient;
- why naive adaptive validation weighting failed;
- why sample-level reliability is a stronger formulation;
- when descriptor-access teachers remain preferable;
- why ECFP-only inference is still useful;
- why Chemprop/pretrained teachers are future extensions rather than required
  for the first version.

## Expected Tables And Figures

### Table 1: Main Regression RMSE

Columns:

- dataset;
- train ratio;
- AdapterFusion;
- fixed `ECFP4_Desc_RF` distillation;
- uniform multi-teacher;
- validation-weighted multi-teacher;
- reliability-gated conflict-aware multi-teacher;
- `ECFP4+Desc RF`;
- `ECFP4+Desc XGB`;
- delta vs AdapterFusion;
- delta vs fixed single-teacher.

### Table 2: Teacher Diagnostic Summary

Columns:

- dataset;
- train ratio;
- best validation teacher;
- best test teacher;
- teacher disagreement mean;
- oracle sample teacher RMSE;
- reliability headroom.

### Table 3: Ablation

Rows:

- no descriptor reconstruction;
- no uncertainty;
- no agreement;
- no OOD coverage;
- no conflict top-k;
- MSE instead of Huber;
- learned gate if implemented.

### Figure 1: Method Diagram

Show:

- ECFP-only student;
- multiple teachers;
- reliability feature extractor;
- conflict detector;
- weighted/top-k distillation;
- ECFP-only inference.

### Figure 2: Teacher Weight Heatmap

Rows: dataset-ratio.

Columns: teacher types.

Values: mean gate weight.

### Figure 3: Conflict vs Gain

x-axis: teacher disagreement.

y-axis: gain over fixed single-teacher distillation.

Expected story: conflict-aware selection helps most when teacher disagreement
is high.

### Figure 4: OOD Reliability

x-axis: descriptor-space distance.

y-axis: teacher error or gate weight.

Expected story: uncertain/OOD samples receive reduced unreliable teacher
pressure.

## Success Criteria

Minimum for a credible method paper:

- beats base AdapterFusion in at least 12/15 regression settings;
- beats fixed single-teacher `lambda_distill=1.0` in at least 8/15 settings;
- improves mean RMSE over fixed single-teacher by at least 0.02;
- does not create a large failure on `vdss_lombardo`;
- reliability analysis shows non-random teacher selection behavior.

If these criteria fail but oracle teacher selection shows headroom, pivot to a
learned reliability gate.

If oracle teacher selection does not show headroom, pivot to single-teacher
uncertainty/OOD-weighted distillation.

## Writing Order

1. Write Problem Formulation.
2. Write Method.
3. Write Experiment Matrix.
4. Run diagnostics and fill Results after data exists.
5. Write Introduction and Abstract last.
6. Write Discussion after failure cases are known.

## Phrases To Use

- "ECFP-only inference"
- "training-time descriptor-informed supervision"
- "teacher reliability is task- and sample-dependent"
- "teacher conflict can make naive averaging harmful"
- "descriptor-access models are controls, not inference-equivalent baselines"
- "selective distillation rather than uniform distillation"

## Phrases To Avoid

- "state-of-the-art ADMET predictor"
- "we solve ADMET prediction"
- "multi-teacher is always better"
- "adaptive weighting works" unless the new gate actually supports it
- "the student replaces descriptor-access RF/XGB"
