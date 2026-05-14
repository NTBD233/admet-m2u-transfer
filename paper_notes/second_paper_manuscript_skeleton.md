# Pretrained Teacher Selection for Reliable Multi-Teacher Distillation in Low-Resource ADMET

## Abstract

Placeholder. Write this after Stage 1 diagnostics and Stage 3 main results are
available.

Must include:

- low-resource regression ADMET motivation;
- ECFP-only inference constraint;
- fixed distillation and naive adaptive weighting limitations from the pilot;
- pretrained teacher selection plus frozen routing distillation;
- main RMSE result;
- mechanism analysis result.

## 1. Introduction

Low-resource ADMET prediction is difficult because labeled measurements are
scarce, noisy, and endpoint-specific. At the same time, molecules have
multiple inexpensive sources of predictive information, including ECFP
fingerprints, physicochemical descriptors, and predictions from traditional
machine learning models trained on these representations. Descriptor-access
models can be strong practical predictors, but many deployment settings still
favor a lightweight predictor that consumes only fingerprints at inference.

Training-time descriptor transfer addresses this mismatch by using descriptor
information during learning while keeping the deployed student ECFP-only. The
pilot study shows that descriptor reconstruction and descriptor-teacher
distillation can improve an ECFP-only AdapterFusion student for low-resource
regression ADMET. However, the pilot also exposes a limitation: fixed
distillation strength is not uniformly optimal, and validation-advantage
adaptive weighting underperforms fixed strong distillation. This suggests that
the unresolved problem is not whether teacher knowledge can help, but when a
student should trust each teacher.

This paper studies teacher selection in low-resource ADMET distillation. We
ask: **how can an ECFP-only student learn which molecular teacher to trust
under low-resource supervision, and why do jointly learned routing gates
underperform?** We propose a two-stage method that first trains a standalone
teacher selector from cross-fit pseudo-oracle labels and then freezes that
selector to route teacher supervision into an ECFP-only AdapterFusion student.

Contributions:

1. We formulate ECFP-only low-resource ADMET prediction as a selective
   teacher-reliability distillation problem.
2. We show that oracle best-teacher labels are predictable from teacher
   uncertainty, consensus deviation, validation priors, and descriptor-space
   coverage, while jointly learned routing gates remain unstable.
3. We propose cross-fit selector pretraining plus frozen top-1 routing, with
   selector-based filtering as a simplified companion variant.
4. We evaluate the method on regression ADMET datasets and analyze which
   teachers are selected across endpoints, train ratios, and chemical regions.

## 2. Problem Formulation

Let \(x\) denote a molecule, \(f(x)\) its ECFP4 fingerprint, \(d(x)\) its
standardized RDKit descriptor vector, and \(y\) the ADMET target. The student
model \(S_\theta\) is constrained to use only \(f(x)\) at inference:

\[
\hat{y}_S = S_\theta(f(x)).
\]

During training, descriptor information and teacher predictions may be used.
Let \(T_1,\ldots,T_K\) be molecular expert teachers, where each teacher
produces:

\[
\hat{y}_i = T_i(x).
\]

The goal is not to average teacher predictions uniformly. Instead, the student
uses a learned teacher selector:

\[
\pi(x)=h_\phi(r(x)),
\]

where \(r(x)\) contains teacher-selection features derived from teacher
uncertainty, consensus deviation, validation priors, and OOD coverage. The
selector is trained first, then frozen, and finally used to route teacher
distillation while keeping inference ECFP-only.

## 3. Method

### 3.1 ECFP-Only AdapterFusion Student

The student follows the AdapterFusion backbone from the pilot study. An ECFP4
fingerprint is encoded into \(z_{fp}\). A pseudo-descriptor adapter maps this
representation into \(z_{desc}\), and a learned gate fuses the two
representations for target prediction:

\[
z_{fp}=E(f(x)), \quad z_{desc}=A(z_{fp}),
\]

\[
g=\sigma(G([z_{fp},z_{desc}])), \quad
z_{fused}=g\odot z_{fp}+(1-g)\odot z_{desc}.
\]

The target head predicts from \(z_{fused}\), while the descriptor head predicts
\(d(x)\) during training. At inference, the model consumes only ECFP4.

### 3.2 Molecular Expert Teachers

The first version uses reproducible RF/XGB teachers:

- fingerprint teachers: `ECFP4_RF`, `ECFP4_XGB`;
- descriptor-only teachers: `Desc_RF`, `Desc_XGB`;
- descriptor-access teachers: `ECFP4_Desc_RF`, `ECFP4_Desc_XGB`.

These teachers represent different molecular views and modeling biases.
Chemprop is deferred until the RF/XGB reliability pipeline is established.

### 3.3 Teacher-Selection Features

For teacher \(T_i\), teacher selection is estimated from:

- validation quality \(q_i\);
- teacher uncertainty \(u_i(x)\);
- teacher agreement \(a_i(x)\);
- teacher-student disagreement \(\delta_i(x)\);
- descriptor-space OOD score \(o(x)\).

### 3.4 Cross-Fit Selector Pretraining

Within each dataset-ratio-seed training split, teachers are re-fit in
cross-validation folds to generate out-of-fold train predictions. The
pseudo-oracle selector label is the teacher with minimum absolute error on each
held-out train sample:

\[
y^{sel}(x)=\arg\min_i |\hat{y}_i(x)-y|.
\]

A selector \(h_\phi\) is trained on these labels. The first formal version uses
a random-forest selector, with logistic regression as a lighter baseline.

### 3.5 Frozen Routing Distillation

After selector pretraining, the selector is frozen. For each training sample,
it outputs either:

- a top-1 teacher for hard routing; or
- a filtered teacher subset for simplified multi-teacher distillation.

The main version uses hard top-1 routing:

\[
i^*(x)=\arg\max_i h_\phi(r(x))_i,
\qquad
\mathcal{L}_{distill}=
\mathrm{Huber}(\hat{y}_S,\hat{y}_{i^*(x)}).
\]

The simplified filtering version keeps only trusted teachers and applies
validation-weighted distillation on the surviving subset.

### 3.5 Training Objective

For regression:

\[
\mathcal{L}=
\mathcal{L}_{task}
+\lambda_{desc}\mathcal{L}_{desc}
+\lambda_{mt}\sum_i w_i(x)c(x)
\mathrm{Huber}(\hat{y}_S,\hat{y}_i).
\]

Defaults:

- \(\lambda_{desc}=0.1\);
- \(\lambda_{mt}=1.0\);
- Huber teacher loss;
- MSE teacher loss as ablation.

## 4. Experiments

### 4.1 Datasets And Protocol

Main experiments use five regression ADMET datasets:

- `caco2_wang`;
- `lipophilicity_astrazeneca`;
- `solubility_aqsoldb`;
- `vdss_lombardo`;
- `ppbr_az`.

Each dataset uses 10%, 20%, and 50% training ratios with seeds 42, 123, and
3407. The primary metric is test RMSE.

### 4.2 Baselines

Student baselines:

- base AdapterFusion;
- fixed `ECFP4_Desc_RF` distillation with `lambda_distill=1.0`;
   - uniform multi-teacher distillation;
   - validation-weighted multi-teacher distillation;
   - top-1 validation teacher distillation;
   - jointly learned soft/hard routing gates.

Descriptor-access controls:

- `ECFP4_RF`;
- `Desc_RF`;
- `ECFP4_Desc_RF`;
- `ECFP4_XGB`;
- `Desc_XGB`;
- `ECFP4_Desc_XGB`.

### 4.3 Main Method

Compare:

- pretrained selector hard top-1 routing;
- pretrained selector top-2 routing;
- selector-filtered validation-weighted distillation.

### 4.4 Evaluation Questions

1. Does selective distillation improve over base AdapterFusion?
2. Does it improve over fixed single-teacher distillation?
3. Does it improve over uniform multi-teacher distillation?
4. Does conflict-aware top-k help under high teacher disagreement?
5. Are teacher weights interpretable across endpoint, train ratio, and
   descriptor-space coverage?

## 5. Results

Placeholder until experiments are complete.

Required result blocks:

- main RMSE comparison;
- aggregate win counts;
- mean RMSE deltas;
- per-dataset discussion;
- comparison against descriptor-access controls without overclaiming.

## 6. Mechanism Analysis

Placeholder until diagnostics are complete.

Required analyses:

- teacher weight heatmap;
- teacher conflict vs gain;
- uncertainty-error correlation;
- OOD distance vs gate weight;
- failure-case table.

## 7. Discussion

Expected discussion points:

- fixed distillation is useful but too coarse;
- validation-only adaptive weighting is insufficient;
- sample-level reliability and conflict handling provide a stronger
  formulation;
- descriptor-access RF/XGB remains a strong control;
- ECFP-only inference is still valuable when descriptors or teachers should not
  be required at deployment;
- Chemprop/pretrained teachers are a natural next extension.

## 8. Limitations

Planned limitations:

- main claim is regression-focused;
- classification remains supplementary unless stabilized;
- RF/XGB teachers are not the full space of molecular teachers;
- reliability features are partly hand-designed in the first version;
- student may still underperform descriptor-access teachers.
