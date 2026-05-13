# Reliability-Gated Conflict-Aware Multi-Teacher Distillation for Low-Resource ADMET Prediction

## Abstract

Placeholder. Write this after Stage 1 diagnostics and Stage 3 main results are
available.

Must include:

- low-resource regression ADMET motivation;
- ECFP-only inference constraint;
- fixed distillation and naive adaptive weighting limitations from the pilot;
- reliability-gated conflict-aware multi-teacher distillation;
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

This paper studies teacher reliability in low-resource ADMET distillation. We
ask: **when should an ECFP-only student trust each molecular teacher, and how
should it learn when teachers disagree?** We propose Reliability-Gated
Conflict-Aware Multi-Teacher Distillation, which trains an ECFP-only
AdapterFusion student from multiple RF/XGB molecular teachers while estimating
teacher reliability at the task and sample levels.

Contributions:

1. We formulate ECFP-only low-resource ADMET prediction as a selective
   teacher-reliability distillation problem.
2. We propose a reliability gate using validation quality, teacher
   uncertainty, teacher agreement, teacher-student disagreement, and
   descriptor-space coverage.
3. We introduce conflict-aware top-k distillation to avoid harmful averaging
   when teachers disagree.
4. We evaluate the method on regression ADMET datasets and analyze which
   teachers are trusted across endpoints, train ratios, and chemical regions.

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
learns a sample-level teacher weight:

\[
w_i(x)=g(r_i(x), \kappa(x)),
\]

where \(r_i(x)\) contains reliability features for teacher \(T_i\), and
\(\kappa(x)\) measures teacher conflict. The training objective uses teacher
knowledge only when it is estimated to be reliable, while keeping inference
ECFP-only.

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

### 3.3 Reliability Features

For teacher \(T_i\), reliability is estimated from:

- validation quality \(q_i\);
- teacher uncertainty \(u_i(x)\);
- teacher agreement \(a_i(x)\);
- teacher-student disagreement \(\delta_i(x)\);
- descriptor-space OOD score \(o(x)\).

The initial transparent reliability score is:

\[
R_i(x)=
\alpha \tilde{q}_i
-\beta \tilde{u}_i(x)
+\eta \tilde{a}_i(x)
-\gamma \tilde{\delta}_i(x)
-\rho \tilde{o}(x).
\]

Teacher weights are then:

\[
w_i(x)=\mathrm{softmax}_i(R_i(x)).
\]

### 3.4 Conflict-Aware Top-k Distillation

Teacher conflict is measured by prediction variance:

\[
\kappa(x)=\mathrm{Var}_i(\hat{y}_i).
\]

If conflict is low, the student distills from all teachers using reliability
weights. If conflict is high, the student distills only from the top-k most
reliable teachers. This prevents a poor or out-of-domain teacher from dragging
the target toward an unhelpful average.

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
- top-1 validation teacher distillation.

Descriptor-access controls:

- `ECFP4_RF`;
- `Desc_RF`;
- `ECFP4_Desc_RF`;
- `ECFP4_XGB`;
- `Desc_XGB`;
- `ECFP4_Desc_XGB`.

### 4.3 Main Method

Compare:

- reliability-gated multi-teacher distillation;
- reliability-gated conflict-aware top-k distillation;
- Huber vs MSE distillation loss.

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

