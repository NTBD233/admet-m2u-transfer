# High-Venue Technical Design

## Working Title

**Selective Multi-Teacher Descriptor Distillation for Low-Resource ADMET Prediction**

## Starting Point From The Pilot Paper

The current repository already supports a complete first-paper story:

- Descriptor-access RF is a strong practical upper-bound baseline.
- AdapterFusion is consistently stronger than simple descriptor prediction on
  regression ADMET.
- Fixed descriptor-teacher distillation improves AdapterFusion in 11/15
  regression dataset-ratio settings.
- The best fixed distillation strength varies by dataset and train ratio.
- Validation teacher-advantage adaptive weighting is a negative ablation.
- Classification remains unstable and should not be the main high-venue claim
  until regression is mechanistically stronger.

The high-venue version should therefore not be framed as "more baselines" or
"larger benchmark only." The stronger question is:

> When should an ECFP-only student trust each molecular teacher, and how should
> the student handle teacher conflict under low-resource ADMET?

## Core Claim To Build Toward

The intended contribution is selective teacher reliability, not simply
multi-teacher distillation.

Proposed high-venue claim:

> A student with ECFP-only inference can benefit more reliably from
> descriptor-informed molecular experts when distillation is gated by
> task-level and sample-level teacher reliability.

This keeps the inference constraint from the pilot paper while turning the
pilot limitations into the new method.

## Why This Is A Plausible High-Venue Direction

Recent molecular property work points toward multi-view and cross-view
learning, but the current gap is that views and teachers are not uniformly
reliable.

- MolFuse explicitly motivates molecular property prediction with multiple
  molecular views and notes that low-quality views can harm fusion when view
  quality disparity is ignored: <https://www.ijcai.org/proceedings/2024/0621>
- MolKD shows that molecular property prediction can benefit from
  cross-modal knowledge distillation, supporting the broader idea that useful
  molecular knowledge can be transferred across modalities:
  <https://neurips.cc/virtual/2024/102868>
- MTGL-ADMET shows that ADMET auxiliary knowledge should be selected
  adaptively rather than used uniformly:
  <https://doi.org/10.1016/j.isci.2023.108285>
- TDC's ADMET benchmark group provides a natural path for expanded evaluation:
  <https://tdcommons.ai/benchmark/admet_group/overview>
- Chemprop is a reproducible D-MPNN package and a practical graph-teacher
  candidate once the RF/XGBoost teacher pipeline is stable:
  <https://doi.org/10.1021/acs.jcim.3c01250>

## Method Overview

### Student

Use the current `ECFP4_MLP_DescAdapterFusion` as the initial student:

- input: ECFP4 only;
- training auxiliary signal: standardized RDKit descriptor reconstruction;
- representation: ECFP encoder, pseudo-descriptor adapter, gated fusion;
- inference: ECFP4 only, no true descriptors and no teacher models required.

Later variants can replace the encoder with a residual/calibrated ECFP MLP, but
the first high-venue extension should keep the student backbone fixed so that
teacher reliability is the isolated contribution.

### Teachers

Start with teachers that can be generated from the existing pipeline:

1. **Descriptor teacher**
   - `ECFP4_Desc_RF`
   - `ECFP4_Desc_XGB`
   - represents descriptor-access physical/chemical knowledge.

2. **Fingerprint teacher**
   - `ECFP4_RF`
   - `ECFP4_XGB`
   - represents non-neural fingerprint pattern recognition.

3. **Descriptor-only teacher**
   - `Desc_RF`
   - `Desc_XGB`
   - separates pure descriptor signal from fingerprint-plus-descriptor signal.

4. **Graph/pretrained teacher**
   - Chemprop D-MPNN first.
   - ADMET-AI or another pretrained ADMET model can be considered later, but
     only after versioning and leakage risks are clear.

The first implementation milestone should use RF/XGBoost teachers only. Graph
teachers add value for the final paper, but they should not block the
reliability-gating experiment.

## Reliability Signals

For each teacher \(T_i\), dataset \(d\), train ratio \(r\), seed \(s\), and
training sample \(x\), compute reliability features that do not use test
labels.

### Teacher Uncertainty

Regression:

- RF: per-tree prediction standard deviation.
- XGBoost: ensemble-over-seeds or bootstrap ensemble standard deviation.
- Chemprop: ensemble standard deviation if using multiple checkpoints.

Classification:

- predictive entropy;
- probability margin;
- ensemble variance if available.

### Teacher Validation Quality

Task-level prior reliability:

\[
q_i(d,r,s) =
\begin{cases}
-\mathrm{RMSE}_{valid}(T_i), & \text{regression}\\
\mathrm{ROC\text{-}AUC}_{valid}(T_i), & \text{classification}
\end{cases}
\]

Normalize across teachers within the same dataset-ratio-seed. This gives a
global teacher prior, but it must not be the only reliability signal because
the existing adaptive validation-advantage method already failed.

### Teacher-Student Disagreement

Use a stop-gradient student prediction:

\[
\delta_i(x) = |\hat{y}_{T_i}(x) - \mathrm{sg}(\hat{y}_{student}(x))|.
\]

Large disagreement can mean either useful correction or unreliable teacher
pressure. It should be combined with teacher uncertainty and teacher agreement,
not used alone.

### Teacher Agreement

For \(K\) teachers, define:

\[
a_i(x) = -\frac{1}{K-1}\sum_{j \neq i}
|\hat{y}_{T_i}(x) - \hat{y}_{T_j}(x)|.
\]

High agreement supports stronger distillation. High conflict should reduce
global distillation strength or select the most reliable teacher rather than
averaging all teachers.

### Coverage / Out-Of-Domain Score

Compute distance from a sample to the training chemical space:

- kNN distance in descriptor space;
- kNN distance in ECFP latent or Tanimoto space;
- optional isolation forest score on descriptors.

The expected behavior is:

- near-distribution samples: trust descriptor/fingerprint teachers more;
- far samples: reduce teacher pressure unless teachers agree and uncertainty
  is low.

## Gating Variants

### Variant A: Fixed Multi-Teacher Average

Baseline:

\[
\mathcal{L}_{distill} =
\frac{1}{K} \sum_i
\ell(\hat{y}_{student}, \hat{y}_{T_i}).
\]

This tests whether simply adding teachers is enough. It is not expected to be
the final method.

### Variant B: Task-Level Teacher Weighting

Use one weight per teacher for each dataset-ratio setting:

\[
w_i(d,r) = \mathrm{softmax}(\gamma q_i(d,r)).
\]

This is a conservative midpoint between global fixed lambda and sample-level
gating.

### Variant C: Rule-Based Sample Reliability Gate

The first serious method candidate:

\[
w_i(x) =
\mathrm{softmax}_i(
\alpha \tilde{q}_i
- \beta \tilde{u}_i(x)
- \gamma \tilde{\delta}_i(x)
+ \eta \tilde{a}_i(x)
- \rho \tilde{o}(x)
).
\]

Where:

- \(\tilde{q}_i\): normalized validation reliability;
- \(\tilde{u}_i(x)\): normalized teacher uncertainty;
- \(\tilde{\delta}_i(x)\): normalized teacher-student disagreement;
- \(\tilde{a}_i(x)\): normalized teacher agreement;
- \(\tilde{o}(x)\): normalized out-of-domain score.

Start with a small grid over \(\alpha,\beta,\gamma,\eta,\rho\). Keep this
version deliberately simple and reproducible.

### Variant D: Conflict-Aware Top-k Teacher Selection

When teacher disagreement is high, avoid averaging all teachers:

\[
i^*(x) = \arg\max_i R_i(x),
\quad
\mathcal{L}_{distill} =
\ell(\hat{y}_{student}, \hat{y}_{T_{i^*}}).
\]

Use top-1 or top-2 teachers only when the teacher-conflict score exceeds a
threshold. Otherwise use the soft gate.

### Variant E: Learned Reliability Gate

Train a small MLP gate on reliability features. To avoid validation leakage:

- split the training set into student-train and gate-calibration folds; or
- use out-of-fold teacher predictions from cross-validation; or
- train the gate only as a differentiable weighting function without fitting
  to validation labels directly.

This should be a second-stage method after rule-based gating proves there is
signal.

## Training Objective

For regression:

\[
\mathcal{L} =
\mathcal{L}_{task}
+ \lambda_{desc}\mathcal{L}_{desc}
+ \lambda_{mt}
\sum_i w_i(x) c(x)
\mathrm{Huber}(\hat{y}_{student}, \hat{y}_{T_i}).
\]

Where:

- \(\lambda_{desc}=0.1\), inherited from the pilot paper;
- \(\lambda_{mt}\) starts at 1.0 because it is the best global fixed setting;
- \(w_i(x)\) is the teacher reliability gate;
- \(c(x)\) is a conflict confidence scalar;
- Huber loss is preferred over MSE in the high-venue version to reduce damage
  from noisy teacher outliers.

Conflict confidence:

\[
c(x) = \exp(-\mathrm{Var}_i(\hat{y}_{T_i}(x)) / \tau).
\]

Fallback: if Huber does not help, keep MSE for direct comparability with the
pilot paper.

## Experiment Plan

### Stage 0: Freeze Pilot Baseline

No new method yet. Confirm the current tables are the baseline:

- base AdapterFusion;
- fixed single-teacher distillation with \(\lambda_{distill}=1.0\);
- best fixed lambda per setting as an oracle analysis;
- adaptive validation-advantage negative ablation.

### Stage 1: Multi-Teacher Diagnostics Without New Student Training

Generate teacher predictions for RF/XGBoost teachers:

- `ECFP4_RF`
- `Desc_RF`
- `ECFP4_Desc_RF`
- `ECFP4_XGB`
- `Desc_XGB`
- `ECFP4_Desc_XGB`

Produce diagnostic tables:

- per-teacher validation/test performance;
- teacher agreement matrix by dataset-ratio;
- oracle best teacher per setting;
- oracle best teacher per sample on validation only;
- teacher uncertainty vs error correlation;
- descriptor-space distance vs teacher error correlation.

Decision gate:

- Continue only if different teachers win on different datasets/ratios or
  sample regions.
- If one teacher dominates everywhere, the high-venue story should shift to
  better single-teacher distillation, not multi-teacher selection.

### Stage 2: Fixed Multi-Teacher Baselines

Train:

- uniform average teacher distillation;
- validation-performance weighted teacher distillation;
- top-1 validation teacher distillation.

These are necessary controls against the selective method.

### Stage 3: Rule-Based Sample Reliability Gate

Train the rule-based gate using only RF/XGBoost teachers.

Primary comparison:

- vs base AdapterFusion;
- vs fixed `ECFP4_Desc_RF` distillation with \(\lambda=1.0\);
- vs best fixed lambda analysis;
- vs uniform multi-teacher distillation;
- vs validation-weighted multi-teacher distillation.

Success threshold for a serious paper signal:

- beats base AdapterFusion in at least 12/15 regression settings;
- beats fixed single-teacher \(\lambda=1.0\) in at least 8/15 settings;
- improves mean RMSE delta over fixed single-teacher by at least 0.02;
- does not create a large failure on `vdss_lombardo`, where fixed strong
  distillation is unstable.

### Stage 4: Add Graph Teacher

Only after Stage 3 is interpretable:

- train Chemprop on the same splits;
- export validation/train/test predictions;
- include Chemprop as an additional teacher and as a standalone baseline.

The paper should show whether graph teachers add complementary knowledge or
mainly duplicate fingerprint teacher behavior.

### Stage 5: Classification Revisit

Keep classification supplementary unless selective gating clearly stabilizes
ROC-AUC.

Minimum requirement to promote classification to main paper:

- improvement over base AdapterFusion in a majority of classification
  settings;
- no collapse on small positive/negative class splits;
- calibration or threshold-independent metrics reported consistently.

## Baseline Matrix

Primary regression baselines:

- `ECFP4_MLP`
- `ECFP4_MLP_DescPred`
- `ECFP4_MLP_DescAdapterFusion`
- `ECFP4_MLP_DescConcat`
- `ECFP4_RF`
- `Desc_RF`
- `ECFP4_Desc_RF`
- `ECFP4_XGB`
- `Desc_XGB`
- `ECFP4_Desc_XGB`
- Chemprop, once stable
- fixed single-teacher AdapterFusion
- uniform multi-teacher AdapterFusion
- task-weighted multi-teacher AdapterFusion
- sample-reliability gated AdapterFusion

Report descriptor-access models as controls, not as direct inference-equivalent
competitors.

## Figures To Target

1. **Method diagram**
   - ECFP-only student;
   - multiple teachers;
   - reliability feature extractor;
   - conflict-aware gate;
   - no teacher at inference.

2. **Main RMSE table**
   - base AdapterFusion;
   - fixed single-teacher;
   - uniform multi-teacher;
   - selective multi-teacher;
   - descriptor-access RF/XGB controls.

3. **Teacher reliability heatmap**
   - teacher weight by dataset and train ratio.

4. **Teacher conflict scatter**
   - x-axis: teacher disagreement;
   - y-axis: selective gain over fixed distillation.

5. **OOD reliability plot**
   - descriptor-space distance vs teacher error vs gate weight.

6. **Failure-case table**
   - settings where selective distillation hurts;
   - dominant teacher;
   - conflict score;
   - likely cause.

## First Implementation Scope

Do not start with the full learned gate. The first code milestone should be:

1. Extend teacher generation to support multiple teacher models in one run.
2. Add teacher uncertainty export for RF/XGBoost.
3. Add a diagnostics script:
   `analyze_teacher_reliability.py`.
4. Build teacher agreement and oracle-selection tables.
5. Decide whether there is enough evidence for sample-level selective gating.

This keeps the first high-venue step lightweight and directly connected to the
existing repository.

## Proposed New Files

Likely code additions:

- `utils/teacher_store.py`
- `utils/reliability.py`
- `analyze_teacher_reliability.py`
- `compare_multiteacher_distillation.py`

Likely future training changes:

- allow `train.py` to accept multiple teacher models;
- load per-teacher predictions and uncertainty arrays;
- compute rule-based reliability weights per batch;
- save per-run gate diagnostics.

## Risk Register

### Risk: XGBoost teachers do not beat RF teachers

Fallback:

- use XGBoost as diversity signal rather than main performance teacher;
- emphasize teacher disagreement and reliability, not teacher rank alone.

### Risk: One teacher dominates every setting

Fallback:

- pivot to single-teacher reliability gating;
- use uncertainty and OOD-aware sample weighting rather than multi-teacher
  selection.

### Risk: Rule-based gate does not beat fixed lambda

Fallback:

- report it as a systematic negative result;
- use oracle teacher selection to motivate why learned reliability is needed;
- move to learned gate only if oracle selection shows meaningful headroom.

### Risk: Chemprop is hard to reproduce

Fallback:

- keep Chemprop as standalone external baseline;
- submit high-venue version with RF/XGBoost teachers first if the reliability
  result is strong enough.

### Risk: Gains remain modest

Fallback:

- frame as controlled reliability analysis under strict ECFP-only inference;
- target a cheminformatics or biomedical AI venue rather than a top general ML
  venue.

## Immediate Next Decision

The next concrete experiment should be **teacher reliability diagnostics**, not
new student training.

Reason:

- current adaptive weighting failed because reliability was too coarse;
- before building a gate, we need evidence that teacher reliability varies by
  task or sample;
- diagnostic tables are cheap compared with full student training;
- the result will tell whether the high-venue method should be multi-teacher,
  single-teacher reliability-weighted, or stronger-student distillation.

