# Second Paper Experiment Matrix

## Goal

Before training the full high-venue method, prove that teacher reliability is
non-uniform across tasks, train ratios, and samples. Then compare the proposed
reliability-gated conflict-aware method against fixed and naive multi-teacher
distillation baselines.

## Fixed Experimental Scope

Datasets:

- `caco2_wang`
- `lipophilicity_astrazeneca`
- `solubility_aqsoldb`
- `vdss_lombardo`
- `ppbr_az`

Train ratios:

- 10
- 20
- 50

Seeds:

- 42
- 123
- 3407

Metric:

- test RMSE, lower is better

Primary student:

- `ECFP4_MLP_DescAdapterFusion`

Inference rule:

- student uses ECFP4 only;
- true RDKit descriptors and teacher models are not used at inference.

## Teacher Set

First-round teachers:

- `ECFP4_RF`
- `Desc_RF`
- `ECFP4_Desc_RF`
- `ECFP4_XGB`
- `Desc_XGB`
- `ECFP4_Desc_XGB`

Deferred teacher:

- Chemprop D-MPNN, only after RF/XGB reliability diagnostics are stable.

## Stage 1: Teacher Reliability Diagnostics

No new student training in this stage.

### Inputs

For every dataset, ratio, seed, teacher:

- train predictions;
- validation predictions;
- test predictions;
- teacher validation metrics;
- teacher test metrics;
- uncertainty estimate where available;
- sample identifiers / SMILES order.

### Required Outputs

1. **Teacher performance table**
   - dataset;
   - ratio;
   - seed;
   - teacher;
   - valid RMSE;
   - test RMSE.

2. **Teacher agreement matrix**
   - pairwise teacher prediction correlation;
   - pairwise teacher RMSE distance;
   - summarized by dataset and ratio.

3. **Oracle setting-level teacher**
   - best validation teacher;
   - best test teacher;
   - whether validation-selected teacher matches test-selected teacher.

4. **Oracle sample-level teacher on validation**
   - for each validation sample, identify teacher with lowest absolute error;
   - summarize teacher win counts by dataset and ratio.

5. **Uncertainty-error correlation**
   - teacher uncertainty vs absolute prediction error;
   - per teacher and dataset-ratio.

6. **Coverage-error correlation**
   - descriptor-space kNN distance vs teacher error;
   - ECFP/Tanimoto distance vs teacher error if feasible.

### Decision Rule

Proceed to sample-level reliability gating if:

- multiple teachers win in different settings or sample regions;
- teacher disagreement varies substantially by dataset-ratio;
- uncertainty or OOD features correlate with error for at least some teachers.

Pivot to single-teacher reliability weighting if:

- `ECFP4_Desc_RF` or `ECFP4_Desc_XGB` dominates almost every setting;
- but uncertainty/OOD features still identify harmful teacher samples.

## Stage 2: Multi-Teacher Baselines

Train the AdapterFusion student with:

1. **Uniform multi-teacher distillation**
   - equal teacher weights;
   - all RF/XGB teachers included.

2. **Validation-weighted multi-teacher distillation**
   - teacher weights are dataset-ratio-level softmax over validation RMSE.

3. **Top-1 validation teacher**
   - use only the best validation teacher for that dataset-ratio-seed.

Purpose:

- prove that the proposed method is not just benefiting from more teachers.

## Stage 3: Main Method

Train:

- reliability-gated multi-teacher AdapterFusion;
- conflict-aware top-k version;
- Huber distillation loss default;
- MSE distillation loss ablation.

Default reliability score:

\[
R_i(x)=
\alpha \tilde{q}_i
-\beta \tilde{u}_i(x)
+\eta \tilde{a}_i(x)
-\gamma \tilde{\delta}_i(x)
-\rho \tilde{o}(x)
\]

Teacher weights:

\[
w_i(x)=\mathrm{softmax}_i(R_i(x)).
\]

Conflict score:

\[
\kappa(x)=\mathrm{Var}_i(\hat{y}_{T_i}(x)).
\]

Conflict handling:

- low conflict: weighted all-teacher distillation;
- high conflict: top-k distillation by reliability.

## Main Comparison Table

Rows:

- dataset;
- train ratio.

Columns:

- AdapterFusion;
- fixed `ECFP4_Desc_RF` distillation, `lambda_distill=1.0`;
- uniform multi-teacher;
- validation-weighted multi-teacher;
- top-1 validation teacher;
- reliability-gated multi-teacher;
- conflict-aware top-k reliability gate;
- `ECFP4_Desc_RF`;
- `ECFP4_Desc_XGB`;
- delta vs AdapterFusion;
- delta vs fixed single-teacher.

## Ablation Table

Ablations:

- no validation-quality prior;
- no uncertainty;
- no teacher agreement;
- no teacher-student disagreement;
- no descriptor-space OOD;
- no conflict top-k;
- MSE teacher loss instead of Huber;
- uniform teacher weights.

Each ablation reports:

- mean RMSE delta vs AdapterFusion;
- mean RMSE delta vs fixed single-teacher;
- settings improved over AdapterFusion;
- settings improved over fixed single-teacher.

## Diagnostic Figures

### Teacher Weight Heatmap

Rows:

- dataset-ratio.

Columns:

- teacher models.

Values:

- mean teacher weight on validation/test samples.

### Teacher Conflict vs Gain

x-axis:

- mean teacher prediction variance or pairwise disagreement.

y-axis:

- RMSE gain over fixed single-teacher distillation.

### OOD Distance vs Gate Weight

x-axis:

- descriptor-space kNN distance.

y-axis:

- average teacher weight or teacher error.

### Teacher Uncertainty Calibration

x-axis:

- uncertainty bin.

y-axis:

- absolute prediction error.

## Acceptance Criteria

Strong method signal:

- reliability-gated method beats AdapterFusion in at least 12/15 settings;
- beats fixed single-teacher in at least 8/15 settings;
- mean RMSE improves by at least 0.02 over fixed single-teacher;
- no catastrophic failure on `vdss_lombardo`;
- gate weight patterns are interpretable.

Acceptable but weaker signal:

- beats AdapterFusion in most settings but not fixed single-teacher;
- use as reliability analysis paper or improve gate.

Stop condition:

- no teacher diversity;
- no oracle selection headroom;
- no reliability feature correlates with teacher error.

