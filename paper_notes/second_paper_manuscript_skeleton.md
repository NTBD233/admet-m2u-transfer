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

Do not treat selector-confidence auto reweighting as a main method unless later
experiments overturn the current partial-regression result. In the current
evidence package, auto reweighting is a diagnostic follow-up: it is
reproducible and locally helpful, but weaker than plain selector routing on
aggregate.

### 4.4 Evaluation Questions

1. Does selective distillation improve over base AdapterFusion?
2. Does it improve over fixed single-teacher distillation?
3. Does it improve over uniform multi-teacher distillation?
4. Does conflict-aware top-k help under high teacher disagreement?
5. Are teacher weights interpretable across endpoint, train ratio, and
   descriptor-space coverage?

## 5. Results

### 5.1 Main Regression Results

Table 1 summarizes the full regression matrix over five ADMET endpoints, three
train ratios, and three random seeds. The base ECFP-only AdapterFusion student
obtains a mean test RMSE of 5.1841. Fixed single-teacher distillation from the
descriptor-access `ECFP4_Desc_RF` teacher improves this to 5.1621, confirming
the pilot observation that descriptor-informed teacher supervision can benefit
an ECFP-only student. However, the gain is modest, and fixed teacher
distillation still loses to stronger multi-teacher selection strategies.

Uniform multi-teacher distillation improves mean test RMSE to 5.1403, while
validation-weighted multi-teacher distillation improves it further to 5.0969.
This shows that multiple molecular teachers contain complementary information,
but also that simple averaging is insufficient. The best setting-level
baseline is `top1_validation`, which selects the single teacher with the best
validation RMSE for each dataset-ratio-seed setting and reaches 5.0792 mean
test RMSE.

The proposed pretrained selector with frozen top-1 routing reaches 5.0889 mean
test RMSE. It improves over base AdapterFusion in 34 of 45 runs and over fixed
single-teacher distillation in 28 of 45 runs, with mean deltas of -0.0953 and
-0.0732 RMSE, respectively. Compared with `top1_validation`, the selector wins
25 of 45 runs but remains slightly worse on average by +0.0096 RMSE. Thus, the
selector closes most of the gap between unstable joint routing and the
strongest setting-level teacher selector, but it does not decisively dominate
the setting-level baseline.

Adding high-resource lambda decay further improves the pretrained selector.
This calibrated variant keeps the original distillation strength at 10% and
20% train ratios, but reduces teacher forcing at 50% train ratio. It reaches
5.0814 mean test RMSE, improves over fixed distillation in 32 of 45 runs, and
improves over the plain pretrained selector on mean RMSE by -0.0075. It also
wins against `top1_validation` in 27 of 45 runs, although its aggregate mean
remains marginally higher than `top1_validation` by +0.0021 RMSE. We therefore
treat high-resource decay as a useful calibration of selector routing rather
than as a separate main contribution.

Overall, the main result supports a restrained claim: teacher-selector
supervision makes sample-level routing competitive with strong setting-level
teacher selection, and the remaining performance gap is small enough to
motivate mechanism analysis rather than another purely heuristic routing rule.

### 5.2 Dataset And Train-Ratio Behavior

The dataset-ratio table shows that the benefit of selector routing is not
uniform. On `caco2_wang`, the pretrained selector is strongest at 10% and 20%
train ratios, while high-resource decay substantially improves the 50% setting
relative to the plain selector. The same high-resource pattern appears on
`ppbr_az`, where the calibrated selector improves the 50% setting from
14.4500 to 14.3472 mean RMSE and beats `top1_validation` at that ratio.

For `lipophilicity_astrazeneca`, pretrained selector routing is strongest at
20%, while the 50% setting is still best served by fixed descriptor-access
teacher distillation. For `solubility_aqsoldb`, the differences among uniform
multi-teacher, top-1 validation, and selector routing are small; high-resource
decay is essentially neutral. For `vdss_lombardo`, base AdapterFusion remains
competitive in the 20% and 50% settings, and high-resource decay slightly
weakens the plain selector at 50%. This endpoint therefore remains a useful
counterexample: not every dataset benefits from more teacher forcing control,
and selective distillation should be evaluated against a strong non-distilled
student.

These endpoint differences are important for the paper's central argument.
They show that teacher selection is not reducible to a single globally best
teacher or a single globally best weighting rule. The optimal behavior depends
on the endpoint and the resource regime, which motivates both selector
pretraining and the analysis of when the selected teacher should be enforced
strongly.

### 5.3 Negative And Secondary Ablations

Several alternatives were tested before settling on pretrained selector
routing. Joint learned gates and supervised joint gates were weak in smoke
experiments, indicating that simply adding a routing network to the student
does not provide a stable teacher-selection signal. Selector-confidence
auto-reweighting was reproducible and locally useful, but on the 18-run
`caco2_wang` + `ppbr_az` partial regression subset it increased mean RMSE from
8.6091 to 8.7875 relative to the plain selector. The route-mode audit reached
the same conclusion: the auto rule won against plain selector in only 7 of 18
settings and had a positive mean delta of +0.1784 RMSE.

The ratio-aware lambda analysis gives a more useful secondary result. Early
lambda decay, which reduces distillation strength already at the 20% train
ratio, is harmful on the same partial subset. High-resource decay is more
defensible because it leaves 10% and 20% settings unchanged and only weakens
teacher forcing at 50%. On the full matrix, this improves the plain selector
from 5.0889 to 5.0814 mean RMSE. This supports the view that teacher choice
and teacher strength are related but distinct problems: selecting a better
teacher is necessary, but enforcing that teacher too strongly can still harm
the student when more labeled data are available.

## 6. Mechanism Analysis

### 6.1 Teacher Selection Is Predictable

The first mechanism question is whether teacher selection contains learnable
signal at all. A leave-one-setting-out diagnostic was run using teacher
uncertainty, teacher consensus deviation, validation-derived teacher priors,
and descriptor-space coverage features. The random-forest gate probe reaches a
weighted oracle-teacher accuracy of 0.5617 across 45 held-out settings. This
substantially exceeds both the global majority baseline (0.4818) and the
setting-level `top1_validation` prior (0.4360). A logistic probe also beats
these baselines with 0.5044 weighted accuracy, but is clearly weaker than the
random-forest probe.

This diagnostic is central to the method design. It shows that the failure of
joint gates is not caused by teacher reliability being random or unobservable.
Instead, the issue is how the teacher-selection signal is supervised and how it
is coupled to student optimization. This motivates separating selector
pretraining from student distillation.

### 6.2 Selector Quality Across Endpoints

The pretrained random-forest selector remains noisy, but its quality is stable
across the full regression matrix. Averaged over 45 dataset-ratio-seed
settings, it obtains 0.5178 validation accuracy and 0.5425 test accuracy, with
validation and test macro-F1 of 0.4269 and 0.4396. These numbers are not high
enough to treat the selector as an oracle, but they are sufficient to route
useful supervision into the student.

Selector predictability varies by endpoint. Test accuracy is highest on
`solubility_aqsoldb` (0.5692) and `vdss_lombardo` (0.5546), followed by
`ppbr_az` (0.5450) and `caco2_wang` (0.5354). It is lowest on
`lipophilicity_astrazeneca` (0.5083), although the selector still performs
competitively in the student-level experiments for that endpoint. This
separation between selector accuracy and student gain is important: predicting
the pseudo-oracle teacher is useful, but the final RMSE also depends on how
well the ECFP-only student can absorb the selected teacher's signal.

### 6.3 Why Pretraining Helps More Than Joint Routing

Joint routing gates attempt to learn task prediction and teacher selection
simultaneously from low-resource supervision. In this setting, the task loss
can suppress teacher-selection behavior before the gate has learned a stable
notion of teacher reliability. The supervised joint-gate smoke results confirm
this failure mode: adding a gate classification loss did not substantially
improve regression RMSE. By contrast, cross-fit selector pretraining creates an
explicit pseudo-oracle target before the student is trained, and frozen routing
prevents the selector from collapsing toward whichever teacher happens to
reduce early training loss.

This explains why the proposed method should be framed as supervised teacher
selection rather than generic adaptive weighting. The contribution is not that
a more flexible gate is added to the student, but that teacher selection is
given its own supervision signal and then used as a fixed training-time
routing mechanism.

### 6.4 Remaining Failure Modes

The remaining gap to `top1_validation` is small but informative. First, some
settings are still better handled by a coarse setting-level teacher choice.
For example, `ppbr_az` at 20% train ratio strongly favors `top1_validation`,
while the selector remains weaker despite strong average selector accuracy on
that endpoint. This suggests that selector correctness against sample-level
pseudo-oracle labels is not always aligned with the student-level RMSE gained
from routed distillation.

Second, the high-resource lambda results show that teacher strength matters
even after teacher choice is fixed. At 50% train ratio, reducing distillation
strength improves the calibrated selector mean RMSE from 4.6001 to 4.5774
across datasets. At 10% and 20%, the calibrated method intentionally matches
the plain selector, because early weakening of teacher supervision hurt the
partial-regression results. The mechanism is therefore two-dimensional:
students need to know which teacher to imitate and how strongly to imitate it
under a given data regime.

Finally, descriptor-access models remain strong controls. The proposed method
does not claim that an ECFP-only student always replaces descriptor-access
teachers. Its contribution is narrower and more deployable: descriptor and
teacher knowledge can be transferred during training, while inference remains
ECFP-only.

## 7. Discussion

Expected discussion points:

- fixed distillation is useful but too coarse;
- validation-only adaptive weighting is insufficient;
- sample-level reliability and conflict handling provide a stronger
  formulation;
- selector correctness and student usefulness are not equivalent;
- confidence-based route-strength scaling is locally meaningful but not yet
  robust enough to become the main method;
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
