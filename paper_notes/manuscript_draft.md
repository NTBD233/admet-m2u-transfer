# Descriptor-Teacher Guided Multi-to-Uni Transfer for Low-Resource Regression ADMET Prediction

## Abstract

Low-resource ADMET prediction is a recurring challenge in early-stage drug
discovery, where labeled assays are limited but inexpensive molecular knowledge
such as physicochemical descriptors is readily available. We study whether
descriptor knowledge can be transferred into an ECFP-only neural predictor, so
that descriptors guide training without being required at inference time. We
develop Descriptor-Teacher AdapterFusion, a lightweight ECFP4-based student
model that combines an adapter-based pseudo-descriptor representation with
descriptor reconstruction and teacher prediction distillation from an
ECFP4+descriptor random forest. Across five regression ADMET datasets, three
low-resource training ratios, and three random seeds, fixed teacher
distillation with `lambda_distill = 1.0` improves AdapterFusion in 11/15
dataset-ratio settings, with an average RMSE change of -0.0656 relative to the
undistilled AdapterFusion baseline. However, descriptor-access random forest
models remain stronger in most settings, indicating that the proposed method
partially transfers descriptor knowledge but does not replace direct descriptor
access. A lambda sweep and adaptive-weighting ablation further show that simple
validation-based adaptive distillation is not sufficient. These results support
a controlled descriptor-guided multi-to-uni transfer perspective for
low-resource regression ADMET prediction.

## 1. Introduction

Accurate ADMET prediction is important for early drug discovery because
absorption, distribution, metabolism, excretion, and toxicity properties affect
which candidate molecules can progress beyond screening [@Vamathevan2019;
@Butler2018; @Xiong2021ADMETlab]. In practice, many
ADMET prediction tasks are low-resource: labels are expensive to obtain,
measurements vary across assays, and each property may have a limited number of
reliable annotated compounds. This makes it useful to study lightweight models
and auxiliary molecular knowledge that can improve generalization under limited
supervision.

Molecular descriptors provide one such source of auxiliary knowledge. RDKit
physicochemical descriptors such as molecular weight, LogP, TPSA, hydrogen bond
counts, rotatable bonds, and ring counts are inexpensive to compute and often
correlate with ADMET behavior [@RDKit2026]. A direct way to exploit them is to concatenate
descriptors with molecular fingerprints or use descriptor-based traditional
models. However, this creates a descriptor-access inference setting. In
contrast, a multi-to-uni transfer setting asks whether descriptor knowledge can
be used during training while the deployed student model uses only a single
modality, here ECFP4 fingerprints [@Rogers2010].

This paper studies descriptor-guided multi-to-uni transfer for low-resource
ADMET prediction. We begin from an ECFP4 MLP baseline and compare simple
auxiliary descriptor prediction with a structured AdapterFusion student that
uses an ECFP-derived pseudo-descriptor adapter and a learned fusion gate. We
then add teacher prediction distillation from an `ECFP4_Desc_RF` model, using a
descriptor-access random forest as a teacher for an ECFP-only neural student.

Our central question is not whether a lightweight neural model can beat all
descriptor-access baselines. Instead, we ask a narrower and more controlled
question: can descriptor-access knowledge improve an ECFP-only student in
low-resource regression ADMET? The answer is positive but bounded. Teacher
distillation improves AdapterFusion in most regression settings, but the
descriptor-access teacher and descriptor-based RF baselines remain strong.

The contributions are:

- We formulate a lightweight descriptor-guided multi-to-uni transfer setup for
  low-resource ADMET prediction.
- We compare ECFP-only descriptor prediction, AdapterFusion, descriptor-access
  controls, random forest baselines, and teacher-distilled AdapterFusion under
  shared low-resource splits.
- We show that fixed descriptor-teacher distillation improves regression
  AdapterFusion in 11/15 dataset-ratio settings.
- We report lambda sensitivity, validation-selected lambda analysis, and a
  negative adaptive-weighting ablation, clarifying when simple teacher
  weighting is insufficient.

## 2. Related Work

Molecular representation learning has increasingly used auxiliary objectives
to inject chemical knowledge into learned representations. Molecular language
models such as MolBERT and related approaches motivate the use of
domain-relevant pretraining or auxiliary tasks rather than relying only on
generic representation learning objectives [@Fabian2020MolBERT]. Graph
pretraining methods such as molecular GNN pretraining, GROVER, and GraphMVP
similarly use self-supervised or cross-view objectives to improve downstream
molecular prediction [@Hu2020PretrainGNN; @Rong2020GROVER; @Liu2022GraphMVP].
Our work is
smaller in scale and does not use a large pretrained sequence encoder, but it
shares the principle that chemically meaningful auxiliary targets can shape
molecular representations.

Auxiliary learning and task-specific adaptation are also relevant. A simple
auxiliary objective may not transfer all useful knowledge to the downstream
task; the architecture through which auxiliary information is routed can
matter [@Ruder2017; @Dey2024AuxAdapt]. In this work, `DescPred` represents plain descriptor prediction, while
AdapterFusion represents a task-specific transfer mechanism that creates and
fuses a pseudo-descriptor representation.

The closest conceptual motivation is multi-to-uni molecular representation
learning, including M2UMol-style knowledge transfer [@Xiong2026M2UMol]. Those
works study how multimodal molecular knowledge can improve unimodal inference.
Our setting is deliberately controlled: descriptors are the auxiliary modality,
ECFP4 is the inference modality, and ADMET prediction is the downstream task.

ADMET benchmark modeling provides the application context. We use TDC-style
ADMET datasets and evaluate low-resource splits across regression and
classification properties [@Huang2021TDC]. MoleculeNet provides an earlier
benchmarking reference point for molecular property prediction metrics, splits,
and baselines [@Wu2018MoleculeNet]. The main method claim is limited to
regression ADMET because the observed distillation signal is more consistent
there than in classification.

Our baseline choices are also grounded in prior molecular property prediction
practice. Fingerprint and descriptor models remain important controls in
cheminformatics, while graph neural networks and directed message-passing
models provide stronger learned-representation baselines in larger studies
[@Kearnes2016GraphConv; @Yang2019DMPNN; @Xiong2020AttentiveFP; @Heid2024Chemprop].
Tree ensembles are included because random forests and gradient boosting
methods remain competitive for tabular molecular features [@Breiman2001;
@Chen2016XGBoost; @Ke2017LightGBM; @Tian2022ADMETBoost].

## 3. Method

### 3.1 Problem Setting

Each molecule has an ECFP4 fingerprint `x_ecfp`, optional descriptor vector
`x_desc`, and target `y`. The primary inference constraint is that the student
model should use only `x_ecfp` at test time. Descriptor information may be used
during training as auxiliary supervision or through teacher predictions.

For regression tasks, the task loss is mean squared error. For classification
reference experiments, the task loss is binary cross entropy. The main paper
focuses on regression.

### 3.2 ECFP Encoder

The base ECFP model maps a 2048-dimensional ECFP4 fingerprint to a
128-dimensional latent representation using a two-layer MLP with batch
normalization, ReLU activations, and dropout:

```text
z_fp = f_fp(x_ecfp)
```

The `ECFP4_MLP` baseline predicts the downstream property from `z_fp`.

### 3.3 Descriptor Prediction Baseline

The `ECFP4_MLP_DescPred` baseline adds an auxiliary descriptor prediction head
to the ECFP encoder. The model predicts the downstream target from `z_fp` and
predicts standardized RDKit descriptors from the same representation.

The descriptor vector contains nine RDKit descriptors:

```text
MolWt, LogP, TPSA, HBA, HBD, RotBonds, AromaticRings, HeavyAtoms, RingCount
```

The training objective is:

```text
L = L_task + lambda_transfer * L_desc
```

where `lambda_transfer = 0.1`.

### 3.4 AdapterFusion Student

The AdapterFusion student uses the same ECFP encoder but introduces a
pseudo-descriptor adapter:

```text
z_desc = f_adapter(z_fp)
```

A learned fusion gate combines the original ECFP representation and the
pseudo-descriptor representation:

```text
g = sigmoid(f_gate([z_fp, z_desc]))
z_fused = g * z_fp + (1 - g) * z_desc
```

The target prediction head receives `z_fused`, while the descriptor prediction
head predicts descriptors from `z_desc`. This structure keeps inference
ECFP-only, because no true descriptors are consumed by the student at test
time.

### 3.5 Descriptor-Access Controls

We include descriptor-access controls to estimate the value of direct
descriptor information. `ECFP4_MLP_DescConcat` concatenates ECFP4 and
standardized descriptors as neural input. We also train random forest baselines
with ECFP4-only, descriptor-only, and ECFP4+descriptor inputs.

The `ECFP4_Desc_RF` baseline is used as the descriptor-access teacher for
distillation. This follows the broader teacher-student idea of knowledge
distillation, where a deployed student is trained to match predictions from a
stronger or richer teacher [@Hinton2015].

### 3.6 Descriptor-Teacher Distillation

The main method, Descriptor-Teacher AdapterFusion, keeps the AdapterFusion
student architecture and adds a teacher prediction loss. For regression, the
teacher loss is:

```text
L_teacher = MSE(y_student, y_teacher)
```

The final loss is:

```text
L = L_task + lambda_transfer * L_desc + lambda_distill * L_teacher
```

The main configuration uses:

```text
lambda_transfer = 0.1
lambda_distill = 1.0
```

This value is selected from the fixed lambda sweep over
`{0.01, 0.1, 0.3, 1.0}`. The model architecture is unchanged relative to
AdapterFusion; only the training objective is augmented with teacher
supervision.

## 4. Experiments

### 4.1 Datasets

The main regression benchmark includes five ADMET datasets:

- `caco2_wang`
- `lipophilicity_astrazeneca`
- `solubility_aqsoldb`
- `vdss_lombardo`
- `ppbr_az`

Supplementary classification experiments include:

- `bbb_martins`
- `hia_hou`
- `pgp_broccatelli`
- `bioavailability_ma`
- `herg`

The main claims are restricted to regression tasks.

### 4.2 Low-Resource Protocol

Each dataset is evaluated with training ratios of 10%, 20%, and 50%. Each
setting is run with seeds 42, 123, and 3407. All models use the same
dataset-ratio-seed splits. Descriptor scalers are fit only on the training
split to avoid leakage.

### 4.3 Baselines

We compare:

- `ECFP4_MLP`
- `ECFP4_MLP_DescPred`
- `ECFP4_MLP_DescAdapterFusion`
- `ECFP4_MLP_DescConcat`
- `ECFP4_RF`
- `Desc_RF`
- `ECFP4_Desc_RF`
- Descriptor-Teacher AdapterFusion

`ECFP4_MLP_DescConcat`, `Desc_RF`, and `ECFP4_Desc_RF` use descriptors at
inference and should be interpreted as descriptor-access controls rather than
ECFP-only students.

### 4.4 Metrics

Regression tasks use MAE and RMSE, with RMSE as the primary metric.
Classification tasks use ROC-AUC and PR-AUC, with ROC-AUC reported in the
supplementary classification table. All tables report mean plus/minus standard
deviation over three seeds.

### 4.5 Ablations

We run three ablation analyses:

- Fixed distillation lambda sweep:
  `lambda_distill in {0.01, 0.1, 0.3, 1.0}`.
- Validation-selected lambda analysis, where the lambda with lowest mean
  validation RMSE is selected per dataset-ratio setting without retraining.
- Adaptive teacher weighting using validation teacher advantage:

```text
lambda_effective = lambda_max * max((base_valid_rmse - teacher_valid_rmse) / base_valid_rmse, 0)
```

The adaptive rule is treated as a negative ablation.

## 5. Results

### 5.1 Main Regression Results

Table 1 is provided in `paper_tables/table1_regression_main_rmse.md`. The main
pattern is that descriptor-access RF models remain strong, confirming that
descriptor information is predictive for ADMET properties. However, among
ECFP-only neural models, AdapterFusion is consistently stronger than simple
descriptor prediction on regression tasks.

Before teacher distillation, AdapterFusion improves over DescPred in all 15
regression dataset-ratio settings. This supports the hypothesis that descriptor
knowledge is better transferred through a structured pseudo-descriptor adapter
and fusion mechanism than through a plain auxiliary descriptor head alone.

### 5.2 Descriptor-Teacher Distillation

Teacher distillation further improves AdapterFusion. With
`lambda_distill = 1.0`, distilled AdapterFusion improves over base AdapterFusion
in 11/15 regression settings. The mean RMSE delta versus base AdapterFusion is
-0.0656.

The gains are most visible in settings where the descriptor-access teacher has
a large advantage over the ECFP-only neural student. For example, `ppbr_az`
benefits from fixed strong distillation across all three training ratios.
Nevertheless, the distilled student remains below the descriptor-access RF
teacher in most settings. The result should therefore be interpreted as partial
descriptor knowledge transfer, not replacement of descriptor-access baselines.

### 5.3 Lambda Sensitivity

The lambda ablation is provided in `paper_tables/table2_lambda_ablation_rmse.md`
and summarized in `paper_tables/table3_lambda_summary.md`.

| lambda_distill | complete settings | beats AdapterFusion | mean delta vs AdapterFusion |
| --- | --- | --- | ---: |
| 0.01 | 15/15 | 8/15 | 0.0075 |
| 0.1 | 15/15 | 8/15 | -0.0221 |
| 0.3 | 15/15 | 10/15 | -0.0046 |
| 1.0 | 15/15 | 11/15 | -0.0656 |

The strongest global fixed value is `lambda_distill = 1.0`. Smaller lambdas can
be better for individual settings, but they do not improve aggregate
performance.

### 5.4 Validation-Selected Lambda

The validation-selected lambda analysis is shown in
`paper_tables/table6_validation_selected_lambda.md`. Selecting lambda by mean
validation RMSE beats fixed `lambda_distill = 1.0` in only 2/15 settings. It
beats base AdapterFusion in 12/15 settings.

This suggests that validation-selected lambda is useful as an analysis but does
not provide a stronger default than fixed `lambda_distill = 1.0` under the
current setup.

### 5.5 Adaptive Distillation Negative Ablation

The adaptive teacher-weighting result is shown in
`paper_tables/table4_adaptive_summary.md` and
`paper_tables/table5_adaptive_vs_fixed.md`. The adaptive rule improves over
base AdapterFusion in 9/15 settings, but beats fixed `lambda_distill = 1.0` in
only 3/15 settings and beats the best fixed lambda in only 1/15 settings. Its
mean RMSE delta relative to fixed `lambda_distill = 1.0` is 0.0817, indicating
worse average performance.

This negative result shows that raw validation teacher advantage is not a
sufficient reliability signal for adaptive distillation.

### 5.6 Classification Reference

Classification results are provided in
`paper_tables/tableS1_classification_main_roc_auc.md`. The results are less
stable than regression results, and the current teacher-distillation claim is
therefore limited to regression ADMET.

## 6. Discussion

The experiments support a controlled descriptor-guided transfer story.
Descriptor information is valuable, and descriptor-access baselines remain
strong. The contribution is not to replace descriptor-access RF, but to show
that descriptor-access knowledge can partially improve an ECFP-only neural
student under low-resource regression ADMET settings.

AdapterFusion improves over plain descriptor prediction, suggesting that the
structure of auxiliary transfer matters. Teacher distillation then provides an
additional improvement by aligning the ECFP-only student with predictions from
a descriptor-access RF teacher.

The lambda sweep shows that stronger teacher supervision is generally useful:
`lambda_distill = 1.0` is the best fixed global setting. However, the
dataset-specific best lambda varies, and neither validation-selected lambda nor
the tested adaptive rule clearly improves over the fixed setting. The adaptive
failure is informative: dataset-level validation teacher advantage can
underweight teacher supervision on settings where strong distillation is
beneficial.

Practically, these results recommend a cautious use of descriptor-teacher
distillation. It can improve an ECFP-only regression student, but descriptor
baselines should still be included and reported.

## 7. Limitations

This work has several limitations. First, the main claim is restricted to
regression ADMET tasks. Classification results are included only as
supplementary evidence because improvements are less consistent. Second, the
student model is lightweight and ECFP-based; the method has not yet been tested
with large pretrained molecular encoders. Third, descriptor-access RF remains
stronger than the distilled student in most settings. Finally, the adaptive
teacher-weighting strategy tested here is simple and negative; more principled
sample-level or learned teacher weighting may be needed.

## 8. Conclusion

We presented Descriptor-Teacher AdapterFusion, a lightweight descriptor-guided
multi-to-uni transfer method for low-resource regression ADMET prediction. The
method trains an ECFP-only AdapterFusion student with descriptor reconstruction
and teacher prediction distillation from an ECFP+descriptor random forest. The
main fixed setting, `lambda_distill = 1.0`, improves AdapterFusion in 11/15
regression dataset-ratio settings. At the same time, descriptor-access RF
baselines remain stronger, so the appropriate conclusion is controlled
descriptor knowledge transfer rather than overall ADMET state-of-the-art
performance.

## References

BibTeX entries are provided in `paper_notes/references.bib`. The current
Markdown draft uses citation keys in square brackets so it can be converted to
LaTeX or another venue-specific format later.

- [Breiman2001] Breiman, L. Random forests. Machine Learning 45, 5-32 (2001).
- [Butler2018] Butler, K. T., Davies, D. W., Cartwright, H., Isayev, O. &
  Walsh, A. Machine learning for molecular and materials science. Nature 559,
  547-555 (2018).
- [Chen2016XGBoost] Chen, T. & Guestrin, C. XGBoost: A scalable tree boosting
  system. In Proceedings of the 22nd ACM SIGKDD International Conference on
  Knowledge Discovery and Data Mining, 785-794 (2016).
- [Dey2024AuxAdapt] Dey, V. & Ning, X. Enhancing molecular property prediction
  with auxiliary learning and task-specific adaptation. Journal of
  Cheminformatics 16, 85 (2024).
- [Fabian2020MolBERT] Fabian, B. et al. Molecular representation learning with
  language models and domain-relevant auxiliary tasks. arXiv:2011.13230
  (2020).
- [Heid2024Chemprop] Heid, E. et al. Chemprop: A machine learning package for
  chemical property prediction. Journal of Chemical Information and Modeling
  64, 4613-4629 (2024).
- [Hinton2015] Hinton, G., Vinyals, O. & Dean, J. Distilling the knowledge in a
  neural network. arXiv:1503.02531 (2015).
- [Hu2020PretrainGNN] Hu, W. et al. Strategies for pre-training graph neural
  networks. International Conference on Learning Representations (2020).
- [Huang2021TDC] Huang, K. et al. Therapeutics Data Commons: Machine learning
  datasets and tasks for drug discovery and development. NeurIPS Datasets and
  Benchmarks (2021).
- [Kearnes2016GraphConv] Kearnes, S. et al. Molecular graph convolutions:
  Moving beyond fingerprints. Journal of Computer-Aided Molecular Design 30,
  595-608 (2016).
- [Ke2017LightGBM] Ke, G. et al. LightGBM: A highly efficient gradient boosting
  decision tree. Advances in Neural Information Processing Systems 30 (2017).
- [Liu2022GraphMVP] Liu, S. et al. Pre-training molecular graph representation
  with 3D geometry. International Conference on Learning Representations
  (2022).
- [RDKit2026] RDKit: Open-source cheminformatics. https://www.rdkit.org.
- [Rogers2010] Rogers, D. & Hahn, M. Extended-connectivity fingerprints.
  Journal of Chemical Information and Modeling 50, 742-754 (2010).
- [Rong2020GROVER] Rong, Y. et al. Self-supervised graph transformer on
  large-scale molecular data. Advances in Neural Information Processing Systems
  33 (2020).
- [Ruder2017] Ruder, S. An overview of multi-task learning in deep neural
  networks. arXiv:1706.05098 (2017).
- [Tian2022ADMETBoost] Tian, H., Ketkar, R. & Tao, P. Accurate ADMET prediction
  with XGBoost. arXiv:2204.07532 (2022).
- [Vamathevan2019] Vamathevan, J. et al. Applications of machine learning in
  drug discovery and development. Nature Reviews Drug Discovery 18, 463-477
  (2019).
- [Wu2018MoleculeNet] Wu, Z. et al. MoleculeNet: A benchmark for molecular
  machine learning. Chemical Science 9, 513-530 (2018).
- [Xiong2020AttentiveFP] Xiong, Z. et al. Pushing the boundaries of molecular
  representation for drug discovery with the graph attention mechanism. Journal
  of Medicinal Chemistry 63, 8749-8760 (2020).
- [Xiong2021ADMETlab] Xiong, G. et al. ADMETlab 2.0: An integrated online
  platform for accurate and comprehensive predictions of ADMET properties.
  Nucleic Acids Research 49, W5-W14 (2021).
- [Xiong2026M2UMol] Xiong, Z. et al. Multi-to-uni modal knowledge transfer
  pre-training for molecular representation learning. Nature Communications 17,
  3797 (2026).
- [Yang2019DMPNN] Yang, K. et al. Analyzing learned molecular representations
  for property prediction. Journal of Chemical Information and Modeling 59,
  3370-3388 (2019).
