# Discussion Draft

## Main Interpretation

The experiments support a controlled descriptor-guided transfer story rather
than a broad state-of-the-art claim. RDKit descriptors provide useful molecular
knowledge for low-resource ADMET prediction, and descriptor-access baselines
are strong. The main question is whether part of this descriptor knowledge can
be transferred into an ECFP-only neural model.

AdapterFusion improves over simple descriptor prediction in regression tasks,
which suggests that the mechanism of transfer matters. Predicting descriptors
as an auxiliary task is not always sufficient. Routing the ECFP representation
through a pseudo-descriptor adapter and fusing it with the original ECFP
representation provides a more effective transfer path.

## Role of the Teacher

Teacher distillation from `ECFP4_Desc_RF` improves AdapterFusion in most
regression settings. This supports the hypothesis that a descriptor-access
teacher can guide an ECFP-only student toward descriptor-informed predictions.

The improvement is partial. The distilled student still does not match the
descriptor-access RF teacher in most settings. This is expected because the
student has access only to ECFP fingerprints at inference time, while the
teacher uses both ECFP and descriptors.

## Why Fixed Lambda Is Kept

The fixed lambda sweep shows that `lambda_distill = 1.0` is the strongest
global setting. Although individual dataset-ratio settings sometimes prefer a
smaller lambda, neither validation-selected lambda nor the tested adaptive
weighting strategy clearly improves over fixed `lambda_distill = 1.0`.

This makes fixed `lambda_distill = 1.0` the most defensible current default. It
is simple, reproducible, and empirically strongest on aggregate.

## Negative Adaptive Result

The validation-advantage adaptive rule is a useful negative ablation. It tests
a natural idea: increase teacher supervision only when the teacher has a
validation advantage over the base student. The result is weaker than fixed
distillation.

The failure mode is informative. Dataset-level validation teacher advantage is
too crude as a reliability signal and can suppress teacher guidance on settings
where strong distillation helps. Future adaptive methods should likely use
sample-level uncertainty, teacher-student disagreement, or a learned weighting
mechanism rather than a simple dataset-level ratio.

## Classification Limitation

Classification results are less consistent than regression results. This may be
because classification ADMET tasks have different label noise, class imbalance,
and thresholding behavior. The current teacher-distillation claim should
therefore be limited to regression ADMET. Classification can remain a
supplementary analysis.

## Practical Implication

For practical low-resource regression ADMET, descriptor-access RF remains a
strong baseline and should not be ignored. The proposed ECFP-only distilled
AdapterFusion student is useful when inference should remain lightweight or
descriptor-free, but the paper should not frame it as replacing descriptor
baselines.

## Limitations and Future Work

The current work is limited to ADMET prediction and does not address
retrosynthesis. The method is also evaluated with lightweight ECFP-based neural
models rather than large pretrained molecular encoders. Future work should test
whether descriptor-teacher distillation improves stronger molecular encoders
and whether more principled adaptive teacher weighting can outperform fixed
distillation.
