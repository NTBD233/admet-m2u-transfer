# Method Draft

## Overview

We study descriptor-guided multi-to-uni transfer for low-resource ADMET
prediction. The goal is to use RDKit physicochemical descriptors as auxiliary
knowledge during training while preserving an ECFP-only neural student at
inference time. The main method is Descriptor-Teacher AdapterFusion: an
ECFP4-based AdapterFusion student trained with task supervision, descriptor
reconstruction supervision, and prediction distillation from a descriptor-access
random forest teacher.

The method is designed for the setting where descriptor information is useful
but the deployed neural model should remain lightweight and depend only on
ECFP4 fingerprints.

## ECFP Encoder

Each molecule is represented by a 2048-dimensional ECFP4 fingerprint. The base
encoder is a two-layer MLP with batch normalization, ReLU activations, and
dropout. It maps the fingerprint to a 128-dimensional latent representation:

```text
z_fp = f_fp(x_ecfp)
```

The baseline `ECFP4_MLP` predicts the downstream ADMET target directly from
`z_fp`.

## Descriptor Prediction Baseline

The `ECFP4_MLP_DescPred` baseline adds an auxiliary head that predicts
standardized RDKit descriptors from `z_fp`. The downstream prediction still
uses only `z_fp`.

The auxiliary descriptor loss is an MSE loss over nine descriptors:

```text
MolWt, LogP, TPSA, HBA, HBD, RotBonds, AromaticRings, HeavyAtoms, RingCount
```

The training loss is:

```text
L = L_task + lambda_transfer * L_desc
```

where `lambda_transfer = 0.1` in all main experiments.

## AdapterFusion Student

The `ECFP4_MLP_DescAdapterFusion` model keeps ECFP-only inference but changes
how descriptor knowledge is routed. Instead of only predicting descriptors from
the main fingerprint representation, it passes `z_fp` through a descriptor
adapter:

```text
z_desc = f_adapter(z_fp)
```

A learned gate fuses the original ECFP representation and the pseudo-descriptor
representation:

```text
g = sigmoid(f_gate([z_fp, z_desc]))
z_fused = g * z_fp + (1 - g) * z_desc
```

The downstream target is predicted from `z_fused`, while descriptors are
predicted from `z_desc`. This preserves ECFP-only inference, because the model
does not consume true descriptors at test time.

## Descriptor-Access Controls

`ECFP4_MLP_DescConcat` concatenates ECFP4 and standardized descriptors as input.
This model is not an ECFP-only inference model. It is included as a
descriptor-access neural control.

Traditional random forest baselines are included with ECFP4-only,
descriptor-only, and ECFP4+descriptor inputs. The `ECFP4_Desc_RF` model is used
as the teacher for distillation.

## Teacher Distillation

The Descriptor-Teacher AdapterFusion student uses the same AdapterFusion
architecture and adds a teacher prediction loss. For regression tasks, the
distillation loss is MSE between the student prediction and the
`ECFP4_Desc_RF` teacher prediction:

```text
L = L_task + lambda_transfer * L_desc + lambda_distill * L_teacher
```

The main paper setting uses:

```text
lambda_transfer = 0.1
lambda_distill = 1.0
```

This choice is based on the regression lambda sweep. The method does not alter
the model architecture relative to AdapterFusion; it changes only the training
objective by adding descriptor-access teacher supervision.

## Inference

At inference time, the main student models use only ECFP4 fingerprints. True
RDKit descriptors are used at inference only by descriptor-access controls such
as `DescConcat`, `Desc_RF`, and `ECFP4_Desc_RF`.
