# Experiments Draft

## Benchmark

Experiments use ADMET datasets prepared in the TDC-style benchmark workflow.
The main regression evaluation covers five datasets:

- `caco2_wang`
- `lipophilicity_astrazeneca`
- `solubility_aqsoldb`
- `vdss_lombardo`
- `ppbr_az`

Classification datasets are included as supplementary reference experiments:

- `bbb_martins`
- `hia_hou`
- `pgp_broccatelli`
- `bioavailability_ma`
- `herg`

The main paper claim is restricted to regression ADMET because teacher
distillation showed the most consistent signal there.

## Low-Resource Splits

Each dataset is evaluated under low-resource training ratios:

```text
10%, 20%, 50%
```

Each setting is run with three random seeds:

```text
42, 123, 3407
```

All models use the same dataset, train ratio, and seed splits for fair
comparison.

## Molecular Features

The ECFP representation is a 2048-dimensional ECFP4 fingerprint. Descriptor
features are nine standardized RDKit descriptors:

```text
MolWt, LogP, TPSA, HBA, HBD, RotBonds, AromaticRings, HeavyAtoms, RingCount
```

Descriptor scalers are fit only on the training split to avoid validation or
test leakage.

## Models and Baselines

The neural models are:

- `ECFP4_MLP`: ECFP-only MLP baseline.
- `ECFP4_MLP_DescPred`: ECFP-only MLP with auxiliary descriptor prediction.
- `ECFP4_MLP_DescAdapterFusion`: ECFP-only AdapterFusion student.
- `ECFP4_MLP_DescConcat`: descriptor-access neural control.
- Distilled AdapterFusion: AdapterFusion trained with `ECFP4_Desc_RF` teacher
  predictions.

The traditional baselines are:

- `ECFP4_RF`
- `Desc_RF`
- `ECFP4_Desc_RF`

`ECFP4_Desc_RF` is also used as the teacher in distillation experiments.

## Metrics

Regression tasks use MAE and RMSE, with RMSE as the primary metric in the main
tables. Lower values are better.

Classification tasks use ROC-AUC and PR-AUC, with ROC-AUC used in the
supplementary classification table. Higher values are better.

All reported values are mean plus/minus standard deviation over three seeds.

## Distillation Experiments

The main distillation experiment trains `ECFP4_MLP_DescAdapterFusion` with the
`ECFP4_Desc_RF` teacher. The primary configuration is:

```text
lambda_transfer = 0.1
lambda_distill = 1.0
```

A fixed lambda sweep evaluates:

```text
lambda_distill in {0.01, 0.1, 0.3, 1.0}
```

An additional validation-selected lambda analysis chooses the lambda with the
lowest mean validation RMSE for each dataset-ratio setting. This analysis uses
only existing trained models and does not retrain any model.

## Adaptive Negative Ablation

A simple adaptive teacher-weighting rule was tested:

```text
lambda_effective = lambda_max * max((base_valid_rmse - teacher_valid_rmse) / base_valid_rmse, 0)
```

with `lambda_max = 1.0`. This rule is reported as a negative ablation because
it underperforms fixed `lambda_distill = 1.0`.
