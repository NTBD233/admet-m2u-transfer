# Lightweight M2U ADMET Project

This project is a script-based organization of the stable corrected lightweight
M2U experiment from `tdc prac.ipynb`.

The original experimental logic is preserved:

- ECFP4 fingerprint is the main molecular input.
- RDKit physicochemical descriptors are auxiliary training knowledge.
- Inference uses only ECFP4.
- Best checkpoint is selected by validation ROC-AUC for classification and
  validation RMSE for regression.
- `ECFP4_MLP_DescConcat` is included as a control that directly uses RDKit
  descriptors at inference time.
- `ECFP4_MLP_DescAdapterFusion` is a lightweight multi-to-uni variant that
  predicts pseudo descriptor knowledge from ECFP4 and fuses it back into the
  target predictor while still using only ECFP4 at inference time.

## Structure

```text
data/
models/
train.py
evaluate.py
utils/
results/
```

## Expected Data

Place the notebook's prepared low-resource splits here:

```text
data/prepared_data_subsample/bbb_martins/train_50.csv
data/prepared_data_subsample/bbb_martins/valid.csv
data/prepared_data_subsample/bbb_martins/test.csv
data/prepared_data_subsample/caco2_wang/train_50.csv
data/prepared_data_subsample/caco2_wang/valid.csv
data/prepared_data_subsample/caco2_wang/test.csv
```

The research configuration now includes these ADMET datasets:

- classification: `bbb_martins`, `hia_hou`, `pgp_broccatelli`,
  `bioavailability_ma`, `herg`
- regression: `caco2_wang`, `lipophilicity_astrazeneca`,
  `solubility_aqsoldb`, `vdss_lombardo`, `ppbr_az`

Each CSV should keep the original notebook columns:

- `Drug`
- `Y`

## Environment

Recommended environment:

```bash
bash setup_env.sh
source .miniforge/etc/profile.d/conda.sh
conda activate tdc-admet
```

This uses Python 3.11 to match the original notebook kernel more closely.

Chemprop is optional for the organized lightweight M2U scripts. To install it
too:

```bash
INSTALL_CHEMPROP=1 bash setup_env.sh
```

## Generate Features

First prepare the CSV splits:

```bash
python prepare_data.py
```

Then generate model features:

```bash
python train.py --generate-features --features-only --train-ratio-tags 10 20 50
```

Equivalent Make targets are available after activating the environment:

```bash
make prepare
make smoke
make train
make train-low-resource
make ml-baselines
make analysis
make paper-tables
make summary
```

This writes `.npz` feature files under `data/features_m2u/`.

## Train

```bash
python train.py
```

For every dataset, model, and seed, outputs are saved under:

```text
results/{dataset}/{model}/seed_{seed}/
```

Each seed directory contains:

- `best_model.pt`
- `training_history.csv`
- `valid_predictions.csv`
- `test_predictions.csv`
- `metrics.json`

## Summarize Existing Runs

```bash
python evaluate.py
```

The summary keeps metrics at four decimals and writes:

- `results/summary/all_seed_metrics.csv`
- `results/summary/mean_std_summary.csv`
- `results/summary/mean_std_summary.json`
- `results/summary/main_table.csv`
- `results/summary/main_table.md`
- `results/summary/low_resource_table.csv`

## Traditional ML Baselines

After installing the updated environment, run:

```bash
python train_ml_baselines.py --train-ratio-tags 10 20 50
```

Long runs can be resumed safely:

```bash
python train.py --train-ratio-tags 10 20 50 --skip-existing
python train_ml_baselines.py --train-ratio-tags 10 20 50 --skip-existing
```

Check completion:

```bash
python experiment_status.py
```

Write an interim research brief from complete three-seed groups:

```bash
python write_research_brief.py
```

Default baselines are:

- `ECFP4_RF`
- `Desc_RF`
- `ECFP4_Desc_RF`

XGBoost baselines are available when `xgboost` is installed, for example:

```bash
python train_ml_baselines.py --models ECFP4_XGB Desc_XGB ECFP4_Desc_XGB
```

## Mechanism Analysis

Run:

```bash
python analyze_adapter_fusion.py
```

This writes:

- `results/summary/adapter_fusion_analysis.csv`
- `results/summary/descriptor_error_correlation.csv`

Paper notes are kept under `paper_notes/`.

## Teacher Distillation Prototype

Generate teacher predictions from a trained descriptor-access RF model:

```bash
python generate_teacher_predictions.py \
  --datasets caco2_wang \
  --teacher-model ECFP4_Desc_RF \
  --train-ratio-tags 10 20 50 \
  --skip-existing
```

Train an ECFP-only AdapterFusion student with teacher distillation:

```bash
python train.py \
  --datasets caco2_wang \
  --models ECFP4_MLP_DescAdapterFusion \
  --train-ratio-tags 10 20 50 \
  --teacher-root data/teacher_predictions \
  --teacher-model ECFP4_Desc_RF \
  --lambda-distill 0.1 \
  --results-root results_distill_caco2 \
  --skip-existing
```
