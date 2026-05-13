from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREPARED_DATA_ROOT = PROJECT_ROOT / "data" / "prepared_data_subsample"
FEATURE_ROOT = PROJECT_ROOT / "data" / "features_m2u"
RESULTS_ROOT = PROJECT_ROOT / "results"

SEEDS = [42, 123, 3407]

BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 30
LR = 5e-4
WEIGHT_DECAY = 1e-5
LAMBDA_TRANSFER = 0.1
TRAIN_RATIO_TAG = 50
TRAIN_RATIO_TAGS = [10, 20, 50]

DATASETS = {
    "bbb_martins": {
        "task_type": "classification",
        "higher_is_better": True,
        "main_metric": "roc_auc",
    },
    "caco2_wang": {
        "task_type": "regression",
        "higher_is_better": False,
        "main_metric": "rmse",
    },
    "lipophilicity_astrazeneca": {
        "task_type": "regression",
        "higher_is_better": False,
        "main_metric": "rmse",
    },
    "solubility_aqsoldb": {
        "task_type": "regression",
        "higher_is_better": False,
        "main_metric": "rmse",
    },
    "vdss_lombardo": {
        "task_type": "regression",
        "higher_is_better": False,
        "main_metric": "rmse",
    },
    "ppbr_az": {
        "task_type": "regression",
        "higher_is_better": False,
        "main_metric": "rmse",
    },
    "hia_hou": {
        "task_type": "classification",
        "higher_is_better": True,
        "main_metric": "roc_auc",
    },
    "pgp_broccatelli": {
        "task_type": "classification",
        "higher_is_better": True,
        "main_metric": "roc_auc",
    },
    "bioavailability_ma": {
        "task_type": "classification",
        "higher_is_better": True,
        "main_metric": "roc_auc",
    },
    "herg": {
        "task_type": "classification",
        "higher_is_better": True,
        "main_metric": "roc_auc",
    },
}

MODELS = [
    "ECFP4_MLP",
    "ECFP4_MLP_DescPred",
    "ECFP4_MLP_DescConcat",
    "ECFP4_MLP_DescAdapterFusion",
]

DESC_COLS = [
    "MolWt",
    "LogP",
    "TPSA",
    "HBA",
    "HBD",
    "RotBonds",
    "AromaticRings",
    "HeavyAtoms",
    "RingCount",
]

FP_SIZE = 2048
FP_RADIUS = 2
