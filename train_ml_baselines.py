import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import DATASETS, RESULTS_ROOT, SEEDS, TRAIN_RATIO_TAG
from utils.dataset import feature_paths
from utils.io import ensure_dir, save_json
from utils.metrics import compute_metrics, prediction_frame
from utils.seed import set_seed
from utils.summary import collect_metrics, save_summaries


def require_sklearn():
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "scikit-learn is required for ML baselines. "
            "Run `bash setup_env.sh` after updating environment.yml, or install scikit-learn "
            "inside the active tdc-admet environment."
        ) from exc
    return RandomForestClassifier, RandomForestRegressor


def optional_xgboost():
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ModuleNotFoundError:
        return None, None
    return XGBClassifier, XGBRegressor


def load_split(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return {
        "fp": data["X_fp"].astype(np.float32),
        "desc": data["X_desc"].astype(np.float32),
        "y": data["y"].astype(np.float32).reshape(-1),
        "smiles": data["smiles"],
    }


def build_features(split, feature_set):
    if feature_set == "ECFP4":
        return split["fp"]
    if feature_set == "Desc":
        return split["desc"]
    if feature_set == "ECFP4_Desc":
        return np.concatenate([split["fp"], split["desc"]], axis=1)
    raise ValueError(f"Unknown feature_set: {feature_set}")


def predict_raw(model, x, task_type):
    if task_type == "classification":
        prob = model.predict_proba(x)[:, 1]
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        return np.log(prob / (1 - prob))
    return model.predict(x)


def model_name_parts(model_name):
    parts = model_name.split("_")
    algorithm = parts[-1]
    feature_set = "_".join(parts[:-1])
    return feature_set, algorithm


def make_model(model_name, task_type, seed):
    RandomForestClassifier, RandomForestRegressor = require_sklearn()
    XGBClassifier, XGBRegressor = optional_xgboost()
    _, algorithm = model_name_parts(model_name)

    if algorithm == "RF":
        if task_type == "classification":
            return RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                n_jobs=-1,
                random_state=seed,
                class_weight="balanced",
            )
        return RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )

    if algorithm == "XGB":
        if XGBClassifier is None or XGBRegressor is None:
            raise SystemExit(
                "xgboost is required for XGB baselines. Install it or omit XGB model names."
            )
        if task_type == "classification":
            return XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            )
        return XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown baseline algorithm in model_name: {model_name}")


def train_one_baseline(
    dataset_name,
    task_type,
    model_name,
    seed,
    train_ratio_tag,
    results_root,
    skip_existing=False,
):
    set_seed(seed)
    feature_set, _ = model_name_parts(model_name)
    save_dir = (
        Path(results_root)
        / dataset_name
        / model_name
        / f"train_{train_ratio_tag}"
        / f"seed_{seed}"
    )
    if skip_existing and (save_dir / "metrics.json").exists():
        print(f"Skipping existing baseline: {save_dir}")
        return None

    train_path, valid_path, test_path = feature_paths(
        dataset_name=dataset_name,
        train_ratio_tag=train_ratio_tag,
    )

    train = load_split(train_path)
    valid = load_split(valid_path)
    test = load_split(test_path)

    x_train = build_features(train, feature_set)
    x_valid = build_features(valid, feature_set)
    x_test = build_features(test, feature_set)

    model = make_model(model_name, task_type, seed)
    model.fit(x_train, train["y"])

    valid_raw = predict_raw(model, x_valid, task_type)
    test_raw = predict_raw(model, x_test, task_type)
    valid_metrics, valid_pred = compute_metrics(valid["y"], valid_raw, task_type)
    test_metrics, test_pred = compute_metrics(test["y"], test_raw, task_type)

    valid_pred_df = prediction_frame(valid["smiles"], valid["y"], valid_pred, valid_raw)
    test_pred_df = prediction_frame(test["smiles"], test["y"], test_pred, test_raw)

    metrics = {
        "dataset": dataset_name,
        "task_type": task_type,
        "model": model_name,
        "train_ratio_tag": train_ratio_tag,
        "seed": seed,
        "lambda_transfer": np.nan,
        "best_epoch": np.nan,
        "best_metric": np.nan,
        "valid_roc_auc": np.nan,
        "valid_pr_auc": np.nan,
        "test_roc_auc": np.nan,
        "test_pr_auc": np.nan,
        "valid_mae": np.nan,
        "valid_rmse": np.nan,
        "test_mae": np.nan,
        "test_rmse": np.nan,
    }
    metrics.update({f"valid_{k}": float(v) for k, v in valid_metrics.items()})
    metrics.update({f"test_{k}": float(v) for k, v in test_metrics.items()})

    ensure_dir(save_dir)
    with (save_dir / "best_model.pkl").open("wb") as f:
        pickle.dump(model, f)
    pd.DataFrame([metrics]).to_csv(save_dir / "training_history.csv", index=False)
    valid_pred_df.to_csv(save_dir / "valid_predictions.csv", index=False)
    test_pred_df.to_csv(save_dir / "test_predictions.csv", index=False)
    save_json(metrics, save_dir / "metrics.json")
    print(f"Saved ML baseline outputs to: {save_dir}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train traditional ML ADMET baselines.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ECFP4_RF", "Desc_RF", "ECFP4_Desc_RF"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--train-ratio-tags", nargs="+", type=int, default=[TRAIN_RATIO_TAG])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    return parser.parse_args()


def main():
    args = parse_args()
    selected_datasets = {name: DATASETS[name] for name in args.datasets}

    for dataset_name, cfg in selected_datasets.items():
        for train_ratio_tag in args.train_ratio_tags:
            for model_name in args.models:
                for seed in args.seeds:
                    train_one_baseline(
                        dataset_name=dataset_name,
                        task_type=cfg["task_type"],
                        model_name=model_name,
                        seed=seed,
                        train_ratio_tag=train_ratio_tag,
                        results_root=args.results_root,
                        skip_existing=args.skip_existing,
                    )

    metrics_df = collect_metrics(args.results_root)
    paths = save_summaries(metrics_df, args.results_root)
    print("Summary files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
