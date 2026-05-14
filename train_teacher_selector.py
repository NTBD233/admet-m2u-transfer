import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_gate_targets import build_prior_weight_map
from analyze_teacher_reliability import descriptor_ood_distances, load_prediction
from generate_teacher_predictions import predict_raw_with_uncertainty
from train_ml_baselines import build_features, load_split, make_model, model_name_parts
from utils.config import DATASETS, FEATURE_ROOT, PROJECT_ROOT, SEEDS, TRAIN_RATIO_TAGS
from utils.dataset import feature_paths


DEFAULT_TEACHERS = ["ECFP4_RF", "Desc_RF", "ECFP4_Desc_RF"]


def require_sklearn():
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import KFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise SystemExit("scikit-learn is required for selector training.") from exc
    return {
        "ColumnTransformer": ColumnTransformer,
        "RandomForestClassifier": RandomForestClassifier,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "KFold": KFold,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }


def compute_train_ood_distances(desc_train, n_neighbors=5):
    try:
        from sklearn.neighbors import NearestNeighbors
    except ModuleNotFoundError:
        return np.full(len(desc_train), np.nan, dtype=float)
    if len(desc_train) <= 1:
        return np.zeros(len(desc_train), dtype=float)
    k = min(n_neighbors + 1, len(desc_train))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(desc_train)
    distances, _ = nn.kneighbors(desc_train)
    if distances.shape[1] > 1:
        distances = distances[:, 1:]
    return distances.mean(axis=1)


def teacher_feature_matrix(records, teachers, ood_distance):
    preds = np.vstack([records[teacher]["pred"] for teacher in teachers]).T
    uncertainties = np.vstack([records[teacher]["uncertainty"] for teacher in teachers]).T
    consensus = preds.mean(axis=1, keepdims=True)
    rows = []
    for teacher_idx, teacher in enumerate(teachers):
        rows.append(uncertainties[:, teacher_idx])
        rows.append(np.abs(preds[:, teacher_idx] - consensus[:, 0]))
        rows.append(records[teacher]["prior_weight"])
    rows.append(ood_distance)
    features = np.column_stack(rows)
    feature_names = []
    for teacher in teachers:
        feature_names.extend(
            [
                f"{teacher}__uncertainty",
                f"{teacher}__consensus_gap",
                f"{teacher}__prior_weight",
            ]
        )
    feature_names.append("ood_distance")
    return features.astype(np.float32), feature_names


def oracle_labels_from_records(records, teachers):
    y = records[teachers[0]]["y"]
    preds = np.vstack([records[teacher]["pred"] for teacher in teachers]).T
    abs_errors = np.abs(preds - y.reshape(-1, 1))
    label_idx = np.argmin(abs_errors, axis=1).astype(int)
    return label_idx, abs_errors


def build_crossfit_train_records(dataset_name, task_type, teachers, train_ratio_tag, seed, n_folds):
    sklearn = require_sklearn()
    kfold = sklearn["KFold"](n_splits=n_folds, shuffle=True, random_state=seed)
    train_path, _, _ = feature_paths(dataset_name=dataset_name, feature_root=FEATURE_ROOT, train_ratio_tag=train_ratio_tag)
    train_split = load_split(train_path)
    y = train_split["y"]
    smiles = train_split["smiles"]
    desc_train = train_split["desc"]
    ood_distance = compute_train_ood_distances(desc_train)
    prior_weights = build_prior_weight_map(
        teacher_root=PROJECT_ROOT / "data" / "teacher_predictions",
        dataset=dataset_name,
        teachers=teachers,
        ratio=train_ratio_tag,
        seed=seed,
    )

    records = {}
    for teacher in teachers:
        feature_set, _ = model_name_parts(teacher)
        x_all = build_features(train_split, feature_set)
        pred = np.zeros(len(y), dtype=np.float32)
        uncertainty = np.full(len(y), np.nan, dtype=np.float32)
        for fold_idx, (fit_idx, hold_idx) in enumerate(kfold.split(x_all)):
            model = make_model(teacher, task_type, seed + fold_idx + 1)
            model.fit(x_all[fit_idx], y[fit_idx])
            pred_raw, fold_uncertainty = predict_raw_with_uncertainty(model, x_all[hold_idx], task_type)
            pred[hold_idx] = pred_raw.astype(np.float32)
            uncertainty[hold_idx] = fold_uncertainty.astype(np.float32)
        records[teacher] = {
            "pred": pred,
            "uncertainty": uncertainty,
            "y": y.astype(np.float32),
            "smiles": smiles,
            "prior_weight": np.full(len(y), prior_weights[teacher], dtype=np.float32),
        }
    return records, ood_distance


def build_eval_records(results_root, teacher_root, dataset_name, teachers, train_ratio_tag, seed, split_name):
    loaded = {}
    for teacher in teachers:
        record, missing = load_prediction(
            results_root=results_root,
            teacher_root=teacher_root,
            dataset=dataset_name,
            teacher=teacher,
            ratio=train_ratio_tag,
            seed=seed,
            split=split_name,
        )
        if record is None:
            raise FileNotFoundError(f"Missing {split_name} teacher prediction for {teacher}: {missing}")
        loaded[teacher] = record
    ood_distance = descriptor_ood_distances(dataset_name, train_ratio_tag, split_name)
    prior_weights = build_prior_weight_map(
        teacher_root=teacher_root,
        dataset=dataset_name,
        teachers=teachers,
        ratio=train_ratio_tag,
        seed=seed,
    )
    for teacher in teachers:
        loaded[teacher]["prior_weight"] = np.full(len(loaded[teacher]["y"]), prior_weights[teacher], dtype=np.float32)
    return loaded, ood_distance


def build_split_examples(records, teachers, ood_distance):
    features, feature_names = teacher_feature_matrix(records, teachers, ood_distance)
    labels, abs_errors = oracle_labels_from_records(records, teachers)
    return {
        "features": features,
        "feature_names": feature_names,
        "labels": labels,
        "abs_errors": abs_errors,
        "y": records[teachers[0]]["y"].astype(np.float32),
        "smiles": records[teachers[0]]["smiles"],
    }


def build_selector_model(selector_model_name, feature_names):
    sklearn = require_sklearn()
    ColumnTransformer = sklearn["ColumnTransformer"]
    SimpleImputer = sklearn["SimpleImputer"]
    StandardScaler = sklearn["StandardScaler"]
    Pipeline = sklearn["Pipeline"]
    LogisticRegression = sklearn["LogisticRegression"]
    RandomForestClassifier = sklearn["RandomForestClassifier"]

    if selector_model_name == "logistic":
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    feature_names,
                )
            ]
        )
        return Pipeline(
            [
                ("prep", preprocessor),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
    if selector_model_name == "rf":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), feature_names),
            ]
        )
        return Pipeline(
            [
                ("prep", preprocessor),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        n_jobs=-1,
                        max_depth=8,
                        min_samples_leaf=5,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown selector model: {selector_model_name}")


def split_metrics(y_true, y_pred, sklearn):
    return {
        "accuracy": float(sklearn["accuracy_score"](y_true, y_pred)),
        "macro_f1": float(sklearn["f1_score"](y_true, y_pred, average="macro", zero_division=0)),
    }


def save_split_predictions(out_dir, split_name, split_data, probs, teachers):
    pred_idx = probs.argmax(axis=1).astype(np.int64)
    np.savez_compressed(
        out_dir / f"{split_name}_selector_predictions.npz",
        probs=probs.astype(np.float32),
        pred_idx=pred_idx,
        oracle_idx=split_data["labels"].astype(np.int64),
        smiles=split_data["smiles"],
        y=split_data["y"].astype(np.float32),
        teachers=np.array(teachers, dtype=object),
        feature_names=np.array(split_data["feature_names"], dtype=object),
    )


def train_selector_for_setting(
    dataset_name,
    task_type,
    teachers,
    train_ratio_tag,
    seed,
    selector_model_name,
    label_source,
    results_root,
    teacher_root,
    output_root,
    n_folds,
):
    out_dir = Path(output_root) / dataset_name / f"{selector_model_name}_{label_source}" / f"train_{train_ratio_tag}" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records, train_ood = build_crossfit_train_records(
        dataset_name=dataset_name,
        task_type=task_type,
        teachers=teachers,
        train_ratio_tag=train_ratio_tag,
        seed=seed,
        n_folds=n_folds,
    )
    train_split = build_split_examples(train_records, teachers, train_ood)

    valid_records, valid_ood = build_eval_records(
        results_root=results_root,
        teacher_root=teacher_root,
        dataset_name=dataset_name,
        teachers=teachers,
        train_ratio_tag=train_ratio_tag,
        seed=seed,
        split_name="valid",
    )
    test_records, test_ood = build_eval_records(
        results_root=results_root,
        teacher_root=teacher_root,
        dataset_name=dataset_name,
        teachers=teachers,
        train_ratio_tag=train_ratio_tag,
        seed=seed,
        split_name="test",
    )
    valid_split = build_split_examples(valid_records, teachers, valid_ood)
    test_split = build_split_examples(test_records, teachers, test_ood)

    selector = build_selector_model(selector_model_name, train_split["feature_names"])
    selector.fit(pd.DataFrame(train_split["features"], columns=train_split["feature_names"]), train_split["labels"])

    sklearn = require_sklearn()
    metrics = {
        "dataset": dataset_name,
        "task_type": task_type,
        "train_ratio_tag": train_ratio_tag,
        "seed": seed,
        "teachers": teachers,
        "selector_model": selector_model_name,
        "label_source": label_source,
        "n_folds": n_folds,
    }

    for split_name, split_data in [("train", train_split), ("valid", valid_split), ("test", test_split)]:
        x_df = pd.DataFrame(split_data["features"], columns=split_data["feature_names"])
        probs = selector.predict_proba(x_df)
        pred_idx = probs.argmax(axis=1)
        split_stat = split_metrics(split_data["labels"], pred_idx, sklearn)
        metrics[f"{split_name}_accuracy"] = split_stat["accuracy"]
        metrics[f"{split_name}_macro_f1"] = split_stat["macro_f1"]
        save_split_predictions(out_dir, split_name, split_data, probs, teachers)

    with (out_dir / "selector_model.pkl").open("wb") as f:
        pickle.dump(selector, f)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(out_dir / "selector_metrics.csv", index=False)
    print(f"Saved selector outputs to: {out_dir}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train pretrained teacher selectors for ADMET distillation.")
    parser.add_argument("--datasets", nargs="+", default=[name for name, cfg in DATASETS.items() if cfg["task_type"] == "regression"])
    parser.add_argument("--teachers", nargs="+", default=DEFAULT_TEACHERS)
    parser.add_argument("--train-ratio-tags", nargs="+", type=int, default=TRAIN_RATIO_TAGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--selector-models", nargs="+", default=["rf", "logistic"])
    parser.add_argument("--label-source", default="crossfit_train_pseudo_oracle")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--teacher-root", default=str(PROJECT_ROOT / "data" / "teacher_predictions"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "data" / "selector_predictions"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.label_source != "crossfit_train_pseudo_oracle":
        raise ValueError("Only crossfit_train_pseudo_oracle is implemented as the formal selector label source")
    for dataset_name in args.datasets:
        task_type = DATASETS[dataset_name]["task_type"]
        for train_ratio_tag in args.train_ratio_tags:
            for seed in args.seeds:
                for selector_model_name in args.selector_models:
                    metrics_path = (
                        Path(args.output_root)
                        / dataset_name
                        / f"{selector_model_name}_{args.label_source}"
                        / f"train_{train_ratio_tag}"
                        / f"seed_{seed}"
                        / "metrics.json"
                    )
                    if args.skip_existing and metrics_path.exists():
                        print(f"Skipping existing selector: {metrics_path.parent}")
                        continue
                    train_selector_for_setting(
                        dataset_name=dataset_name,
                        task_type=task_type,
                        teachers=args.teachers,
                        train_ratio_tag=train_ratio_tag,
                        seed=seed,
                        selector_model_name=selector_model_name,
                        label_source=args.label_source,
                        results_root=args.results_root,
                        teacher_root=args.teacher_root,
                        output_root=args.output_root,
                        n_folds=args.n_folds,
                    )


if __name__ == "__main__":
    main()
