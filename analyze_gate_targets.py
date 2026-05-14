import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyze_teacher_reliability import (
    REGRESSION_DATASETS,
    descriptor_ood_distances,
    load_prediction,
)
from train import load_teacher_valid_metrics
from utils.config import PROJECT_ROOT, RESULTS_ROOT, SEEDS, TRAIN_RATIO_TAGS
from utils.summary import dataframe_to_markdown


DEFAULT_TEACHERS = ["ECFP4_RF", "Desc_RF", "ECFP4_Desc_RF"]


def build_prior_weight_map(teacher_root, dataset, teachers, ratio, seed):
    valid_metrics = load_teacher_valid_metrics(
        teacher_root=teacher_root,
        dataset_name=dataset,
        teacher_models=teachers,
        train_ratio_tag=ratio,
        seed=seed,
        metric="rmse",
    )
    scores = np.asarray([-valid_metrics[name] for name in teachers], dtype=np.float32)
    shifted = scores - scores.max()
    weights = np.exp(shifted)
    weights = weights / weights.sum()
    return {teacher: float(weight) for teacher, weight in zip(teachers, weights)}


def build_gate_target_rows(results_root, teacher_root, datasets, teachers, ratios, seeds):
    rows = []
    for dataset in datasets:
        for ratio in ratios:
            try:
                ood = descriptor_ood_distances(dataset, ratio, "valid")
            except Exception:
                ood = None
            for seed in seeds:
                loaded = {}
                for teacher in teachers:
                    record, _ = load_prediction(
                        results_root=results_root,
                        teacher_root=teacher_root,
                        dataset=dataset,
                        teacher=teacher,
                        ratio=ratio,
                        seed=seed,
                        split="valid",
                    )
                    if record is None:
                        loaded = {}
                        break
                    loaded[teacher] = record
                if len(loaded) != len(teachers):
                    continue

                n = len(next(iter(loaded.values()))["y"])
                y = next(iter(loaded.values()))["y"]
                smiles = next(iter(loaded.values()))["smiles"]
                preds = np.vstack([loaded[teacher]["pred"] for teacher in teachers]).T
                uncertainties = np.vstack([loaded[teacher]["uncertainty"] for teacher in teachers]).T
                consensus = preds.mean(axis=1, keepdims=True)
                abs_errors = np.abs(preds - y.reshape(-1, 1))
                winner_idx = np.argmin(abs_errors, axis=1)
                priors = build_prior_weight_map(teacher_root, dataset, teachers, ratio, seed)
                prior_vector = np.asarray([priors[teacher] for teacher in teachers], dtype=float)

                for i in range(n):
                    row = {
                        "dataset": dataset,
                        "train_ratio_tag": ratio,
                        "seed": seed,
                        "group_id": f"{dataset}|{ratio}|{seed}",
                        "sample_idx": i,
                        "smiles": smiles[i],
                        "y_true": float(y[i]),
                        "oracle_teacher_idx": int(winner_idx[i]),
                        "oracle_teacher": teachers[int(winner_idx[i])],
                        "ood_distance": float(ood[i]) if ood is not None and len(ood) == n else np.nan,
                        "setting_prior_teacher": teachers[int(np.argmax(prior_vector))],
                    }
                    for teacher_idx, teacher in enumerate(teachers):
                        row[f"{teacher}__pred"] = float(preds[i, teacher_idx])
                        row[f"{teacher}__uncertainty"] = float(uncertainties[i, teacher_idx])
                        row[f"{teacher}__consensus_gap"] = float(abs(preds[i, teacher_idx] - consensus[i, 0]))
                        row[f"{teacher}__abs_error"] = float(abs_errors[i, teacher_idx])
                        row[f"{teacher}__prior_weight"] = float(prior_vector[teacher_idx])
                    rows.append(row)
    return pd.DataFrame(rows)


def build_feature_matrix(df, teachers):
    feature_cols = ["ood_distance"]
    for teacher in teachers:
        feature_cols.extend(
            [
                f"{teacher}__uncertainty",
                f"{teacher}__consensus_gap",
                f"{teacher}__prior_weight",
            ]
        )
    return df[feature_cols].copy(), feature_cols


def evaluate_grouped_models(df, teachers):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    x, feature_cols = build_feature_matrix(df, teachers)
    y = df["oracle_teacher_idx"].to_numpy(dtype=int)
    groups = df["group_id"].to_numpy()

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
                feature_cols,
            )
        ]
    )

    model_specs = {
        "majority_train": None,
        "setting_prior_top1": "setting_prior",
        "logistic_gate_probe": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                    ),
                ),
            ]
        ),
        "rf_gate_probe": Pipeline(
            [
                ("prep", ColumnTransformer(
                    transformers=[("num", SimpleImputer(strategy="median"), feature_cols)]
                )),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=80,
                        random_state=42,
                        n_jobs=-1,
                        max_depth=8,
                        min_samples_leaf=5,
                    ),
                ),
            ]
        ),
    }

    fold_rows = []
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(x, y, groups):
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        group_id = df.iloc[test_idx[0]]["group_id"]

        majority_label = int(pd.Series(y_train).value_counts().idxmax())
        setting_prior_labels = []
        for teacher_name in df.iloc[test_idx]["setting_prior_teacher"]:
            setting_prior_labels.append(teachers.index(teacher_name))
        setting_prior_labels = np.asarray(setting_prior_labels, dtype=int)

        for model_name, model in model_specs.items():
            if model_name == "majority_train":
                pred = np.full_like(y_test, majority_label)
            elif model_name == "setting_prior_top1":
                pred = setting_prior_labels
            else:
                model.fit(x_train, y_train)
                pred = model.predict(x_test)
            fold_rows.append(
                {
                    "model": model_name,
                    "group_id": group_id,
                    "n_samples": len(y_test),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = (
        fold_df.groupby("model", as_index=False)
        .agg(
            groups=("group_id", "nunique"),
            total_samples=("n_samples", "sum"),
            mean_group_accuracy=("accuracy", "mean"),
            mean_group_macro_f1=("macro_f1", "mean"),
            weighted_accuracy=("accuracy", lambda s: np.average(s, weights=fold_df.loc[s.index, "n_samples"])),
        )
        .sort_values("weighted_accuracy", ascending=False, kind="stable")
    )
    for col in ["mean_group_accuracy", "mean_group_macro_f1", "weighted_accuracy"]:
        summary_df[col] = summary_df[col].round(4)
    return fold_df, summary_df


def write_outputs(sample_df, fold_df, summary_df, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sample_df.to_csv(output_root / "oracle_gate_targets.csv", index=False)
    fold_df.to_csv(output_root / "gate_target_group_cv.csv", index=False)
    summary_df.to_csv(output_root / "gate_target_cv_summary.csv", index=False)

    group_summary = (
        sample_df.groupby(["dataset", "train_ratio_tag", "seed", "oracle_teacher"], as_index=False)
        .size()
        .rename(columns={"size": "win_count"})
    )
    group_summary.to_csv(output_root / "oracle_gate_target_group_summary.csv", index=False)

    lines = [
        "# Oracle Gate Target Diagnostics",
        "",
        "## Cross-Setting Probe Summary",
        "",
        dataframe_to_markdown(summary_df),
        "",
        "## Interpretation",
        "",
        "These probes test whether current reliability signals can predict the per-sample oracle best teacher.",
        "If learned probes cannot beat the setting-level prior baseline, the next method should use stronger gate supervision or different signals.",
    ]
    (output_root / "oracle_gate_target_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Probe oracle teacher-selection targets for the second ADMET paper.")
    parser.add_argument("--datasets", nargs="+", default=REGRESSION_DATASETS)
    parser.add_argument("--teachers", nargs="+", default=DEFAULT_TEACHERS)
    parser.add_argument("--train-ratio-tags", nargs="+", type=int, default=TRAIN_RATIO_TAGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--teacher-root", default=str(PROJECT_ROOT / "data" / "teacher_predictions"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results_gate_targets" / "summary"))
    return parser.parse_args()


def main():
    args = parse_args()
    sample_df = build_gate_target_rows(
        results_root=args.results_root,
        teacher_root=args.teacher_root,
        datasets=args.datasets,
        teachers=args.teachers,
        ratios=args.train_ratio_tags,
        seeds=args.seeds,
    )
    fold_df, summary_df = evaluate_grouped_models(sample_df, args.teachers)
    write_outputs(sample_df, fold_df, summary_df, args.output_root)
    print(f"gate_target_summary: {Path(args.output_root) / 'oracle_gate_target_diagnostics.md'}")


if __name__ == "__main__":
    main()
