import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from train_ml_baselines import build_features, load_split, model_name_parts
from utils.config import DATASETS, FEATURE_ROOT, PROJECT_ROOT, RESULTS_ROOT, SEEDS, TRAIN_RATIO_TAGS
from utils.dataset import feature_paths
from utils.summary import dataframe_to_markdown


REGRESSION_DATASETS = [
    name for name, cfg in DATASETS.items()
    if cfg["task_type"] == "regression"
]
DEFAULT_TEACHERS = [
    "ECFP4_RF",
    "Desc_RF",
    "ECFP4_Desc_RF",
    "ECFP4_XGB",
    "Desc_XGB",
    "ECFP4_Desc_XGB",
]


def rmse(y_true, pred):
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - pred) ** 2)))


def mae(y_true, pred):
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(y_true - pred)))


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return np.nan
    a = a[valid]
    b = b[valid]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def prediction_path(results_root, dataset, teacher, ratio, seed, split):
    return (
        Path(results_root)
        / dataset
        / teacher
        / f"train_{ratio}"
        / f"seed_{seed}"
        / f"{split}_predictions.csv"
    )


def teacher_npz_path(teacher_root, dataset, teacher, ratio, seed, split):
    filename = f"train_{ratio}_teacher_predictions.npz" if split == "train" else f"{split}_teacher_predictions.npz"
    return (
        Path(teacher_root)
        / dataset
        / teacher
        / f"train_{ratio}"
        / f"seed_{seed}"
        / filename
    )


def run_dir(results_root, dataset, teacher, ratio, seed):
    return (
        Path(results_root)
        / dataset
        / teacher
        / f"train_{ratio}"
        / f"seed_{seed}"
    )


def split_feature_path(dataset, ratio, split):
    train_path, valid_path, test_path = feature_paths(
        dataset_name=dataset,
        feature_root=FEATURE_ROOT,
        train_ratio_tag=ratio,
    )
    if split == "train":
        return train_path
    if split == "valid":
        return valid_path
    if split == "test":
        return test_path
    raise ValueError(f"Unsupported split: {split}")


def classifier_positive_probability(estimator, x):
    prob = estimator.predict_proba(x)
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        return prob[:, -1]
    classes = list(classes)
    if 1 in classes:
        return prob[:, classes.index(1)]
    return np.zeros(len(x), dtype=np.float32)


def compute_model_uncertainty(results_root, dataset, teacher, ratio, seed, split):
    model_path = run_dir(results_root, dataset, teacher, ratio, seed) / "best_model.pkl"
    if not model_path.exists():
        return None

    with model_path.open("rb") as f:
        model = pickle.load(f)

    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        return None

    feature_set, _ = model_name_parts(teacher)
    split = load_split(split_feature_path(dataset, ratio, split))
    x = build_features(split, feature_set)
    task_type = DATASETS[dataset]["task_type"]

    raw_preds = []
    for estimator in estimators:
        if task_type == "classification":
            prob = classifier_positive_probability(estimator, x)
            prob = np.clip(prob, 1e-7, 1 - 1e-7)
            raw = np.log(prob / (1 - prob))
        else:
            raw = estimator.predict(x)
        raw_preds.append(np.asarray(raw, dtype=np.float32).reshape(-1))

    if not raw_preds:
        return None
    return np.vstack(raw_preds).std(axis=0).astype(float)


def load_prediction(results_root, teacher_root, dataset, teacher, ratio, seed, split):
    csv_path = prediction_path(results_root, dataset, teacher, ratio, seed, split)
    if not csv_path.exists():
        return None, f"missing_predictions:{csv_path}"

    df = pd.read_csv(csv_path)
    required = {"smiles", "y_true", "pred"}
    if not required.issubset(df.columns):
        return None, f"bad_prediction_columns:{csv_path}"

    out = {
        "smiles": df["smiles"].astype(str).to_numpy(),
        "y": df["y_true"].to_numpy(dtype=float),
        "pred": df["pred"].to_numpy(dtype=float),
        "uncertainty": np.full(len(df), np.nan, dtype=float),
    }

    npz_path = teacher_npz_path(teacher_root, dataset, teacher, ratio, seed, split)
    if npz_path.exists():
        z = np.load(npz_path, allow_pickle=True)
        if "pred_uncertainty" in z.files and len(z["pred_uncertainty"]) == len(df):
            out["uncertainty"] = z["pred_uncertainty"].astype(float)
    if not np.isfinite(out["uncertainty"]).any():
        uncertainty = compute_model_uncertainty(results_root, dataset, teacher, ratio, seed, split)
        if uncertainty is not None and len(uncertainty) == len(df):
            out["uncertainty"] = uncertainty

    return out, None


def descriptor_ood_distances(dataset, ratio, split):
    train_path, valid_path, test_path = feature_paths(
        dataset_name=dataset,
        feature_root=FEATURE_ROOT,
        train_ratio_tag=ratio,
    )
    split_path = valid_path if split == "valid" else test_path
    train = np.load(train_path, allow_pickle=True)["X_desc"].astype(float)
    target = np.load(split_path, allow_pickle=True)["X_desc"].astype(float)

    try:
        from sklearn.neighbors import NearestNeighbors
    except ModuleNotFoundError:
        return np.full(len(target), np.nan, dtype=float)

    n_neighbors = min(5, len(train))
    if n_neighbors == 0:
        return np.full(len(target), np.nan, dtype=float)
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(train)
    distances, _ = nn.kneighbors(target)
    return distances.mean(axis=1)


def validate_alignment(records):
    first = records[0]
    for record in records[1:]:
        if len(record["y"]) != len(first["y"]):
            return False
        if not np.allclose(record["y"], first["y"], equal_nan=True):
            return False
        if not np.array_equal(record["smiles"], first["smiles"]):
            return False
    return True


def build_diagnostics(results_root, teacher_root, datasets, teachers, ratios, seeds, splits):
    performance_rows = []
    agreement_rows = []
    oracle_setting_rows = []
    oracle_sample_rows = []
    uncertainty_rows = []
    coverage_rows = []
    missing_rows = []

    for dataset in datasets:
        for ratio in ratios:
            ood_by_split = {}
            for split in splits:
                if split in {"valid", "test"}:
                    try:
                        ood_by_split[split] = descriptor_ood_distances(dataset, ratio, split)
                    except Exception:
                        ood_by_split[split] = None

            for seed in seeds:
                for split in splits:
                    loaded = {}
                    for teacher in teachers:
                        record, missing_reason = load_prediction(
                            results_root=results_root,
                            teacher_root=teacher_root,
                            dataset=dataset,
                            teacher=teacher,
                            ratio=ratio,
                            seed=seed,
                            split=split,
                        )
                        if record is None:
                            missing_rows.append({
                                "dataset": dataset,
                                "train_ratio_tag": ratio,
                                "seed": seed,
                                "split": split,
                                "teacher": teacher,
                                "reason": missing_reason,
                            })
                            continue

                        loaded[teacher] = record
                        abs_error = np.abs(record["y"] - record["pred"])
                        performance_rows.append({
                            "dataset": dataset,
                            "train_ratio_tag": ratio,
                            "seed": seed,
                            "split": split,
                            "teacher": teacher,
                            "n": len(record["y"]),
                            "rmse": rmse(record["y"], record["pred"]),
                            "mae": mae(record["y"], record["pred"]),
                            "mean_uncertainty": float(np.nanmean(record["uncertainty"]))
                            if np.isfinite(record["uncertainty"]).any()
                            else np.nan,
                        })

                        uncertainty_rows.append({
                            "dataset": dataset,
                            "train_ratio_tag": ratio,
                            "seed": seed,
                            "split": split,
                            "teacher": teacher,
                            "n": int(np.isfinite(record["uncertainty"]).sum()),
                            "uncertainty_error_corr": safe_corr(record["uncertainty"], abs_error),
                        })

                        ood = ood_by_split.get(split)
                        if ood is not None and len(ood) == len(abs_error):
                            coverage_rows.append({
                                "dataset": dataset,
                                "train_ratio_tag": ratio,
                                "seed": seed,
                                "split": split,
                                "teacher": teacher,
                                "n": len(abs_error),
                                "descriptor_ood_error_corr": safe_corr(ood, abs_error),
                                "mean_descriptor_ood": float(np.nanmean(ood)),
                            })

                    if len(loaded) < 2:
                        continue

                    records = list(loaded.values())
                    if not validate_alignment(records):
                        missing_rows.append({
                            "dataset": dataset,
                            "train_ratio_tag": ratio,
                            "seed": seed,
                            "split": split,
                            "teacher": "ALL",
                            "reason": "prediction_alignment_mismatch",
                        })
                        continue

                    teacher_names = list(loaded.keys())
                    for idx, teacher_a in enumerate(teacher_names):
                        for teacher_b in teacher_names[idx + 1:]:
                            pred_a = loaded[teacher_a]["pred"]
                            pred_b = loaded[teacher_b]["pred"]
                            agreement_rows.append({
                                "dataset": dataset,
                                "train_ratio_tag": ratio,
                                "seed": seed,
                                "split": split,
                                "teacher_a": teacher_a,
                                "teacher_b": teacher_b,
                                "n": len(pred_a),
                                "prediction_corr": safe_corr(pred_a, pred_b),
                                "prediction_rmse_distance": rmse(pred_a, pred_b),
                                "mean_abs_distance": mae(pred_a, pred_b),
                            })

                    teacher_scores = []
                    for teacher in teacher_names:
                        record = loaded[teacher]
                        teacher_scores.append((teacher, rmse(record["y"], record["pred"])))
                    best_teacher, best_rmse = min(teacher_scores, key=lambda item: item[1])
                    oracle_setting_rows.append({
                        "dataset": dataset,
                        "train_ratio_tag": ratio,
                        "seed": seed,
                        "split": split,
                        "best_teacher": best_teacher,
                        "best_rmse": best_rmse,
                        "n_teachers": len(teacher_names),
                    })

                    if split == "valid":
                        y = records[0]["y"]
                        preds = np.vstack([loaded[teacher]["pred"] for teacher in teacher_names])
                        errors = np.abs(preds - y.reshape(1, -1))
                        winners = np.argmin(errors, axis=0)
                        oracle_pred = preds[winners, np.arange(len(y))]
                        oracle_rmse = rmse(y, oracle_pred)
                        for idx, teacher in enumerate(teacher_names):
                            wins = int((winners == idx).sum())
                            oracle_sample_rows.append({
                                "dataset": dataset,
                                "train_ratio_tag": ratio,
                                "seed": seed,
                                "teacher": teacher,
                                "oracle_sample_wins": wins,
                                "n_samples": len(y),
                                "win_fraction": wins / len(y),
                                "oracle_valid_rmse": oracle_rmse,
                            })

    return {
        "teacher_performance": pd.DataFrame(performance_rows),
        "teacher_agreement": pd.DataFrame(agreement_rows),
        "oracle_setting_teachers": pd.DataFrame(oracle_setting_rows),
        "oracle_sample_validation": pd.DataFrame(oracle_sample_rows),
        "uncertainty_error_correlation": pd.DataFrame(uncertainty_rows),
        "coverage_error_correlation": pd.DataFrame(coverage_rows),
        "missing_teacher_runs": pd.DataFrame(missing_rows),
    }


def summarize_performance(performance_df):
    if performance_df.empty:
        return pd.DataFrame()
    test_df = performance_df[performance_df["split"] == "test"].copy()
    if test_df.empty:
        return pd.DataFrame()
    out = test_df.groupby("teacher", as_index=False).agg(
        completed_runs=("rmse", "count"),
        mean_test_rmse=("rmse", "mean"),
        median_test_rmse=("rmse", "median"),
        mean_test_mae=("mae", "mean"),
    )
    for col in ["mean_test_rmse", "median_test_rmse", "mean_test_mae"]:
        out[col] = out[col].round(4)
    return out.sort_values("mean_test_rmse", kind="stable")


def write_outputs(tables, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(output_root / f"{name}.csv", index=False)

    performance_summary = summarize_performance(tables["teacher_performance"])
    performance_summary.to_csv(output_root / "teacher_performance_summary.csv", index=False)

    lines = [
        "# Teacher Reliability Diagnostics",
        "",
        "## Teacher Performance Summary",
        "",
        dataframe_to_markdown(performance_summary),
        "## Generated Tables",
        "",
    ]
    for name, df in tables.items():
        lines.append(f"- `{name}.csv`: {len(df)} rows")
    lines.extend([
        "- `teacher_performance_summary.csv`: aggregate test RMSE by teacher",
        "",
        "## Next Decision",
        "",
        "Use these tables to decide whether teacher reliability varies enough to justify sample-level gating.",
        "If one teacher dominates every setting and oracle sample selection has little headroom, pivot to single-teacher uncertainty/OOD weighting.",
    ])
    (output_root / "teacher_reliability_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze RF/XGB teacher reliability for the second ADMET paper.")
    parser.add_argument("--datasets", nargs="+", default=REGRESSION_DATASETS)
    parser.add_argument("--teachers", nargs="+", default=DEFAULT_TEACHERS)
    parser.add_argument("--train-ratio-tags", nargs="+", type=int, default=TRAIN_RATIO_TAGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--teacher-root", default=str(PROJECT_ROOT / "data" / "teacher_predictions"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results_teacher_reliability" / "summary"))
    return parser.parse_args()


def main():
    args = parse_args()
    tables = build_diagnostics(
        results_root=args.results_root,
        teacher_root=args.teacher_root,
        datasets=args.datasets,
        teachers=args.teachers,
        ratios=args.train_ratio_tags,
        seeds=args.seeds,
        splits=args.splits,
    )
    write_outputs(tables, args.output_root)
    print(f"teacher_reliability_summary: {Path(args.output_root) / 'teacher_reliability_diagnostics.md'}")


if __name__ == "__main__":
    main()
