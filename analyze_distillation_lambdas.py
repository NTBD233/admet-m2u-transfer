import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_distillation_lambdas import DEFAULT_LAMBDA_ROOTS
from utils.config import PROJECT_ROOT, SEEDS, TRAIN_RATIO_TAGS
from utils.summary import dataframe_to_markdown


REGRESSION_DATASETS = [
    "caco2_wang",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "vdss_lombardo",
    "ppbr_az",
]
STUDENT_MODEL = "ECFP4_MLP_DescAdapterFusion"
TEACHER_MODEL = "ECFP4_Desc_RF"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def read_metrics(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_dir(root, dataset, ratio, seed):
    return Path(root) / dataset / STUDENT_MODEL / f"train_{ratio}" / f"seed_{seed}"


def teacher_npz_path(root, dataset, ratio, seed):
    return (
        Path(root)
        / dataset
        / TEACHER_MODEL
        / f"train_{ratio}"
        / f"seed_{seed}"
        / "test_teacher_predictions.npz"
    )


def read_predictions(path):
    df = pd.read_csv(path)
    return df["smiles"].astype(str).to_numpy(), df["y_true"].to_numpy(), df["pred"].to_numpy()


def read_teacher(path):
    z = np.load(path, allow_pickle=True)
    return z["smiles"].astype(str), z["y"].astype(float), z["pred"].astype(float)


def build_seed_diagnostics(base_root, lambda_roots, teacher_root, datasets, ratios, seeds):
    rows = []
    for dataset in datasets:
        for ratio in ratios:
            for seed in seeds:
                base_dir = run_dir(base_root, dataset, ratio, seed)
                base_metrics_path = base_dir / "metrics.json"
                base_pred_path = base_dir / "test_predictions.csv"
                teacher_path = teacher_npz_path(teacher_root, dataset, ratio, seed)

                if not base_metrics_path.exists() or not base_pred_path.exists() or not teacher_path.exists():
                    continue

                base_metrics = read_metrics(base_metrics_path)
                base_smiles, y_true, base_pred = read_predictions(base_pred_path)
                teacher_smiles, teacher_y, teacher_pred = read_teacher(teacher_path)

                if len(base_smiles) != len(teacher_smiles) or not np.array_equal(base_smiles, teacher_smiles):
                    raise ValueError(f"Prediction order mismatch for {dataset} train_{ratio} seed_{seed}")
                if not np.allclose(y_true.astype(float), teacher_y.astype(float), equal_nan=True):
                    raise ValueError(f"Label mismatch for {dataset} train_{ratio} seed_{seed}")

                base_teacher_rmse = rmse(base_pred, teacher_pred)
                base_teacher_corr = corr(base_pred, teacher_pred)
                teacher_test_rmse = rmse(teacher_pred, y_true)

                for lambda_value, root in lambda_roots.items():
                    distill_dir = run_dir(root, dataset, ratio, seed)
                    metrics_path = distill_dir / "metrics.json"
                    pred_path = distill_dir / "test_predictions.csv"
                    if not metrics_path.exists() or not pred_path.exists():
                        continue

                    distill_metrics = read_metrics(metrics_path)
                    distill_smiles, distill_y, distill_pred = read_predictions(pred_path)
                    if len(distill_smiles) != len(base_smiles) or not np.array_equal(distill_smiles, base_smiles):
                        raise ValueError(
                            f"Distilled prediction order mismatch for {lambda_value} "
                            f"{dataset} train_{ratio} seed_{seed}"
                        )
                    if not np.allclose(distill_y.astype(float), y_true.astype(float), equal_nan=True):
                        raise ValueError(
                            f"Distilled label mismatch for {lambda_value} {dataset} train_{ratio} seed_{seed}"
                        )

                    rows.append({
                        "lambda_distill": lambda_value,
                        "dataset": dataset,
                        "train_ratio_tag": ratio,
                        "seed": seed,
                        "base_valid_rmse": float(base_metrics["valid_rmse"]),
                        "distilled_valid_rmse": float(distill_metrics["valid_rmse"]),
                        "base_test_rmse": float(base_metrics["test_rmse"]),
                        "distilled_test_rmse": float(distill_metrics["test_rmse"]),
                        "teacher_test_rmse": teacher_test_rmse,
                        "delta_valid_rmse": float(distill_metrics["valid_rmse"] - base_metrics["valid_rmse"]),
                        "delta_test_rmse": float(distill_metrics["test_rmse"] - base_metrics["test_rmse"]),
                        "teacher_gap_after_distill": float(distill_metrics["test_rmse"] - teacher_test_rmse),
                        "teacher_advantage_over_base": float(base_metrics["test_rmse"] - teacher_test_rmse),
                        "base_teacher_rmse": base_teacher_rmse,
                        "distilled_teacher_rmse": rmse(distill_pred, teacher_pred),
                        "delta_teacher_rmse": rmse(distill_pred, teacher_pred) - base_teacher_rmse,
                        "base_teacher_corr": base_teacher_corr,
                        "distilled_teacher_corr": corr(distill_pred, teacher_pred),
                        "delta_teacher_corr": corr(distill_pred, teacher_pred) - base_teacher_corr,
                        "base_teacher_mae": mae(base_pred, teacher_pred),
                        "distilled_teacher_mae": mae(distill_pred, teacher_pred),
                    })
    return pd.DataFrame(rows)


def summarize_by_setting(seed_df):
    group_cols = ["lambda_distill", "dataset", "train_ratio_tag"]
    out = seed_df.groupby(group_cols, as_index=False).agg(
        n_seeds=("seed", "nunique"),
        delta_test_rmse_mean=("delta_test_rmse", "mean"),
        delta_test_rmse_std=("delta_test_rmse", "std"),
        delta_valid_rmse_mean=("delta_valid_rmse", "mean"),
        teacher_gap_mean=("teacher_gap_after_distill", "mean"),
        teacher_advantage_over_base_mean=("teacher_advantage_over_base", "mean"),
        delta_teacher_rmse_mean=("delta_teacher_rmse", "mean"),
        delta_teacher_corr_mean=("delta_teacher_corr", "mean"),
        distilled_teacher_rmse_mean=("distilled_teacher_rmse", "mean"),
        distilled_teacher_corr_mean=("distilled_teacher_corr", "mean"),
    )
    for col in out.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            out[col] = out[col].round(4)
    return out


def summarize_by_dataset(setting_df):
    rows = []
    for dataset, group in setting_df.groupby("dataset", sort=False):
        best = group.loc[group.groupby("train_ratio_tag")["delta_test_rmse_mean"].idxmin()]
        rows.append({
            "dataset": dataset,
            "best_lambda_pattern": ", ".join(
                f"{int(row.train_ratio_tag)}%:{row.lambda_distill}" for row in best.itertuples()
            ),
            "best_mean_delta_rmse": round(float(best["delta_test_rmse_mean"].mean()), 4),
            "lambda_1_delta_rmse": round(float(
                group[group["lambda_distill"] == "1.0"]["delta_test_rmse_mean"].mean()
            ), 4),
            "mean_teacher_advantage_over_base": round(float(
                group[group["lambda_distill"] == "1.0"]["teacher_advantage_over_base_mean"].mean()
            ), 4),
            "lambda_1_mean_delta_teacher_rmse": round(float(
                group[group["lambda_distill"] == "1.0"]["delta_teacher_rmse_mean"].mean()
            ), 4),
        })
    return pd.DataFrame(rows)


def summarize_by_ratio(setting_df):
    rows = []
    for ratio, group in setting_df.groupby("train_ratio_tag", sort=True):
        best = group.loc[group.groupby("dataset")["delta_test_rmse_mean"].idxmin()]
        rows.append({
            "train_ratio_tag": ratio,
            "best_mean_delta_rmse": round(float(best["delta_test_rmse_mean"].mean()), 4),
            "lambda_1_delta_rmse": round(float(
                group[group["lambda_distill"] == "1.0"]["delta_test_rmse_mean"].mean()
            ), 4),
            "lambda_1_beats_base": f"{int((group[group['lambda_distill'] == '1.0']['delta_test_rmse_mean'] < 0).sum())}/5",
            "most_common_best_lambda": best["lambda_distill"].mode().iloc[0],
        })
    return pd.DataFrame(rows)


def write_markdown(setting_df, dataset_df, ratio_df, output_path):
    lambda_summary = setting_df.groupby("lambda_distill", as_index=False).agg(
        complete_settings=("n_seeds", lambda s: f"{int((s == 3).sum())}/{len(s)}"),
        beats_base=("delta_test_rmse_mean", lambda s: f"{int((s < 0).sum())}/{len(s)}"),
        mean_delta_rmse=("delta_test_rmse_mean", "mean"),
        mean_delta_teacher_rmse=("delta_teacher_rmse_mean", "mean"),
        mean_delta_teacher_corr=("delta_teacher_corr_mean", "mean"),
    )
    for col in ["mean_delta_rmse", "mean_delta_teacher_rmse", "mean_delta_teacher_corr"]:
        lambda_summary[col] = lambda_summary[col].map(lambda value: f"{value:.4f}")

    lines = [
        "# Distillation Lambda Diagnostics",
        "",
        "## Lambda-Level Summary",
        "",
        dataframe_to_markdown(lambda_summary),
        "",
        "## Dataset-Level Pattern",
        "",
        dataframe_to_markdown(dataset_df),
        "",
        "## Train-Ratio Pattern",
        "",
        dataframe_to_markdown(ratio_df),
        "",
        "## Interpretation",
        "",
        "- Stronger teacher supervision (`lambda_distill = 1.0`) gives the best global test RMSE delta.",
        "- The best lambda is not uniform across datasets and ratios, so a fixed global lambda is a practical baseline, not the final method design.",
        "- Negative `delta_teacher_rmse` means the distilled student moved closer to the RF teacher in prediction space.",
        "- If performance improves while teacher alignment also improves, the result supports actual teacher transfer rather than random regularization.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze distillation lambda behavior.")
    parser.add_argument("--base-results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--teacher-root", default=str(PROJECT_ROOT / "data" / "teacher_predictions"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results_distill_lambda" / "analysis"))
    parser.add_argument("--lambda-root", action="append", default=None, help="Format: lambda=results_root")
    parser.add_argument("--datasets", nargs="+", default=REGRESSION_DATASETS)
    parser.add_argument("--train-ratio-tags", nargs="+", type=int, default=TRAIN_RATIO_TAGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    return parser.parse_args()


def main():
    args = parse_args()
    lambda_roots = DEFAULT_LAMBDA_ROOTS
    if args.lambda_root is not None:
        lambda_roots = {}
        for item in args.lambda_root:
            value, root = item.split("=", 1)
            lambda_roots[value] = Path(root)

    seed_df = build_seed_diagnostics(
        base_root=Path(args.base_results_root),
        lambda_roots=lambda_roots,
        teacher_root=Path(args.teacher_root),
        datasets=args.datasets,
        ratios=args.train_ratio_tags,
        seeds=args.seeds,
    )
    if seed_df.empty:
        raise SystemExit("No seed diagnostics were generated.")

    setting_df = summarize_by_setting(seed_df)
    dataset_df = summarize_by_dataset(setting_df)
    ratio_df = summarize_by_ratio(setting_df)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seed_path = output_root / "distillation_lambda_seed_diagnostics.csv"
    setting_path = output_root / "distillation_lambda_setting_diagnostics.csv"
    dataset_path = output_root / "distillation_lambda_dataset_summary.csv"
    ratio_path = output_root / "distillation_lambda_ratio_summary.csv"
    md_path = output_root / "distillation_lambda_diagnostics.md"

    seed_df.to_csv(seed_path, index=False)
    setting_df.to_csv(setting_path, index=False)
    dataset_df.to_csv(dataset_path, index=False)
    ratio_df.to_csv(ratio_path, index=False)
    write_markdown(setting_df, dataset_df, ratio_df, md_path)

    print(f"Saved seed diagnostics: {seed_path}")
    print(f"Saved setting diagnostics: {setting_path}")
    print(f"Saved dataset summary: {dataset_path}")
    print(f"Saved ratio summary: {ratio_path}")
    print(f"Saved markdown: {md_path}")


if __name__ == "__main__":
    main()
