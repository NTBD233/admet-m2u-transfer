import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compare_distillation import REGRESSION_DATASETS, fmt, summarize_model
from utils.config import PROJECT_ROOT, TRAIN_RATIO_TAGS
from utils.summary import collect_metrics, dataframe_to_markdown


DEFAULT_LAMBDA_ROOTS = {
    "0.01": PROJECT_ROOT / "results_distill_lambda" / "lambda_0_01",
    "0.1": PROJECT_ROOT / "results_distill_regression",
    "0.3": PROJECT_ROOT / "results_distill_lambda" / "lambda_0_3",
    "1.0": PROJECT_ROOT / "results_distill_lambda" / "lambda_1_0",
}


def parse_lambda_roots(values):
    roots = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --lambda-root value: {value}. Expected lambda=path")
        lambda_value, root = value.split("=", 1)
        roots[lambda_value] = Path(root)
    return roots


def build_lambda_comparison(base_metrics, lambda_roots, datasets, train_ratio_tags):
    rows = []
    for lambda_value, root in lambda_roots.items():
        metrics = collect_metrics(root)
        if metrics.empty:
            print(f"Skipping empty lambda result root: {lambda_value} -> {root}")
            continue

        for dataset in datasets:
            for train_ratio_tag in train_ratio_tags:
                base_mean, base_std, base_n = summarize_model(
                    base_metrics,
                    dataset,
                    "ECFP4_MLP_DescAdapterFusion",
                    train_ratio_tag,
                )
                teacher_mean, teacher_std, teacher_n = summarize_model(
                    base_metrics,
                    dataset,
                    "ECFP4_Desc_RF",
                    train_ratio_tag,
                )
                ecfp_rf_mean, ecfp_rf_std, ecfp_rf_n = summarize_model(
                    base_metrics,
                    dataset,
                    "ECFP4_RF",
                    train_ratio_tag,
                )
                distill_mean, distill_std, distill_n = summarize_model(
                    metrics,
                    dataset,
                    "ECFP4_MLP_DescAdapterFusion",
                    train_ratio_tag,
                )

                rows.append({
                    "lambda_distill": lambda_value,
                    "dataset": dataset,
                    "train_ratio_tag": train_ratio_tag,
                    "base_adapter_rmse": fmt(base_mean, base_std),
                    "distilled_adapter_rmse": fmt(distill_mean, distill_std),
                    "ecfp_rf_rmse": fmt(ecfp_rf_mean, ecfp_rf_std),
                    "teacher_ecfp_desc_rf_rmse": fmt(teacher_mean, teacher_std),
                    "base_adapter_rmse_mean": base_mean,
                    "distilled_adapter_rmse_mean": distill_mean,
                    "ecfp_rf_rmse_mean": ecfp_rf_mean,
                    "teacher_ecfp_desc_rf_rmse_mean": teacher_mean,
                    "distilled_n_seeds": distill_n,
                    "base_n_seeds": base_n,
                    "teacher_n_seeds": teacher_n,
                    "ecfp_rf_n_seeds": ecfp_rf_n,
                    "distill_minus_adapter_rmse": (
                        round(distill_mean - base_mean, 4)
                        if not pd.isna(distill_mean) and not pd.isna(base_mean)
                        else np.nan
                    ),
                    "distill_minus_teacher_rmse": (
                        round(distill_mean - teacher_mean, 4)
                        if not pd.isna(distill_mean) and not pd.isna(teacher_mean)
                        else np.nan
                    ),
                    "distill_beats_adapter": (
                        bool(distill_mean < base_mean)
                        if not pd.isna(distill_mean) and not pd.isna(base_mean)
                        else False
                    ),
                    "distill_beats_ecfp_rf": (
                        bool(distill_mean < ecfp_rf_mean)
                        if not pd.isna(distill_mean) and not pd.isna(ecfp_rf_mean)
                        else False
                    ),
                })
    return pd.DataFrame(rows)


def build_lambda_summary(comparison_df):
    if comparison_df.empty:
        return pd.DataFrame()

    rows = []
    for lambda_value, group in comparison_df.groupby("lambda_distill", sort=False):
        completed = group[group["distilled_n_seeds"] == 3]
        rows.append({
            "lambda_distill": lambda_value,
            "completed_settings": f"{len(completed)}/{len(group)}",
            "beats_adapter": f"{int(completed['distill_beats_adapter'].sum())}/{len(completed)}",
            "beats_ecfp_rf": f"{int(completed['distill_beats_ecfp_rf'].sum())}/{len(completed)}",
            "mean_delta_vs_adapter": f"{float(completed['distill_minus_adapter_rmse'].mean()):.4f}",
            "mean_gap_to_teacher": f"{float(completed['distill_minus_teacher_rmse'].mean()):.4f}",
        })
    return pd.DataFrame(rows)


def best_lambda_by_setting(comparison_df):
    completed = comparison_df[comparison_df["distilled_n_seeds"] == 3].copy()
    if completed.empty:
        return pd.DataFrame()
    idx = completed.groupby(["dataset", "train_ratio_tag"])["distilled_adapter_rmse_mean"].idxmin()
    best = completed.loc[idx].copy()
    for column in ["distill_minus_adapter_rmse", "distill_minus_teacher_rmse"]:
        best[column] = best[column].map(lambda value: f"{value:.4f}")
    return best[
        [
            "dataset",
            "train_ratio_tag",
            "lambda_distill",
            "base_adapter_rmse",
            "distilled_adapter_rmse",
            "teacher_ecfp_desc_rf_rmse",
            "distill_minus_adapter_rmse",
            "distill_minus_teacher_rmse",
        ]
    ].sort_values(["dataset", "train_ratio_tag"], kind="stable")


def write_markdown(summary_df, best_df, output_path):
    lines = [
        "# Distillation Lambda Sweep",
        "",
        "## Lambda Summary",
        "",
        dataframe_to_markdown(summary_df),
        "## Best Lambda Per Setting",
        "",
        dataframe_to_markdown(best_df),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare distillation lambda sweep results.")
    parser.add_argument("--base-results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results_distill_lambda" / "summary"))
    parser.add_argument("--lambda-root", action="append", default=None, help="Format: lambda=results_root")
    parser.add_argument("--datasets", nargs="+", default=REGRESSION_DATASETS)
    parser.add_argument("--train-ratio-tags", nargs="+", type=int, default=TRAIN_RATIO_TAGS)
    return parser.parse_args()


def main():
    args = parse_args()
    base_metrics = collect_metrics(args.base_results_root)
    if base_metrics.empty:
        raise SystemExit(f"No base metrics found under: {args.base_results_root}")

    lambda_roots = (
        parse_lambda_roots(args.lambda_root)
        if args.lambda_root is not None
        else DEFAULT_LAMBDA_ROOTS
    )
    comparison_df = build_lambda_comparison(
        base_metrics=base_metrics,
        lambda_roots=lambda_roots,
        datasets=args.datasets,
        train_ratio_tags=args.train_ratio_tags,
    )
    if comparison_df.empty:
        raise SystemExit("No lambda comparison rows were generated.")

    summary_df = build_lambda_summary(comparison_df)
    best_df = best_lambda_by_setting(comparison_df)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_path = output_root / "distillation_lambda_comparison.csv"
    summary_path = output_root / "distillation_lambda_summary.csv"
    best_path = output_root / "distillation_lambda_best_by_setting.csv"
    md_path = output_root / "distillation_lambda_summary.md"

    comparison_df.to_csv(comparison_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    best_df.to_csv(best_path, index=False)
    write_markdown(summary_df, best_df, md_path)

    print(f"Saved comparison: {comparison_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved best-by-setting: {best_path}")
    print(f"Saved markdown: {md_path}")


if __name__ == "__main__":
    main()
