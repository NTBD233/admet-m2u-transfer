from pathlib import Path

import numpy as np
import pandas as pd

from utils.io import load_json, save_json


CLASSIFICATION_METRICS = ["valid_roc_auc", "valid_pr_auc", "test_roc_auc", "test_pr_auc"]
REGRESSION_METRICS = ["valid_mae", "valid_rmse", "test_mae", "test_rmse"]


def has_lambda_ablation(metrics_df):
    if "lambda_transfer" not in metrics_df.columns:
        return False
    values = metrics_df["lambda_transfer"].dropna().unique()
    return len(values) > 1


def collect_metrics(results_root):
    rows = []
    results_root = Path(results_root)

    for metrics_path in sorted(results_root.rglob("metrics.json")):
        try:
            metrics = load_json(metrics_path)
        except Exception as exc:
            print(f"Skipping unreadable metrics file: {metrics_path} ({exc})")
            continue
        metrics["_metrics_path"] = str(metrics_path)
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df

    dedupe_cols = [
        col for col in [
            "dataset",
            "task_type",
            "model",
            "train_ratio_tag",
            "seed",
        ] if col in metrics_df.columns
    ]
    if has_lambda_ablation(metrics_df) and "lambda_transfer" in metrics_df.columns:
        dedupe_cols.append("lambda_transfer")

    if dedupe_cols:
        metrics_df = metrics_df.drop_duplicates(dedupe_cols, keep="last")
    return metrics_df.drop(columns=["_metrics_path"], errors="ignore")


def format_mean_std(mean, std):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        std = 0.0
    return f"{mean:.4f}±{std:.4f}"


def dataframe_to_markdown(df):
    if df.empty:
        return ""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = ["" if pd.isna(row[col]) else str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def build_mean_std_summary(metrics_df):
    if metrics_df.empty:
        return pd.DataFrame()

    metric_cols = [
        col for col in CLASSIFICATION_METRICS + REGRESSION_METRICS
        if col in metrics_df.columns
    ]

    group_cols = ["dataset", "task_type", "model", "train_ratio_tag"]
    if has_lambda_ablation(metrics_df):
        group_cols.append("lambda_transfer")

    grouped = metrics_df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"])

    rows = []
    for index, row in grouped.iterrows():
        out = {
            "dataset": index[0],
            "task_type": index[1],
            "model": index[2],
            "train_ratio_tag": index[3],
        }
        if "lambda_transfer" in group_cols:
            out["lambda_transfer"] = index[4]
        for metric in metric_cols:
            mean = row[(metric, "mean")]
            std = row[(metric, "std")]
            if not pd.isna(mean):
                out[metric] = format_mean_std(mean, std)
                out[f"{metric}_mean"] = round(float(mean), 4)
                out[f"{metric}_std"] = round(float(0.0 if pd.isna(std) else std), 4)
        rows.append(out)

    return pd.DataFrame(rows)


def save_summaries(metrics_df, results_root):
    results_root = Path(results_root)
    summary_dir = results_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_metrics_path = summary_dir / "all_seed_metrics.csv"
    summary_path = summary_dir / "mean_std_summary.csv"
    summary_json_path = summary_dir / "mean_std_summary.json"
    main_table_path = summary_dir / "main_table.csv"
    main_table_md_path = summary_dir / "main_table.md"
    low_resource_path = summary_dir / "low_resource_table.csv"

    metrics_df.to_csv(all_metrics_path, index=False)
    summary_df = build_mean_std_summary(metrics_df)
    summary_df.to_csv(summary_path, index=False)

    main_table_df = build_main_table(summary_df)
    main_table_df.to_csv(main_table_path, index=False)
    main_table_md_path.write_text(
        dataframe_to_markdown(main_table_df),
        encoding="utf-8",
    )

    low_resource_df = build_low_resource_table(summary_df)
    low_resource_df.to_csv(low_resource_path, index=False)

    records = summary_df.replace({np.nan: None}).to_dict(orient="records")
    save_json(records, summary_json_path)

    return {
        "all_metrics": str(all_metrics_path),
        "mean_std_csv": str(summary_path),
        "mean_std_json": str(summary_json_path),
        "main_table_csv": str(main_table_path),
        "main_table_md": str(main_table_md_path),
        "low_resource_csv": str(low_resource_path),
    }


def build_main_table(summary_df):
    if summary_df.empty:
        return pd.DataFrame()

    cols = [
        "dataset",
        "task_type",
        "model",
        "train_ratio_tag",
        "test_roc_auc",
        "test_pr_auc",
        "test_mae",
        "test_rmse",
    ]
    present = [col for col in cols if col in summary_df.columns]
    return summary_df[present].sort_values(
        ["dataset", "train_ratio_tag", "model"],
        kind="stable",
    )


def build_low_resource_table(summary_df):
    if summary_df.empty or "train_ratio_tag" not in summary_df.columns:
        return pd.DataFrame()

    low_df = summary_df[summary_df["train_ratio_tag"].isin([10, 20, 50])].copy()
    return build_main_table(low_df)
