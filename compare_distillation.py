import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import DATASETS, PROJECT_ROOT, TRAIN_RATIO_TAGS
from utils.summary import collect_metrics, dataframe_to_markdown


REGRESSION_DATASETS = [
    name for name, cfg in DATASETS.items()
    if cfg["task_type"] == "regression"
]

COMPARISON_MODELS = [
    "ECFP4_MLP_DescPred",
    "ECFP4_MLP_DescConcat",
    "ECFP4_MLP_DescAdapterFusion",
    "ECFP4_RF",
    "ECFP4_Desc_RF",
]


def summarize_model(metrics_df, dataset, model, train_ratio_tag):
    rows = metrics_df[
        (metrics_df["dataset"] == dataset)
        & (metrics_df["model"] == model)
        & (metrics_df["train_ratio_tag"] == train_ratio_tag)
    ]
    if rows.empty:
        return np.nan, np.nan, 0
    return (
        round(float(rows["test_rmse"].mean()), 4),
        round(float(rows["test_rmse"].std(ddof=1)), 4),
        int(rows["seed"].nunique()),
    )


def fmt(mean, std):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        std = 0.0
    return f"{mean:.4f}±{std:.4f}"


def build_comparison(base_metrics, distill_metrics, datasets, train_ratio_tags):
    rows = []
    for dataset in datasets:
        for train_ratio_tag in train_ratio_tags:
            out = {
                "dataset": dataset,
                "train_ratio_tag": train_ratio_tag,
            }

            for model in COMPARISON_MODELS:
                mean, std, n_seeds = summarize_model(
                    base_metrics,
                    dataset,
                    model,
                    train_ratio_tag,
                )
                key = model.replace("ECFP4_MLP_", "").replace("ECFP4_", "ECFP4_")
                out[f"{key}_rmse"] = fmt(mean, std)
                out[f"{key}_rmse_mean"] = mean
                out[f"{key}_n_seeds"] = n_seeds

            distill_mean, distill_std, distill_n = summarize_model(
                distill_metrics,
                dataset,
                "ECFP4_MLP_DescAdapterFusion",
                train_ratio_tag,
            )
            out["DistilledAdapterFusion_rmse"] = fmt(distill_mean, distill_std)
            out["DistilledAdapterFusion_rmse_mean"] = distill_mean
            out["DistilledAdapterFusion_n_seeds"] = distill_n

            base_adapter = out["DescAdapterFusion_rmse_mean"]
            teacher = out["ECFP4_Desc_RF_rmse_mean"]
            ecfp_rf = out["ECFP4_RF_rmse_mean"]

            out["distill_minus_adapter_rmse"] = (
                round(distill_mean - base_adapter, 4)
                if not pd.isna(distill_mean) and not pd.isna(base_adapter)
                else np.nan
            )
            out["distill_beats_adapter"] = (
                bool(distill_mean < base_adapter)
                if not pd.isna(distill_mean) and not pd.isna(base_adapter)
                else False
            )
            out["distill_minus_teacher_rf_rmse"] = (
                round(distill_mean - teacher, 4)
                if not pd.isna(distill_mean) and not pd.isna(teacher)
                else np.nan
            )
            out["distill_beats_ecfp_rf"] = (
                bool(distill_mean < ecfp_rf)
                if not pd.isna(distill_mean) and not pd.isna(ecfp_rf)
                else False
            )
            rows.append(out)
    return pd.DataFrame(rows)


def write_brief(comparison_df, output_path):
    completed = comparison_df[comparison_df["DistilledAdapterFusion_n_seeds"] > 0]
    total = len(comparison_df)
    complete = int((comparison_df["DistilledAdapterFusion_n_seeds"] == 3).sum())
    beats_adapter = int(completed["distill_beats_adapter"].sum())
    beats_ecfp_rf = int(completed["distill_beats_ecfp_rf"].sum())

    mean_delta = completed["distill_minus_adapter_rmse"].mean()
    mean_teacher_gap = completed["distill_minus_teacher_rf_rmse"].mean()

    lines = [
        "# Regression Distillation Comparison",
        "",
        f"- Completed settings: {complete}/{total}",
        f"- Distilled AdapterFusion beats base AdapterFusion: {beats_adapter}/{len(completed)}",
        f"- Distilled AdapterFusion beats ECFP4_RF: {beats_ecfp_rf}/{len(completed)}",
        f"- Mean RMSE delta vs base AdapterFusion: {mean_delta:.4f}",
        f"- Mean RMSE gap to ECFP4_Desc_RF teacher: {mean_teacher_gap:.4f}",
        "",
    ]
    table_df = completed[
        [
            "dataset",
            "train_ratio_tag",
            "DescPred_rmse",
            "DescConcat_rmse",
            "DescAdapterFusion_rmse",
            "DistilledAdapterFusion_rmse",
            "ECFP4_RF_rmse",
            "ECFP4_Desc_RF_rmse",
            "distill_minus_adapter_rmse",
            "distill_minus_teacher_rf_rmse",
        ]
    ].copy()
    for col in ["distill_minus_adapter_rmse", "distill_minus_teacher_rf_rmse"]:
        table_df[col] = table_df[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    lines.append(dataframe_to_markdown(table_df))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare regression teacher-distilled AdapterFusion runs.")
    parser.add_argument("--base-results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--distill-results-root", default=str(PROJECT_ROOT / "results_distill_regression"))
    parser.add_argument("--datasets", nargs="+", default=REGRESSION_DATASETS)
    parser.add_argument("--train-ratio-tags", nargs="+", type=int, default=TRAIN_RATIO_TAGS)
    return parser.parse_args()


def main():
    args = parse_args()
    base_metrics = collect_metrics(args.base_results_root)
    distill_metrics = collect_metrics(args.distill_results_root)

    if base_metrics.empty:
        raise SystemExit(f"No base metrics found under: {args.base_results_root}")
    if distill_metrics.empty:
        raise SystemExit(f"No distillation metrics found under: {args.distill_results_root}")

    comparison_df = build_comparison(
        base_metrics=base_metrics,
        distill_metrics=distill_metrics,
        datasets=args.datasets,
        train_ratio_tags=args.train_ratio_tags,
    )

    summary_dir = Path(args.distill_results_root) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "regression_distillation_comparison.csv"
    md_path = summary_dir / "regression_distillation_comparison.md"
    comparison_df.to_csv(csv_path, index=False)
    write_brief(comparison_df, md_path)

    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison MD: {md_path}")


if __name__ == "__main__":
    main()
