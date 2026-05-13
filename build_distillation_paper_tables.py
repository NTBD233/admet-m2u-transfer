import argparse
from pathlib import Path

import pandas as pd

from compare_distillation import REGRESSION_DATASETS, summarize_model
from compare_distillation_lambdas import DEFAULT_LAMBDA_ROOTS
from utils.config import PROJECT_ROOT
from utils.summary import collect_metrics, dataframe_to_markdown, format_mean_std


REGRESSION_MODELS = [
    "ECFP4_MLP",
    "ECFP4_MLP_DescPred",
    "ECFP4_MLP_DescAdapterFusion",
    "ECFP4_MLP_DescConcat",
    "ECFP4_RF",
    "ECFP4_Desc_RF",
]
CLASSIFICATION_MODELS = [
    "ECFP4_MLP",
    "ECFP4_MLP_DescPred",
    "ECFP4_MLP_DescAdapterFusion",
    "ECFP4_MLP_DescConcat",
    "ECFP4_RF",
    "ECFP4_Desc_RF",
]
MODEL_LABELS = {
    "ECFP4_MLP": "ECFP4 MLP",
    "ECFP4_MLP_DescPred": "DescPred",
    "ECFP4_MLP_DescAdapterFusion": "AdapterFusion",
    "ECFP4_MLP_DescConcat": "DescConcat",
    "ECFP4_RF": "ECFP4 RF",
    "ECFP4_Desc_RF": "ECFP4+Desc RF",
}


def fmt_metric(mean, std):
    return format_mean_std(mean, std)


def metric_mean_std(metrics_df, dataset, model, ratio, metric):
    rows = metrics_df[
        (metrics_df["dataset"] == dataset)
        & (metrics_df["model"] == model)
        & (metrics_df["train_ratio_tag"] == ratio)
    ]
    if rows.empty:
        return None, None, 0
    return (
        float(rows[metric].mean()),
        float(rows[metric].std(ddof=1)),
        int(rows["seed"].nunique()),
    )


def build_regression_main_table(base_metrics, fixed_metrics, adaptive_metrics, best_fixed_df, datasets, ratios):
    rows = []
    for dataset in datasets:
        for ratio in ratios:
            row = {"dataset": dataset, "train_ratio": ratio}
            means = {}
            for model in REGRESSION_MODELS:
                mean, std, _ = metric_mean_std(base_metrics, dataset, model, ratio, "test_rmse")
                means[model] = mean
                row[MODEL_LABELS[model]] = fmt_metric(mean, std)

            fixed_mean, fixed_std, _ = metric_mean_std(fixed_metrics, dataset, "ECFP4_MLP_DescAdapterFusion", ratio, "test_rmse")
            adaptive_mean, adaptive_std, _ = metric_mean_std(
                adaptive_metrics,
                dataset,
                "ECFP4_MLP_DescAdapterFusion",
                ratio,
                "test_rmse",
            )
            best_row = best_fixed_df[
                (best_fixed_df["dataset"] == dataset)
                & (best_fixed_df["train_ratio_tag"] == ratio)
            ].iloc[0]

            row["Distilled λ=1.0"] = fmt_metric(fixed_mean, fixed_std)
            row["Best fixed λ"] = f"{best_row['distilled_adapter_rmse']} ({best_row['lambda_distill']})"
            row["Adaptive λ"] = fmt_metric(adaptive_mean, adaptive_std)
            row["Δ Distilled λ=1.0 vs AdapterFusion"] = (
                f"{fixed_mean - means['ECFP4_MLP_DescAdapterFusion']:.4f}"
                if fixed_mean is not None and means["ECFP4_MLP_DescAdapterFusion"] is not None
                else ""
            )
            row["Δ Adaptive vs AdapterFusion"] = (
                f"{adaptive_mean - means['ECFP4_MLP_DescAdapterFusion']:.4f}"
                if adaptive_mean is not None and means["ECFP4_MLP_DescAdapterFusion"] is not None
                else ""
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_lambda_ablation_table(lambda_comparison):
    cols = [
        "dataset",
        "train_ratio_tag",
        "lambda_distill",
        "base_adapter_rmse",
        "distilled_adapter_rmse",
        "teacher_ecfp_desc_rf_rmse",
        "distill_minus_adapter_rmse",
        "distill_minus_teacher_rmse",
    ]
    out = lambda_comparison[cols].copy()
    out = out.rename(columns={
        "train_ratio_tag": "train_ratio",
        "base_adapter_rmse": "AdapterFusion",
        "distilled_adapter_rmse": "Distilled AdapterFusion",
        "teacher_ecfp_desc_rf_rmse": "ECFP4+Desc RF teacher",
        "distill_minus_adapter_rmse": "Δ vs AdapterFusion",
        "distill_minus_teacher_rmse": "Gap to teacher",
    })
    for col in ["Δ vs AdapterFusion", "Gap to teacher"]:
        out[col] = out[col].map(lambda value: f"{float(value):.4f}")
    return out.sort_values(["dataset", "train_ratio", "lambda_distill"], kind="stable")


def build_classification_main_table(base_metrics, ratios):
    datasets = sorted(base_metrics[base_metrics["task_type"] == "classification"]["dataset"].unique())
    rows = []
    for dataset in datasets:
        for ratio in ratios:
            row = {"dataset": dataset, "train_ratio": ratio}
            for model in CLASSIFICATION_MODELS:
                mean, std, _ = metric_mean_std(base_metrics, dataset, model, ratio, "test_roc_auc")
                row[MODEL_LABELS[model]] = fmt_metric(mean, std)
            rows.append(row)
    return pd.DataFrame(rows)


def collect_lambda_metrics(lambda_roots):
    frames = []
    for lambda_value, root in lambda_roots.items():
        metrics = collect_metrics(root)
        if metrics.empty:
            continue
        metrics = metrics.copy()
        metrics["lambda_distill_value"] = lambda_value
        frames.append(metrics)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_validation_selected_lambda_table(base_metrics, lambda_metrics, datasets, ratios):
    rows = []
    for dataset in datasets:
        for ratio in ratios:
            setting = lambda_metrics[
                (lambda_metrics["dataset"] == dataset)
                & (lambda_metrics["train_ratio_tag"] == ratio)
                & (lambda_metrics["model"] == "ECFP4_MLP_DescAdapterFusion")
            ].copy()
            if setting.empty:
                continue

            summary = setting.groupby("lambda_distill_value").agg(
                valid_rmse_mean=("valid_rmse", "mean"),
                valid_rmse_std=("valid_rmse", "std"),
                test_rmse_mean=("test_rmse", "mean"),
                test_rmse_std=("test_rmse", "std"),
                n_seeds=("seed", "nunique"),
            ).reset_index()
            selected = summary.loc[summary["valid_rmse_mean"].idxmin()]
            fixed = summary[summary["lambda_distill_value"] == "1.0"].iloc[0]
            base_mean, base_std, _ = metric_mean_std(
                base_metrics,
                dataset,
                "ECFP4_MLP_DescAdapterFusion",
                ratio,
                "test_rmse",
            )

            rows.append({
                "dataset": dataset,
                "train_ratio": ratio,
                "selected_lambda_by_valid": selected["lambda_distill_value"],
                "selected_valid_rmse": fmt_metric(selected["valid_rmse_mean"], selected["valid_rmse_std"]),
                "selected_test_rmse": fmt_metric(selected["test_rmse_mean"], selected["test_rmse_std"]),
                "fixed_lambda_1_test_rmse": fmt_metric(fixed["test_rmse_mean"], fixed["test_rmse_std"]),
                "base_adapter_test_rmse": fmt_metric(base_mean, base_std),
                "selected_minus_fixed_1": f"{selected['test_rmse_mean'] - fixed['test_rmse_mean']:.4f}",
                "selected_minus_base": f"{selected['test_rmse_mean'] - base_mean:.4f}",
                "selected_beats_fixed_1": bool(selected["test_rmse_mean"] < fixed["test_rmse_mean"]),
                "selected_beats_base": bool(selected["test_rmse_mean"] < base_mean),
                "n_seeds": int(selected["n_seeds"]),
            })
    return pd.DataFrame(rows)


def write_table(df, output_root, stem):
    csv_path = output_root / f"{stem}.csv"
    md_path = output_root / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(df), encoding="utf-8")
    return csv_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(description="Build paper-ready distillation result tables.")
    parser.add_argument("--base-results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--fixed-lambda-1-root", default=str(PROJECT_ROOT / "results_distill_lambda" / "lambda_1_0"))
    parser.add_argument("--adaptive-results-root", default=str(PROJECT_ROOT / "results_distill_adaptive_regression"))
    parser.add_argument(
        "--lambda-comparison",
        default=str(PROJECT_ROOT / "results_distill_lambda" / "summary" / "distillation_lambda_comparison.csv"),
    )
    parser.add_argument(
        "--best-fixed-summary",
        default=str(PROJECT_ROOT / "results_distill_lambda" / "summary" / "distillation_lambda_best_by_setting.csv"),
    )
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "paper_tables"))
    parser.add_argument("--train-ratio-tags", nargs="+", type=int, default=[10, 20, 50])
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base_metrics = collect_metrics(args.base_results_root)
    fixed_metrics = collect_metrics(args.fixed_lambda_1_root)
    adaptive_metrics = collect_metrics(args.adaptive_results_root)
    lambda_comparison = pd.read_csv(args.lambda_comparison)
    best_fixed_df = pd.read_csv(args.best_fixed_summary)
    lambda_metrics = collect_lambda_metrics(DEFAULT_LAMBDA_ROOTS)

    regression_main = build_regression_main_table(
        base_metrics=base_metrics,
        fixed_metrics=fixed_metrics,
        adaptive_metrics=adaptive_metrics,
        best_fixed_df=best_fixed_df,
        datasets=REGRESSION_DATASETS,
        ratios=args.train_ratio_tags,
    )
    lambda_ablation = build_lambda_ablation_table(lambda_comparison)
    lambda_summary = pd.read_csv(PROJECT_ROOT / "results_distill_lambda" / "summary" / "distillation_lambda_summary.csv")
    adaptive_summary = pd.read_csv(PROJECT_ROOT / "results_distill_adaptive_regression" / "summary" / "adaptive_vs_fixed_summary.csv")
    adaptive_comparison = pd.read_csv(PROJECT_ROOT / "results_distill_adaptive_regression" / "summary" / "adaptive_vs_fixed_comparison.csv")
    classification_main = build_classification_main_table(base_metrics, args.train_ratio_tags)
    validation_selected = build_validation_selected_lambda_table(
        base_metrics=base_metrics,
        lambda_metrics=lambda_metrics,
        datasets=REGRESSION_DATASETS,
        ratios=args.train_ratio_tags,
    )

    outputs = []
    outputs.extend(write_table(regression_main, output_root, "table1_regression_main_rmse"))
    outputs.extend(write_table(lambda_ablation, output_root, "table2_lambda_ablation_rmse"))
    outputs.extend(write_table(lambda_summary, output_root, "table3_lambda_summary"))
    outputs.extend(write_table(adaptive_summary, output_root, "table4_adaptive_summary"))
    outputs.extend(write_table(adaptive_comparison, output_root, "table5_adaptive_vs_fixed"))
    outputs.extend(write_table(validation_selected, output_root, "table6_validation_selected_lambda"))
    outputs.extend(write_table(classification_main, output_root, "tableS1_classification_main_roc_auc"))

    index_lines = [
        "# Paper Tables",
        "",
        "- `table1_regression_main_rmse`: main regression comparison, RMSE lower is better.",
        "- `table2_lambda_ablation_rmse`: fixed distillation lambda ablation.",
        "- `table3_lambda_summary`: aggregate lambda sweep summary.",
        "- `table4_adaptive_summary`: adaptive distillation aggregate negative ablation.",
        "- `table5_adaptive_vs_fixed`: adaptive setting-level comparison.",
        "- `table6_validation_selected_lambda`: lambda selected by validation RMSE, no retraining.",
        "- `tableS1_classification_main_roc_auc`: classification reference table, ROC-AUC higher is better.",
        "",
        "Primary method currently recommended for regression: fixed `lambda_distill = 1.0`.",
        "Adaptive validation-advantage weighting should be reported as a negative ablation, not the main method.",
    ]
    (output_root / "README.md").write_text("\n".join(index_lines), encoding="utf-8")

    print(f"paper_tables_dir: {output_root}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
