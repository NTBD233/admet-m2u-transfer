import argparse
from pathlib import Path

import pandas as pd

from compare_distillation import REGRESSION_DATASETS, summarize_model
from utils.config import PROJECT_ROOT, TRAIN_RATIO_TAGS
from utils.summary import collect_metrics, dataframe_to_markdown


MODEL = "ECFP4_MLP_DescAdapterFusion"


def build_comparison(base_metrics, adaptive_metrics, fixed_metrics, best_fixed_df, datasets, ratios):
    rows = []
    for dataset in datasets:
        for ratio in ratios:
            base_mean, base_std, base_n = summarize_model(base_metrics, dataset, MODEL, ratio)
            adaptive_mean, adaptive_std, adaptive_n = summarize_model(adaptive_metrics, dataset, MODEL, ratio)
            fixed_mean, fixed_std, fixed_n = summarize_model(fixed_metrics, dataset, MODEL, ratio)
            best_row = best_fixed_df[
                (best_fixed_df["dataset"] == dataset)
                & (best_fixed_df["train_ratio_tag"] == ratio)
            ].iloc[0]
            best_fixed_mean = float(str(best_row["distilled_adapter_rmse"]).split("±")[0])

            rows.append({
                "dataset": dataset,
                "train_ratio_tag": ratio,
                "base_adapter_rmse": f"{base_mean:.4f}±{base_std:.4f}",
                "adaptive_rmse": f"{adaptive_mean:.4f}±{adaptive_std:.4f}",
                "fixed_lambda_1_rmse": f"{fixed_mean:.4f}±{fixed_std:.4f}",
                "best_fixed_lambda": best_row["lambda_distill"],
                "best_fixed_rmse": best_row["distilled_adapter_rmse"],
                "adaptive_n_seeds": adaptive_n,
                "adaptive_minus_base": round(adaptive_mean - base_mean, 4),
                "adaptive_minus_fixed_1": round(adaptive_mean - fixed_mean, 4),
                "adaptive_minus_best_fixed": round(adaptive_mean - best_fixed_mean, 4),
                "adaptive_beats_base": bool(adaptive_mean < base_mean),
                "adaptive_beats_fixed_1": bool(adaptive_mean < fixed_mean),
                "adaptive_beats_best_fixed": bool(adaptive_mean < best_fixed_mean),
            })
    return pd.DataFrame(rows)


def summarize(comparison_df):
    total = len(comparison_df)
    return pd.DataFrame([{
        "complete_settings": f"{int((comparison_df['adaptive_n_seeds'] == 3).sum())}/{total}",
        "beats_base": f"{int(comparison_df['adaptive_beats_base'].sum())}/{total}",
        "beats_fixed_lambda_1": f"{int(comparison_df['adaptive_beats_fixed_1'].sum())}/{total}",
        "beats_best_fixed_lambda": f"{int(comparison_df['adaptive_beats_best_fixed'].sum())}/{total}",
        "mean_delta_vs_base": f"{comparison_df['adaptive_minus_base'].mean():.4f}",
        "mean_delta_vs_fixed_1": f"{comparison_df['adaptive_minus_fixed_1'].mean():.4f}",
        "mean_delta_vs_best_fixed": f"{comparison_df['adaptive_minus_best_fixed'].mean():.4f}",
    }])


def write_markdown(summary_df, comparison_df, output_path):
    lines = [
        "# Adaptive Distillation Comparison",
        "",
        "## Summary",
        "",
        dataframe_to_markdown(summary_df),
        "",
        "## Setting-Level Comparison",
        "",
        dataframe_to_markdown(comparison_df),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare adaptive distillation against fixed lambda baselines.")
    parser.add_argument("--base-results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--adaptive-results-root", default=str(PROJECT_ROOT / "results_distill_adaptive_regression"))
    parser.add_argument("--fixed-lambda-1-root", default=str(PROJECT_ROOT / "results_distill_lambda" / "lambda_1_0"))
    parser.add_argument(
        "--best-fixed-summary",
        default=str(PROJECT_ROOT / "results_distill_lambda" / "summary" / "distillation_lambda_best_by_setting.csv"),
    )
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results_distill_adaptive_regression" / "summary"))
    parser.add_argument("--datasets", nargs="+", default=REGRESSION_DATASETS)
    parser.add_argument("--train-ratio-tags", nargs="+", type=int, default=TRAIN_RATIO_TAGS)
    return parser.parse_args()


def main():
    args = parse_args()
    base_metrics = collect_metrics(args.base_results_root)
    adaptive_metrics = collect_metrics(args.adaptive_results_root)
    fixed_metrics = collect_metrics(args.fixed_lambda_1_root)
    best_fixed_df = pd.read_csv(args.best_fixed_summary)

    comparison_df = build_comparison(
        base_metrics=base_metrics,
        adaptive_metrics=adaptive_metrics,
        fixed_metrics=fixed_metrics,
        best_fixed_df=best_fixed_df,
        datasets=args.datasets,
        ratios=args.train_ratio_tags,
    )
    summary_df = summarize(comparison_df)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_path = output_root / "adaptive_vs_fixed_comparison.csv"
    summary_path = output_root / "adaptive_vs_fixed_summary.csv"
    md_path = output_root / "adaptive_vs_fixed_comparison.md"

    comparison_df.to_csv(comparison_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_markdown(summary_df, comparison_df, md_path)

    print(f"Saved comparison: {comparison_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved markdown: {md_path}")


if __name__ == "__main__":
    main()
