import argparse
import numbers
from pathlib import Path

import pandas as pd

from utils.config import RESULTS_ROOT, SEEDS


PRIMARY_MODELS = [
    "ECFP4_MLP",
    "ECFP4_MLP_DescPred",
    "ECFP4_MLP_DescAdapterFusion",
    "ECFP4_MLP_DescConcat",
]


def metric_for_task(task_type):
    if task_type == "classification":
        return "test_roc_auc_mean", "higher"
    return "test_rmse_mean", "lower"


def fmt(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def complete_summary(summary_df, all_metrics_df, min_seeds):
    count_df = (
        all_metrics_df.groupby(["dataset", "task_type", "model", "train_ratio_tag"])
        .size()
        .reset_index(name="n_seeds")
    )
    merged = summary_df.merge(
        count_df,
        on=["dataset", "task_type", "model", "train_ratio_tag"],
        how="left",
    )
    return merged[merged["n_seeds"] >= min_seeds].copy()


def build_model_comparison(df):
    rows = []
    primary_df = df[df["model"].isin(PRIMARY_MODELS)].copy()
    for (dataset, task_type, ratio), group in primary_df.groupby(
        ["dataset", "task_type", "train_ratio_tag"]
    ):
        metric, direction = metric_for_task(task_type)
        if metric not in group.columns:
            continue
        pivot = group.set_index("model")
        if "ECFP4_MLP" not in pivot.index:
            continue

        base = pivot.loc["ECFP4_MLP", metric]
        descpred = pivot.loc["ECFP4_MLP_DescPred", metric] if "ECFP4_MLP_DescPred" in pivot.index else pd.NA
        adapter = (
            pivot.loc["ECFP4_MLP_DescAdapterFusion", metric]
            if "ECFP4_MLP_DescAdapterFusion" in pivot.index
            else pd.NA
        )
        concat = pivot.loc["ECFP4_MLP_DescConcat", metric] if "ECFP4_MLP_DescConcat" in pivot.index else pd.NA

        def improvement(candidate, reference):
            if pd.isna(candidate) or pd.isna(reference):
                return pd.NA
            if direction == "higher":
                return candidate - reference
            return reference - candidate

        rows.append({
            "dataset": dataset,
            "task_type": task_type,
            "train_ratio_tag": ratio,
            "metric": metric.replace("_mean", ""),
            "ecfp_mlp": base,
            "descpred": descpred,
            "adapter": adapter,
            "descconcat": concat,
            "descpred_vs_ecfp": improvement(descpred, base),
            "adapter_vs_ecfp": improvement(adapter, base),
            "adapter_vs_descpred": improvement(adapter, descpred),
            "descconcat_vs_ecfp": improvement(concat, base),
        })
    return pd.DataFrame(rows)


def dataframe_to_md(df):
    if df.empty:
        return "(no complete rows yet)\n"
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(format_cell)
    headers = list(out.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines) + "\n"


def format_cell(value):
    if pd.isna(value):
        return ""
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        return f"{float(value):.4f}"
    return str(value)


def write_brief(comparison_df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if comparison_df.empty:
        text = "# Interim Findings\n\nNo complete three-seed neural comparisons yet.\n"
        output_path.write_text(text, encoding="utf-8")
        return output_path

    adapter_wins = comparison_df["adapter_vs_descpred"].dropna()
    adapter_positive = int((adapter_wins > 0).sum())
    adapter_total = int(len(adapter_wins))
    concat_gain = comparison_df["descconcat_vs_ecfp"].dropna()
    concat_positive = int((concat_gain > 0).sum())
    concat_total = int(len(concat_gain))

    text = [
        "# Interim Findings",
        "",
        "This file is generated from complete three-seed result groups only.",
        "",
        "## Current Signal",
        "",
        f"- DescConcat improves over ECFP_MLP in {concat_positive}/{concat_total} complete settings.",
        f"- AdapterFusion improves over DescPred in {adapter_positive}/{adapter_total} complete settings.",
        "- Positive values in improvement columns mean better performance.",
        "",
        "## Neural Model Comparison",
        "",
        dataframe_to_md(comparison_df.sort_values(["dataset", "train_ratio_tag"])),
    ]
    output_path.write_text("\n".join(text), encoding="utf-8")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Write an interim research brief from completed results.")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--min-seeds", type=int, default=len(SEEDS))
    parser.add_argument("--output", default="paper_notes/interim_findings.md")
    return parser.parse_args()


def main():
    args = parse_args()
    summary_path = Path(args.results_root) / "summary" / "mean_std_summary.csv"
    all_metrics_path = Path(args.results_root) / "summary" / "all_seed_metrics.csv"
    if not summary_path.exists() or not all_metrics_path.exists():
        raise SystemExit("Run evaluate.py before write_research_brief.py")

    summary_df = pd.read_csv(summary_path)
    all_metrics_df = pd.read_csv(all_metrics_path)
    complete_df = complete_summary(summary_df, all_metrics_df, args.min_seeds)
    comparison_df = build_model_comparison(complete_df)
    path = write_brief(comparison_df, args.output)
    print(f"wrote: {path}")


if __name__ == "__main__":
    main()
