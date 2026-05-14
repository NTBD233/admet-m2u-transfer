import argparse
from pathlib import Path

import pandas as pd

from utils.summary import dataframe_to_markdown


def load_metrics(results_root):
    root = Path(results_root)
    rows = []
    for metrics_path in root.glob("*/ECFP4_MLP_DescAdapterFusion/train_*/seed_*/metrics.json"):
        rows.append(pd.read_json(metrics_path, typ="series").to_dict())
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summarize(results_map):
    summary_rows = []
    method_frames = {}
    for method_name, root in results_map.items():
        df = load_metrics(root)
        if df.empty:
            continue
        method_frames[method_name] = df
        summary_rows.append(
            {
                "method": method_name,
                "completed_settings": df[["dataset", "train_ratio_tag", "seed"]].drop_duplicates().shape[0],
                "mean_test_rmse": round(df["test_rmse"].mean(), 4),
                "mean_valid_rmse": round(df["valid_rmse"].mean(), 4),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_test_rmse", kind="stable")

    comparison_rows = []
    if "base" in method_frames:
        base = method_frames["base"][["dataset", "train_ratio_tag", "seed", "test_rmse"]].rename(columns={"test_rmse": "base_test_rmse"})
        for method_name, df in method_frames.items():
            if method_name == "base":
                continue
            merged = df.merge(base, on=["dataset", "train_ratio_tag", "seed"], how="inner")
            if merged.empty:
                continue
            comparison_rows.append(
                {
                    "method": method_name,
                    "beats_base": int((merged["test_rmse"] < merged["base_test_rmse"]).sum()),
                    "total_runs": int(len(merged)),
                    "mean_delta_vs_base": round((merged["test_rmse"] - merged["base_test_rmse"]).mean(), 4),
                }
            )
    comparison_df = pd.DataFrame(comparison_rows)
    return summary_df, comparison_df


def parse_args():
    parser = argparse.ArgumentParser(description="Compare pretrained selector distillation runs.")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Pairs of method_name=results_root",
    )
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    results_map = {}
    for item in args.results:
        method_name, root = item.split("=", 1)
        results_map[method_name] = root

    summary_df, comparison_df = summarize(results_map)
    print("## Summary")
    print(dataframe_to_markdown(summary_df))
    if not comparison_df.empty:
        print("\n## Delta vs Base")
        print(dataframe_to_markdown(comparison_df))

    if args.output_root is not None:
        out = Path(args.output_root)
        out.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out / "pretrained_selector_summary.csv", index=False)
        comparison_df.to_csv(out / "pretrained_selector_vs_base.csv", index=False)


if __name__ == "__main__":
    main()
