import argparse
from pathlib import Path

import pandas as pd

from utils.config import RESULTS_ROOT
from utils.summary import build_low_resource_table, build_main_table, collect_metrics, dataframe_to_markdown, save_summaries


def collect_ablation(results_lambda_root):
    root = Path(results_lambda_root)
    frames = []
    for metrics_path in sorted(root.glob("lambda_*/*/*/*/metrics.json")):
        frames.append(pd.read_json(metrics_path, typ="series").to_frame().T)
    for metrics_path in sorted(root.glob("lambda_*/*/*/seed_*/metrics.json")):
        frames.append(pd.read_json(metrics_path, typ="series").to_frame().T)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Build paper-ready result tables from existing metrics.")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--lambda-root", default="results_lambda")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_df = collect_metrics(args.results_root)
    if metrics_df.empty:
        print(f"No metrics found under: {args.results_root}")
        return

    paths = save_summaries(metrics_df, args.results_root)
    summary_df = pd.read_csv(paths["mean_std_csv"])
    summary_dir = Path(args.results_root) / "summary"

    main_table = build_main_table(summary_df)
    low_resource_table = build_low_resource_table(summary_df)
    main_table.to_csv(summary_dir / "main_table.csv", index=False)
    low_resource_table.to_csv(summary_dir / "low_resource_table.csv", index=False)
    (summary_dir / "main_table.md").write_text(dataframe_to_markdown(main_table), encoding="utf-8")
    (summary_dir / "low_resource_table.md").write_text(
        dataframe_to_markdown(low_resource_table),
        encoding="utf-8",
    )

    ablation_df = collect_metrics(args.lambda_root)
    if not ablation_df.empty:
        from utils.summary import build_mean_std_summary

        ablation_summary = build_mean_std_summary(ablation_df)
        ablation_summary.to_csv(summary_dir / "ablation_table.csv", index=False)
        (summary_dir / "ablation_table.md").write_text(
            dataframe_to_markdown(build_main_table(ablation_summary)),
            encoding="utf-8",
        )

    print(f"paper_summary_dir: {summary_dir}")


if __name__ == "__main__":
    main()
