import argparse
from pathlib import Path

import pandas as pd

from utils.config import PROJECT_ROOT
from utils.summary import collect_metrics, dataframe_to_markdown


MODEL = "ECFP4_MLP_DescAdapterFusion"
KEYS = ["dataset", "train_ratio_tag", "seed"]


DEFAULT_METHODS = {
    "base": "results",
    "fixed_ecfp_desc": "results_distill_regression",
    "top1_validation": "results_multiteacher_top1",
    "plain_selector": "results_pretrained_selector_top1_regression",
    "low_decay": "results_pretrained_selector_top1_ratio_lambda_partial_regression",
    "high_decay": "results_pretrained_selector_top1_high_resource_lambda_partial_regression",
}


def parse_method_specs(specs):
    if not specs:
        return DEFAULT_METHODS
    methods = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Method spec must be name=results_root: {spec}")
        name, root = spec.split("=", 1)
        methods[name] = root
    return methods


def load_method_metrics(methods, datasets, ratios, metric):
    frames = []
    for method, root in methods.items():
        df = collect_metrics(PROJECT_ROOT / root)
        if df.empty:
            raise ValueError(f"No metrics found for {method}: {root}")
        df = df[df["model"] == MODEL].copy()
        if datasets:
            df = df[df["dataset"].isin(datasets)]
        if ratios:
            df = df[df["train_ratio_tag"].isin(ratios)]
        missing_cols = [col for col in KEYS + [metric] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{method} is missing columns: {missing_cols}")
        df = df[KEYS + [metric]].rename(columns={metric: method})
        frames.append(df)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=KEYS, how="inner")
    if merged.empty:
        raise ValueError("No overlapping dataset/ratio/seed settings across methods.")
    return merged


def build_aggregate(setting_df, methods):
    rows = []
    for method in methods:
        rows.append(
            {
                "method": method,
                "complete_runs": int(setting_df[method].notna().sum()),
                "mean_test_rmse": round(setting_df[method].mean(), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_test_rmse", kind="stable")


def build_wins(setting_df, target, references):
    rows = []
    for reference in references:
        delta = setting_df[target] - setting_df[reference]
        rows.append(
            {
                "target": target,
                "reference": reference,
                "wins": int((delta < 0).sum()),
                "total": int(delta.notna().sum()),
                "mean_delta": round(delta.mean(), 4),
            }
        )
    return pd.DataFrame(rows)


def build_grouped(setting_df, methods, group_cols):
    rows = []
    for group_values, group_df in setting_df.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_cols, group_values))
        if "train_ratio_tag" in row:
            row["train_ratio_tag"] = int(row["train_ratio_tag"])
        for method in methods:
            row[method] = round(group_df[method].mean(), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def build_setting_table(setting_df, methods):
    out = setting_df.copy()
    out["best_method"] = out[list(methods)].idxmin(axis=1)
    for method in methods:
        out[method] = out[method].round(4)
    return out.sort_values(KEYS)


def write_markdown(output_path, aggregate_df, wins_df, by_ratio_df, by_dataset_df, setting_df):
    lines = [
        "# Selector Calibration Comparison",
        "",
        "## Aggregate",
        "",
        dataframe_to_markdown(aggregate_df),
        "",
        "## Target Wins",
        "",
        dataframe_to_markdown(wins_df),
        "",
        "## By Train Ratio",
        "",
        dataframe_to_markdown(by_ratio_df),
        "",
        "## By Dataset",
        "",
        dataframe_to_markdown(by_dataset_df),
        "",
        "## Per Setting",
        "",
        dataframe_to_markdown(setting_df),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare pretrained selector routing and lambda calibration variants."
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        help="Optional method specs as name=results_root. Defaults to the selector calibration comparison set.",
    )
    parser.add_argument("--target-method", default="high_decay")
    parser.add_argument(
        "--reference-methods",
        nargs="+",
        default=["base", "fixed_ecfp_desc", "top1_validation", "plain_selector", "low_decay"],
    )
    parser.add_argument("--datasets", nargs="+", default=["caco2_wang", "ppbr_az"])
    parser.add_argument("--train-ratio-tags", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--metric", default="test_rmse")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "results_pretrained_selector_top1_high_resource_lambda_partial_regression" / "summary"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    methods = parse_method_specs(args.methods)
    if args.target_method not in methods:
        raise ValueError(f"Unknown target method: {args.target_method}")
    unknown_refs = [name for name in args.reference_methods if name not in methods]
    if unknown_refs:
        raise ValueError(f"Unknown reference methods: {unknown_refs}")

    setting_df = load_method_metrics(
        methods=methods,
        datasets=args.datasets,
        ratios=args.train_ratio_tags,
        metric=args.metric,
    )
    method_names = list(methods)
    aggregate_df = build_aggregate(setting_df, method_names)
    wins_df = build_wins(setting_df, args.target_method, args.reference_methods)
    by_ratio_df = build_grouped(setting_df, method_names, ["train_ratio_tag"])
    by_dataset_df = build_grouped(setting_df, method_names, ["dataset"])
    setting_table = build_setting_table(setting_df, method_names)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    aggregate_df.to_csv(output_root / "selector_calibration_aggregate.csv", index=False)
    wins_df.to_csv(output_root / "selector_calibration_wins.csv", index=False)
    by_ratio_df.to_csv(output_root / "selector_calibration_by_ratio.csv", index=False)
    by_dataset_df.to_csv(output_root / "selector_calibration_by_dataset.csv", index=False)
    setting_table.to_csv(output_root / "selector_calibration_by_setting.csv", index=False)
    write_markdown(
        output_root / "selector_calibration_comparison.md",
        aggregate_df=aggregate_df,
        wins_df=wins_df,
        by_ratio_df=by_ratio_df,
        by_dataset_df=by_dataset_df,
        setting_df=setting_table,
    )

    print(f"Saved selector calibration comparison to: {output_root}")
    print(dataframe_to_markdown(aggregate_df))


if __name__ == "__main__":
    main()
