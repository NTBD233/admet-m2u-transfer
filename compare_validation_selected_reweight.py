import argparse
from pathlib import Path

import pandas as pd

from utils.config import PROJECT_ROOT
from utils.summary import collect_metrics, dataframe_to_markdown


MODEL = "ECFP4_MLP_DescAdapterFusion"
KEYS = ["dataset", "train_ratio_tag", "seed"]


DEFAULT_METHODS = {
    "plain_selector": "results_pretrained_selector_top1_regression",
    "global_confidence": [
        "results_pretrained_selector_top1_reweight_caco2_regression",
        "results_pretrained_selector_top1_reweight_ppbr_regression",
    ],
    "disagreement_confidence": [
        "results_pretrained_selector_top1_disagreement_reweight_caco2_regression",
        "results_pretrained_selector_top1_disagreement_reweight_ppbr_regression",
    ],
    "top1_validation": "results_multiteacher_top1",
}


def iter_roots(root_spec):
    if isinstance(root_spec, (list, tuple)):
        return [PROJECT_ROOT / root for root in root_spec]
    return [PROJECT_ROOT / root_spec]


def load_method(method, root_spec, datasets, ratios):
    frames = []
    for root in iter_roots(root_spec):
        df = collect_metrics(root)
        if df.empty:
            continue
        df = df[df["model"] == MODEL].copy()
        if datasets:
            df = df[df["dataset"].isin(datasets)]
        if ratios:
            df = df[df["train_ratio_tag"].isin(ratios)]
        if not df.empty:
            frames.append(df)
    if not frames:
        raise ValueError(f"No metrics found for {method}: {root_spec}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(KEYS, keep="last")
    return df[KEYS + ["valid_rmse", "test_rmse"]].rename(
        columns={
            "valid_rmse": f"{method}__valid_rmse",
            "test_rmse": f"{method}__test_rmse",
        }
    )


def build_merged(methods, datasets, ratios):
    frames = [load_method(method, root, datasets, ratios) for method, root in methods.items()]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=KEYS, how="inner")
    if merged.empty:
        raise ValueError("No overlapping settings across methods.")
    return merged


def add_validation_selected_columns(df, candidate_methods):
    out = df.copy()
    valid_cols = [f"{method}__valid_rmse" for method in candidate_methods]
    test_cols = {method: f"{method}__test_rmse" for method in candidate_methods}
    out["selected_reweight_mode"] = out[valid_cols].idxmin(axis=1).str.replace("__valid_rmse", "", regex=False)
    out["selected_valid_rmse"] = [
        row[f"{row['selected_reweight_mode']}__valid_rmse"] for _, row in out.iterrows()
    ]
    out["selected_test_rmse"] = [
        row[test_cols[row["selected_reweight_mode"]]] for _, row in out.iterrows()
    ]
    return out


def aggregate(df, methods):
    rows = [
        {
            "method": "validation_selected_reweight",
            "runs": len(df),
            "mean_test_rmse": round(float(df["selected_test_rmse"].mean()), 4),
            "mean_valid_rmse": round(float(df["selected_valid_rmse"].mean()), 4),
        }
    ]
    for method in methods:
        rows.append(
            {
                "method": method,
                "runs": len(df),
                "mean_test_rmse": round(float(df[f"{method}__test_rmse"].mean()), 4),
                "mean_valid_rmse": round(float(df[f"{method}__valid_rmse"].mean()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_test_rmse", kind="stable")


def wins(df, target_col, reference_methods):
    rows = []
    for method in reference_methods:
        delta = df[target_col] - df[f"{method}__test_rmse"]
        rows.append(
            {
                "target": target_col.replace("_test_rmse", ""),
                "reference": method,
                "wins": int((delta < 0).sum()),
                "total": int(delta.notna().sum()),
                "mean_delta": round(float(delta.mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def mode_counts(df):
    return (
        df.groupby(["dataset", "selected_reweight_mode"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["dataset", "selected_reweight_mode"])
    )


def by_group(df, group_cols, methods):
    rows = []
    for group_values, sub in df.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_cols, group_values))
        row["validation_selected_reweight"] = round(float(sub["selected_test_rmse"].mean()), 4)
        for method in methods:
            row[method] = round(float(sub[f"{method}__test_rmse"].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(path, aggregate_df, wins_df, mode_counts_df, by_dataset_df, by_ratio_df, setting_df):
    lines = [
        "# Validation-Selected Reweight Comparison",
        "",
        "Candidate modes are selected by validation RMSE within each dataset, ratio, and seed.",
        "",
        "## Aggregate",
        "",
        dataframe_to_markdown(aggregate_df),
        "",
        "## Wins",
        "",
        dataframe_to_markdown(wins_df),
        "",
        "## Selected Mode Counts",
        "",
        dataframe_to_markdown(mode_counts_df),
        "",
        "## By Dataset",
        "",
        dataframe_to_markdown(by_dataset_df),
        "",
        "## By Train Ratio",
        "",
        dataframe_to_markdown(by_ratio_df),
        "",
        "## Per Setting",
        "",
        dataframe_to_markdown(setting_df),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare validation-selected selector reweight modes.")
    parser.add_argument("--datasets", nargs="+", default=["caco2_wang", "ppbr_az"])
    parser.add_argument("--train-ratio-tags", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "results_validation_selected_reweight_partial_regression" / "summary"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    methods = DEFAULT_METHODS
    candidate_methods = ["plain_selector", "global_confidence", "disagreement_confidence"]
    merged = build_merged(methods, args.datasets, args.train_ratio_tags)
    selected = add_validation_selected_columns(merged, candidate_methods)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate_df = aggregate(selected, list(methods))
    wins_df = wins(selected, "selected_test_rmse", list(methods))
    mode_counts_df = mode_counts(selected)
    by_dataset_df = by_group(selected, ["dataset"], list(methods))
    by_ratio_df = by_group(selected, ["train_ratio_tag"], list(methods))

    setting_cols = (
        KEYS
        + ["selected_reweight_mode", "selected_valid_rmse", "selected_test_rmse"]
        + [f"{method}__test_rmse" for method in methods]
    )
    setting_df = selected[setting_cols].sort_values(KEYS).copy()
    for col in setting_df.columns:
        if col.endswith("rmse"):
            setting_df[col] = setting_df[col].round(4)

    aggregate_df.to_csv(output_root / "validation_selected_reweight_aggregate.csv", index=False)
    wins_df.to_csv(output_root / "validation_selected_reweight_wins.csv", index=False)
    mode_counts_df.to_csv(output_root / "validation_selected_reweight_mode_counts.csv", index=False)
    by_dataset_df.to_csv(output_root / "validation_selected_reweight_by_dataset.csv", index=False)
    by_ratio_df.to_csv(output_root / "validation_selected_reweight_by_ratio.csv", index=False)
    setting_df.to_csv(output_root / "validation_selected_reweight_by_setting.csv", index=False)
    write_markdown(
        output_root / "validation_selected_reweight_comparison.md",
        aggregate_df=aggregate_df,
        wins_df=wins_df,
        mode_counts_df=mode_counts_df,
        by_dataset_df=by_dataset_df,
        by_ratio_df=by_ratio_df,
        setting_df=setting_df,
    )

    print(f"Saved validation-selected reweight comparison to: {output_root}")
    print(dataframe_to_markdown(aggregate_df))


if __name__ == "__main__":
    main()
