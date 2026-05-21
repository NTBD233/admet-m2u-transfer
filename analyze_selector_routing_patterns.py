import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import SEEDS, TRAIN_RATIO_TAGS
from utils.summary import dataframe_to_markdown


DEFAULT_DATASETS = [
    "caco2_wang",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "vdss_lombardo",
    "ppbr_az",
]
DEFAULT_TEACHERS = ["ECFP4_RF", "Desc_RF", "ECFP4_Desc_RF"]


def load_npz(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def load_selector(selector_root, selector_model_name, dataset, ratio, seed, split):
    path = (
        Path(selector_root)
        / dataset
        / selector_model_name
        / f"train_{ratio}"
        / f"seed_{seed}"
        / f"{split}_selector_predictions.npz"
    )
    return load_npz(path)


def load_teacher_pred(teacher_root, dataset, teacher, ratio, seed, split):
    path = (
        Path(teacher_root)
        / dataset
        / teacher
        / f"train_{ratio}"
        / f"seed_{seed}"
        / f"{split}_teacher_predictions.npz"
    )
    return load_npz(path)["pred_raw"].astype(float)


def load_metrics(results_root, dataset, ratio, seed):
    path = (
        Path(results_root)
        / dataset
        / "ECFP4_MLP_DescAdapterFusion"
        / f"train_{ratio}"
        / f"seed_{seed}"
        / "metrics.json"
    )
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def entropy_from_props(props):
    props = np.asarray(props, dtype=float)
    props = props[props > 0]
    if len(props) == 0:
        return 0.0
    return float(-(props * np.log(props)).sum() / np.log(len(DEFAULT_TEACHERS)))


def teacher_weight_top1(metrics, teachers):
    weights = metrics.get("teacher_weight_map") if metrics else None
    if not weights:
        return None
    return teachers[int(np.argmax([weights[teacher] for teacher in teachers]))]


def collect_setting_rows(args):
    rows = []
    for dataset in args.datasets:
        for ratio in args.train_ratio_tags:
            for seed in args.seeds:
                try:
                    selector = load_selector(
                        args.selector_root,
                        args.selector_model_name,
                        dataset,
                        ratio,
                        seed,
                        args.split,
                    )
                    teacher_preds = np.vstack(
                        [
                            load_teacher_pred(args.teacher_root, dataset, teacher, ratio, seed, args.split)
                            for teacher in args.teachers
                        ]
                    ).T
                except FileNotFoundError:
                    continue

                pred_idx = selector["pred_idx"].astype(int)
                oracle_idx = selector["oracle_idx"].astype(int)
                if len(pred_idx) == 0:
                    continue

                props = [(pred_idx == i).mean() for i in range(len(args.teachers))]
                oracle_props = [(oracle_idx == i).mean() for i in range(len(args.teachers))]
                selector_metrics = load_metrics(args.selector_results_root, dataset, ratio, seed)
                selector_decay_metrics = load_metrics(args.selector_decay_results_root, dataset, ratio, seed)
                top1_metrics = load_metrics(args.top1_results_root, dataset, ratio, seed)

                row = {
                    "dataset": dataset,
                    "train_ratio": int(ratio),
                    "seed": int(seed),
                    "n_test": int(len(pred_idx)),
                    "selector_accuracy": float((pred_idx == oracle_idx).mean()),
                    "selector_top1_disagreement_rate": np.nan,
                    "route_entropy": entropy_from_props(props),
                    "teacher_pred_range_mean": float(np.ptp(teacher_preds, axis=1).mean()),
                    "teacher_pred_std_mean": float(teacher_preds.std(axis=1).mean()),
                    "top1_teacher": teacher_weight_top1(top1_metrics, args.teachers),
                }
                if row["top1_teacher"] is not None:
                    top1_idx = args.teachers.index(row["top1_teacher"])
                    row["selector_top1_disagreement_rate"] = float((pred_idx != top1_idx).mean())

                for teacher, prop, oracle_prop in zip(args.teachers, props, oracle_props):
                    row[f"select_{teacher}"] = float(prop)
                    row[f"oracle_{teacher}"] = float(oracle_prop)

                if selector_metrics and top1_metrics:
                    row["selector_rmse"] = float(selector_metrics["test_rmse"])
                    row["top1_rmse"] = float(top1_metrics["test_rmse"])
                    row["selector_delta_vs_top1"] = row["selector_rmse"] - row["top1_rmse"]
                    row["selector_wins_top1"] = row["selector_delta_vs_top1"] < 0
                if selector_decay_metrics and top1_metrics:
                    row["selector_decay_rmse"] = float(selector_decay_metrics["test_rmse"])
                    row["selector_decay_delta_vs_top1"] = row["selector_decay_rmse"] - row["top1_rmse"]
                    row["selector_decay_wins_top1"] = row["selector_decay_delta_vs_top1"] < 0
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_route_mix(setting_df, teachers):
    group_cols = ["dataset", "train_ratio"]
    value_cols = [f"select_{teacher}" for teacher in teachers] + [
        "selector_accuracy",
        "selector_top1_disagreement_rate",
        "route_entropy",
        "teacher_pred_range_mean",
        "teacher_pred_std_mean",
        "selector_delta_vs_top1",
        "selector_decay_delta_vs_top1",
    ]
    out = setting_df.groupby(group_cols, as_index=False)[value_cols].mean()
    for col in value_cols:
        out[col] = out[col].round(4)
    return out.sort_values(group_cols, kind="stable")


def summarize_conflict(setting_df):
    rows = []
    comparisons = [
        ("plain_selector", "selector_wins_top1", "selector_delta_vs_top1"),
        ("selector_decay", "selector_decay_wins_top1", "selector_decay_delta_vs_top1"),
    ]
    for method, win_col, delta_col in comparisons:
        if win_col not in setting_df:
            continue
        for label, sub in setting_df.groupby(setting_df[win_col].map({True: "wins_top1", False: "loses_top1"})):
            rows.append(
                {
                    "method": method,
                    "group": label,
                    "settings": int(len(sub)),
                    "mean_delta_vs_top1": float(sub[delta_col].mean()),
                    "mean_teacher_pred_range": float(sub["teacher_pred_range_mean"].mean()),
                    "mean_teacher_pred_std": float(sub["teacher_pred_std_mean"].mean()),
                    "mean_selector_top1_disagreement": float(sub["selector_top1_disagreement_rate"].mean()),
                    "mean_route_entropy": float(sub["route_entropy"].mean()),
                    "mean_selector_accuracy": float(sub["selector_accuracy"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    numeric_cols = [col for col in out.columns if col not in {"method", "group", "settings"}]
    for col in numeric_cols:
        out[col] = out[col].round(4)
    return out


def write_report(output_root, route_mix, conflict_summary):
    lines = [
        "# Selector Routing Pattern Analysis",
        "",
        "## Route Mix By Dataset And Train Ratio",
        "",
        dataframe_to_markdown(route_mix),
        "",
        "## Conflict And Top-1 Comparison",
        "",
        dataframe_to_markdown(conflict_summary),
        "",
        "## Key Reading",
        "",
        "- `select_*` columns are test-split proportions of samples routed to each teacher, averaged across seeds.",
        "- `teacher_pred_range_mean` measures mean per-sample spread between the largest and smallest teacher predictions.",
        "- `selector_top1_disagreement_rate` measures how often sample-level routing differs from the setting-level top-1 teacher.",
        "- Higher route entropy means the selector uses a more mixed teacher policy instead of collapsing to one teacher.",
    ]
    (output_root / "selector_routing_pattern_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze selector routing choices and conflict patterns.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--teachers", nargs="+", default=DEFAULT_TEACHERS)
    parser.add_argument("--train-ratio-tags", nargs="+", type=int, default=TRAIN_RATIO_TAGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--selector-root", default="data/selector_predictions")
    parser.add_argument("--selector-model-name", default="rf_crossfit_train_pseudo_oracle")
    parser.add_argument("--teacher-root", default="data/teacher_predictions")
    parser.add_argument("--selector-results-root", default="results_pretrained_selector_top1_regression")
    parser.add_argument("--selector-decay-results-root", default="results_pretrained_selector_top1_high_resource_lambda_regression")
    parser.add_argument("--top1-results-root", default="results_multiteacher_top1")
    parser.add_argument("--output-root", default="results_selector_routing_patterns")
    parser.add_argument("--paper-table-root", default="paper_tables_second")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    paper_table_root = Path(args.paper_table_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paper_table_root.mkdir(parents=True, exist_ok=True)

    setting_df = collect_setting_rows(args)
    if setting_df.empty:
        raise SystemExit("No selector routing records found.")

    route_mix = summarize_route_mix(setting_df, args.teachers)
    conflict_summary = summarize_conflict(setting_df)

    setting_df.round(4).to_csv(output_root / "selector_routing_by_setting.csv", index=False)
    route_mix.to_csv(output_root / "selector_route_mix_by_dataset_ratio.csv", index=False)
    conflict_summary.to_csv(output_root / "selector_conflict_win_summary.csv", index=False)
    route_mix.to_csv(paper_table_root / "table7_selector_route_mix.csv", index=False)
    conflict_summary.to_csv(paper_table_root / "table8_selector_conflict_win_summary.csv", index=False)
    (paper_table_root / "table7_selector_route_mix.md").write_text(dataframe_to_markdown(route_mix), encoding="utf-8")
    (paper_table_root / "table8_selector_conflict_win_summary.md").write_text(
        dataframe_to_markdown(conflict_summary),
        encoding="utf-8",
    )
    write_report(output_root, route_mix, conflict_summary)

    print("## Route Mix")
    print(dataframe_to_markdown(route_mix))
    print("## Conflict Summary")
    print(dataframe_to_markdown(conflict_summary))


if __name__ == "__main__":
    main()
