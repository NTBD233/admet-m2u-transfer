import argparse
from pathlib import Path

import pandas as pd

from utils.config import PROJECT_ROOT
from utils.summary import collect_metrics, dataframe_to_markdown, format_mean_std


MODEL = "ECFP4_MLP_DescAdapterFusion"
KEYS = ["dataset", "train_ratio_tag", "seed"]
DATASETS = [
    "caco2_wang",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "vdss_lombardo",
    "ppbr_az",
]
EXTENSION_DATASETS = [
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
    "half_life_obach",
    "ld50_zhu",
]
RATIOS = [10, 20, 50]

MAIN_METHODS = {
    "Base AdapterFusion": "results",
    "Fixed ECFP4+Desc RF": "results_distill_regression",
    "Uniform multi-teacher": "results_multiteacher_uniform",
    "Validation-weighted": "results_multiteacher_validation",
    "Top-1 validation": "results_multiteacher_top1",
    "Pretrained selector": "results_pretrained_selector_top1_regression",
    "Selector + high-resource decay": "results_pretrained_selector_top1_high_resource_lambda_regression",
}

EXTENSION_METHODS = {
    "Base AdapterFusion": "results_extension_base_adapterfusion",
    "Fixed ECFP4+Desc RF": "results_distill_extension_regression",
    "Top-1 validation": "results_multiteacher_top1_extension_regression",
    "Pretrained selector": "results_pretrained_selector_top1_extension_regression",
    "Selector + high-resource decay": "results_pretrained_selector_top1_high_resource_lambda_extension_regression",
}

PARTIAL_METHODS = {
    "Plain selector": "results_pretrained_selector_top1_regression",
    "Auto confidence reweight": "results_pretrained_selector_top1_auto_reweight_partial_regression",
    "Low-resource decay": "results_pretrained_selector_top1_ratio_lambda_partial_regression",
    "High-resource decay": "results_pretrained_selector_top1_high_resource_lambda_partial_regression",
}


def load_method_frame(method_name, root, datasets=None, ratios=None):
    df = collect_metrics(PROJECT_ROOT / root)
    if df.empty:
        raise ValueError(f"No metrics found for {method_name}: {root}")
    df = df[df["model"] == MODEL].copy()
    if datasets is not None:
        df = df[df["dataset"].isin(datasets)]
    if ratios is not None:
        df = df[df["train_ratio_tag"].isin(ratios)]
    if df.empty:
        raise ValueError(f"No matching rows found for {method_name}: {root}")
    return df


def merge_methods(method_roots, datasets=None, ratios=None):
    frames = []
    for method, root in method_roots.items():
        df = load_method_frame(method, root, datasets=datasets, ratios=ratios)
        frames.append(df[KEYS + ["test_rmse"]].rename(columns={"test_rmse": method}))
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=KEYS, how="inner")
    if merged.empty:
        raise ValueError("No overlapping settings across methods.")
    return merged


def mean_std_for(df, dataset, ratio):
    rows = df[(df["dataset"] == dataset) & (df["train_ratio_tag"] == ratio)]
    if rows.empty:
        return ""
    return format_mean_std(rows["test_rmse"].mean(), rows["test_rmse"].std(ddof=1))


def build_main_aggregate(method_roots):
    merged = merge_methods(method_roots, datasets=DATASETS, ratios=RATIOS)
    rows = []
    references = {
        "base": "Base AdapterFusion",
        "fixed": "Fixed ECFP4+Desc RF",
        "top1": "Top-1 validation",
        "selector": "Pretrained selector",
    }
    for method in method_roots:
        row = {
            "method": method,
            "runs": int(merged[method].notna().sum()),
            "mean_test_rmse": f"{merged[method].mean():.4f}",
        }
        for label, reference in references.items():
            if method == reference:
                row[f"wins_{label}"] = ""
                row[f"delta_{label}"] = ""
                continue
            delta = merged[method] - merged[reference]
            row[f"wins_{label}"] = f"{int((delta < 0).sum())}/{len(delta)}"
            row[f"delta_{label}"] = f"{delta.mean():.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def build_main_setting_table(method_roots):
    rows = []
    method_frames = {
        method: load_method_frame(method, root, datasets=DATASETS, ratios=RATIOS)
        for method, root in method_roots.items()
    }
    for dataset in DATASETS:
        for ratio in RATIOS:
            row = {"dataset": dataset, "train_ratio": ratio}
            means = {}
            for method, df in method_frames.items():
                setting = df[(df["dataset"] == dataset) & (df["train_ratio_tag"] == ratio)]
                means[method] = setting["test_rmse"].mean()
                row[method] = format_mean_std(means[method], setting["test_rmse"].std(ddof=1))
            row["best_method"] = min(means, key=means.get)
            row["delta_selector_decay_vs_plain"] = f"{means['Selector + high-resource decay'] - means['Pretrained selector']:.4f}"
            row["delta_selector_decay_vs_top1"] = f"{means['Selector + high-resource decay'] - means['Top-1 validation']:.4f}"
            rows.append(row)
    return pd.DataFrame(rows)


def build_extension_aggregate(method_roots):
    merged = merge_methods(method_roots, datasets=EXTENSION_DATASETS, ratios=RATIOS)
    rows = []
    references = {
        "base": "Base AdapterFusion",
        "fixed": "Fixed ECFP4+Desc RF",
        "top1": "Top-1 validation",
        "selector": "Pretrained selector",
    }
    for method in method_roots:
        row = {
            "method": method,
            "runs": int(merged[method].notna().sum()),
            "mean_test_rmse": f"{merged[method].mean():.4f}",
        }
        for label, reference in references.items():
            if method == reference:
                row[f"wins_{label}"] = ""
                row[f"delta_{label}"] = ""
                continue
            delta = merged[method] - merged[reference]
            row[f"wins_{label}"] = f"{int((delta < 0).sum())}/{len(delta)}"
            row[f"delta_{label}"] = f"{delta.mean():.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def build_extension_setting_table(method_roots):
    rows = []
    method_frames = {
        method: load_method_frame(method, root, datasets=EXTENSION_DATASETS, ratios=RATIOS)
        for method, root in method_roots.items()
    }
    for dataset in EXTENSION_DATASETS:
        for ratio in RATIOS:
            row = {"dataset": dataset, "train_ratio": ratio}
            means = {}
            for method, df in method_frames.items():
                setting = df[(df["dataset"] == dataset) & (df["train_ratio_tag"] == ratio)]
                means[method] = setting["test_rmse"].mean()
                row[method] = format_mean_std(means[method], setting["test_rmse"].std(ddof=1))
            row["best_method"] = min(means, key=means.get)
            row["delta_selector_vs_base"] = f"{means['Pretrained selector'] - means['Base AdapterFusion']:.4f}"
            row["delta_selector_vs_fixed"] = f"{means['Pretrained selector'] - means['Fixed ECFP4+Desc RF']:.4f}"
            row["delta_selector_vs_top1"] = f"{means['Pretrained selector'] - means['Top-1 validation']:.4f}"
            row["delta_decay_vs_selector"] = f"{means['Selector + high-resource decay'] - means['Pretrained selector']:.4f}"
            rows.append(row)
    return pd.DataFrame(rows)


def collect_selector_metrics(selector_root, selector_model_name):
    rows = []
    root = PROJECT_ROOT / selector_root
    for path in sorted(root.glob(f"*/{selector_model_name}/train_*/seed_*/metrics.json")):
        rows.append(pd.read_json(path, typ="series").to_dict())
    if not rows:
        raise ValueError(f"No selector metrics found under {root}")
    return pd.DataFrame(rows)


def build_selector_diagnostics(selector_root, selector_model_name):
    selector_df = collect_selector_metrics(selector_root, selector_model_name)
    selector_df = selector_df[selector_df["dataset"].isin(DATASETS)].copy()
    rows = [
        {
            "diagnostic": "RF selector validation accuracy",
            "scope": "45 dataset-ratio-seed settings",
            "value": f"{selector_df['valid_accuracy'].mean():.4f}",
        },
        {
            "diagnostic": "RF selector test accuracy",
            "scope": "45 dataset-ratio-seed settings",
            "value": f"{selector_df['test_accuracy'].mean():.4f}",
        },
        {
            "diagnostic": "RF selector validation macro-F1",
            "scope": "45 dataset-ratio-seed settings",
            "value": f"{selector_df['valid_macro_f1'].mean():.4f}",
        },
        {
            "diagnostic": "RF selector test macro-F1",
            "scope": "45 dataset-ratio-seed settings",
            "value": f"{selector_df['test_macro_f1'].mean():.4f}",
        },
    ]

    gate_summary_path = PROJECT_ROOT / "results_gate_targets" / "summary" / "gate_target_cv_summary.csv"
    if gate_summary_path.exists():
        gate_df = pd.read_csv(gate_summary_path)
        for _, row in gate_df.iterrows():
            rows.append(
                {
                    "diagnostic": f"{row['model']} weighted accuracy",
                    "scope": f"{int(row['groups'])} leave-one-setting-out groups",
                    "value": f"{float(row['weighted_accuracy']):.4f}",
                }
            )
    return pd.DataFrame(rows)


def build_selector_dataset_table(selector_root, selector_model_name):
    selector_df = collect_selector_metrics(selector_root, selector_model_name)
    selector_df = selector_df[selector_df["dataset"].isin(DATASETS)].copy()
    rows = []
    for dataset, sub in selector_df.groupby("dataset", sort=True):
        rows.append(
            {
                "dataset": dataset,
                "valid_accuracy": f"{sub['valid_accuracy'].mean():.4f}",
                "test_accuracy": f"{sub['test_accuracy'].mean():.4f}",
                "valid_macro_f1": f"{sub['valid_macro_f1'].mean():.4f}",
                "test_macro_f1": f"{sub['test_macro_f1'].mean():.4f}",
            }
        )
    return pd.DataFrame(rows)


def build_partial_ablation_table():
    merged = merge_methods(PARTIAL_METHODS, datasets=["caco2_wang", "ppbr_az"], ratios=RATIOS)
    rows = []
    plain = "Plain selector"
    for method in PARTIAL_METHODS:
        row = {
            "method": method,
            "scope": "caco2_wang + ppbr_az, 18 runs",
            "mean_test_rmse": f"{merged[method].mean():.4f}",
        }
        if method != plain:
            delta = merged[method] - merged[plain]
            row["wins_vs_plain_selector"] = f"{int((delta < 0).sum())}/{len(delta)}"
            row["delta_vs_plain_selector"] = f"{delta.mean():.4f}"
        else:
            row["wins_vs_plain_selector"] = ""
            row["delta_vs_plain_selector"] = ""
        rows.append(row)

    route_summary = PROJECT_ROOT / "results_selector_calibration_route_audit" / "selector_route_audit_summary.csv"
    if route_summary.exists():
        route_df = pd.read_csv(route_summary)
        all_row = route_df[route_df["dataset"] == "ALL"]
        if not all_row.empty:
            row = all_row.iloc[0]
            rows.append(
                {
                    "method": "Auto route-mode audit",
                    "scope": "caco2_wang + ppbr_az, 18 settings",
                    "mean_test_rmse": "",
                    "wins_vs_plain_selector": f"{int(row['auto_wins_vs_plain'])}/{int(row['settings'])}",
                    "delta_vs_plain_selector": f"{float(row['mean_delta_vs_plain']):.4f}",
                }
            )
    return pd.DataFrame(rows)


def write_table(df, output_root, stem):
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{stem}.csv"
    md_path = output_root / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(df), encoding="utf-8")
    return [csv_path, md_path]


def write_readme(output_root):
    lines = [
        "# Second Paper Tables",
        "",
        "- `table1_main_aggregate_rmse`: full 45-run aggregate comparison.",
        "- `table2_main_by_setting_rmse`: dataset x train-ratio RMSE table.",
        "- `table3_selector_diagnostics`: selector predictability and gate-target diagnostics.",
        "- `table4_selector_quality_by_dataset`: selector accuracy by endpoint.",
        "- `table5_secondary_ablation_summary`: partial secondary/negative ablations.",
        "- `table6_failure_mode_probe`: setting-level probes separating teacher-selection",
        "  quality from student utilization of routed teacher supervision.",
        "- `table7_selector_route_mix`: test-split selector routing proportions by",
        "  dataset and train ratio.",
        "- `table8_selector_conflict_win_summary`: teacher-conflict summaries grouped by",
        "  whether selector variants beat the setting-level top-1 teacher baseline.",
        "- `table9_extension_aggregate_rmse`: four-endpoint extension-panel aggregate.",
        "- `table10_extension_by_setting_rmse`: extension-panel dataset x train-ratio RMSE table.",
        "",
        "Lower RMSE is better. Deltas are target minus reference, so negative is better.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Build paper-ready tables for the second ADMET selector paper.")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "paper_tables_second"))
    parser.add_argument("--selector-root", default="data/selector_predictions")
    parser.add_argument("--selector-model-name", default="rf_crossfit_train_pseudo_oracle")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    outputs = []
    outputs.extend(write_table(build_main_aggregate(MAIN_METHODS), output_root, "table1_main_aggregate_rmse"))
    outputs.extend(write_table(build_main_setting_table(MAIN_METHODS), output_root, "table2_main_by_setting_rmse"))
    outputs.extend(write_table(build_extension_aggregate(EXTENSION_METHODS), output_root, "table9_extension_aggregate_rmse"))
    outputs.extend(write_table(build_extension_setting_table(EXTENSION_METHODS), output_root, "table10_extension_by_setting_rmse"))
    outputs.extend(
        write_table(
            build_selector_diagnostics(args.selector_root, args.selector_model_name),
            output_root,
            "table3_selector_diagnostics",
        )
    )
    outputs.extend(
        write_table(
            build_selector_dataset_table(args.selector_root, args.selector_model_name),
            output_root,
            "table4_selector_quality_by_dataset",
        )
    )
    outputs.extend(write_table(build_partial_ablation_table(), output_root, "table5_secondary_ablation_summary"))
    write_readme(output_root)

    print(f"second_paper_tables_dir: {output_root}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
