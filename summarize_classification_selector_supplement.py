import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASSIFICATION_DATASETS = [
    "bbb_martins",
    "hia_hou",
    "pgp_broccatelli",
    "bioavailability_ma",
    "herg",
]


def load_metrics(selector_root, selector_model_name):
    rows = []
    root = Path(selector_root)
    for dataset in CLASSIFICATION_DATASETS:
        dataset_root = root / dataset / selector_model_name
        for metrics_path in sorted(dataset_root.glob("train_*/seed_*/metrics.json")):
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            if metrics.get("task_type") != "classification":
                continue
            out_dir = metrics_path.parent
            row = dict(metrics)
            train_labels = np.load(out_dir / "train_selector_predictions.npz", allow_pickle=True)["oracle_idx"]
            values, counts = np.unique(train_labels, return_counts=True)
            majority_label = int(values[np.argmax(counts)])
            for split in ["valid", "test"]:
                split_data = np.load(out_dir / f"{split}_selector_predictions.npz", allow_pickle=True)
                labels = split_data["oracle_idx"]
                pred = split_data["pred_idx"]
                row[f"{split}_majority_accuracy"] = float((labels == majority_label).mean())
                row[f"{split}_selector_accuracy"] = float((labels == pred).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df):
    columns = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column == "train_ratio_tag":
                values.append(str(int(value)))
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize classification selector supplementary diagnostics.")
    parser.add_argument("--selector-root", default="data/selector_predictions")
    parser.add_argument("--selector-model-name", default="rf_crossfit_train_pseudo_oracle")
    parser.add_argument(
        "--output",
        default="paper_notes/second_paper_classification_selector_supplement.md",
    )
    args = parser.parse_args()

    df = load_metrics(args.selector_root, args.selector_model_name)
    if df.empty:
        raise SystemExit("No classification selector metrics found.")

    metric_cols = [
        "valid_accuracy",
        "test_accuracy",
        "valid_majority_accuracy",
        "test_majority_accuracy",
        "valid_macro_f1",
        "test_macro_f1",
    ]
    overall = df[metric_cols].mean().to_frame("mean").T.reset_index(drop=True)
    by_dataset = df.groupby("dataset", as_index=False)[metric_cols].mean()
    by_ratio = df.groupby("train_ratio_tag", as_index=False)[metric_cols].mean()

    lines = [
        "# Classification Selector Supplementary Diagnostics",
        "",
        "## Scope",
        "",
        "This note summarizes RF selector diagnostics for five classification endpoints:",
        "`bbb_martins`, `hia_hou`, `pgp_broccatelli`, `bioavailability_ma`, and `herg`.",
        "Each endpoint uses train ratios 10/20/50 and seeds 42/123/3407, for 45 settings.",
        "",
        "The selector is the same `rf_crossfit_train_pseudo_oracle` model used in the",
        "regression main experiments, with RF teachers `ECFP4_RF`, `Desc_RF`, and",
        "`ECFP4_Desc_RF`.",
        "",
        "## Overall Result",
        "",
        markdown_table(overall),
        "",
        "## By Dataset",
        "",
        markdown_table(by_dataset),
        "",
        "## By Train Ratio",
        "",
        markdown_table(by_ratio),
        "",
        "## Interpretation",
        "",
        "Classification selector transfer is not ready for the main claim. The RF selector",
        "overfits the cross-fit train labels but generalizes poorly: mean test selector",
        "accuracy is below the train-majority baseline. This supports keeping classification",
        "outside the main regression claim and treating it as a future calibration problem.",
        "",
        "The likely issue is not only teacher availability. Classification pseudo-oracle",
        "labels are based on probability-space absolute error, while downstream ADMET",
        "classification is evaluated by ranking metrics such as ROC-AUC. That mismatch",
        "can make sample-level teacher labels noisy and poorly aligned with student-level",
        "classification gains.",
        "",
        "## Paper Action",
        "",
        "- Do not promote classification routing to a main result.",
        "- Mention classification as supplementary diagnostics or limitation only.",
        "- If classification is revisited, redesign pseudo-oracle labels around",
        "  classification-calibrated criteria instead of copying the regression protocol.",
        "",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
