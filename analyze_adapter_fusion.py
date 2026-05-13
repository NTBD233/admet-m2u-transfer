import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models import build_model
from utils.config import DATASETS, DESC_COLS, FEATURE_ROOT, RESULTS_ROOT, SEEDS, TRAIN_RATIO_TAG
from utils.dataset import M2UFeatureDataset, feature_paths
from utils.summary import collect_metrics


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_dir(results_root, dataset_name, model_name, train_ratio_tag, seed):
    ratio_dir = (
        Path(results_root)
        / dataset_name
        / model_name
        / f"train_{train_ratio_tag}"
        / f"seed_{seed}"
    )
    if ratio_dir.exists():
        return ratio_dir
    return Path(results_root) / dataset_name / model_name / f"seed_{seed}"


@torch.no_grad()
def analyze_one(dataset_name, task_type, model_name, train_ratio_tag, seed, split, results_root):
    _, valid_path, test_path = feature_paths(
        dataset_name=dataset_name,
        feature_root=FEATURE_ROOT,
        train_ratio_tag=train_ratio_tag,
    )
    npz_path = valid_path if split == "valid" else test_path
    ds = M2UFeatureDataset(npz_path, task_type)

    checkpoint_path = run_dir(
        results_root=results_root,
        dataset_name=dataset_name,
        model_name=model_name,
        train_ratio_tag=train_ratio_tag,
        seed=seed,
    ) / "best_model.pt"
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = build_model(model_name).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fp = ds.X_fp.to(DEVICE)
    desc = ds.X_desc.to(DEVICE)
    outputs = model(fp, desc)

    row = {
        "dataset": dataset_name,
        "task_type": task_type,
        "model": model_name,
        "train_ratio_tag": train_ratio_tag,
        "seed": seed,
        "split": split,
    }

    if "desc_hat" in outputs:
        desc_hat = outputs["desc_hat"].detach().cpu().numpy()
        desc_true = ds.X_desc.detach().cpu().numpy()
        desc_abs_err = np.abs(desc_hat - desc_true)
        row["desc_mae"] = float(desc_abs_err.mean())
        for idx, name in enumerate(DESC_COLS):
            row[f"desc_mae_{name}"] = float(desc_abs_err[:, idx].mean())

    if "fusion_gate" in outputs:
        gate = outputs["fusion_gate"].detach().cpu().numpy()
        row["gate_mean"] = float(gate.mean())
        row["gate_std"] = float(gate.std())
        row["gate_p10"] = float(np.quantile(gate, 0.10))
        row["gate_p50"] = float(np.quantile(gate, 0.50))
        row["gate_p90"] = float(np.quantile(gate, 0.90))

    return row


def build_correlation_table(analysis_df, metrics_df):
    if analysis_df.empty or metrics_df.empty:
        return pd.DataFrame()

    merged = analysis_df.merge(
        metrics_df,
        on=["dataset", "task_type", "model", "train_ratio_tag", "seed"],
        how="left",
    )
    rows = []
    for task_type, group in merged.groupby("task_type"):
        if "desc_mae" not in group.columns:
            continue
        metric = "test_roc_auc" if task_type == "classification" else "test_rmse"
        valid = group[["desc_mae", metric]].dropna()
        if len(valid) < 2:
            corr = np.nan
        else:
            corr = valid["desc_mae"].corr(valid[metric])
        rows.append({
            "task_type": task_type,
            "metric": metric,
            "n": len(valid),
            "desc_mae_metric_corr": corr,
        })
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze AdapterFusion gates and descriptor prediction errors.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ECFP4_MLP_DescPred", "ECFP4_MLP_DescAdapterFusion"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--train-ratio-tags", nargs="+", type=int, default=[TRAIN_RATIO_TAG])
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for dataset_name in args.datasets:
        cfg = DATASETS[dataset_name]
        for train_ratio_tag in args.train_ratio_tags:
            for model_name in args.models:
                for seed in args.seeds:
                    row = analyze_one(
                        dataset_name=dataset_name,
                        task_type=cfg["task_type"],
                        model_name=model_name,
                        train_ratio_tag=train_ratio_tag,
                        seed=seed,
                        split=args.split,
                        results_root=args.results_root,
                    )
                    if row is not None:
                        rows.append(row)

    summary_dir = Path(args.results_root) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    analysis_df = pd.DataFrame(rows)
    analysis_path = summary_dir / "adapter_fusion_analysis.csv"
    analysis_df.to_csv(analysis_path, index=False)

    metrics_df = collect_metrics(args.results_root)
    corr_df = build_correlation_table(analysis_df, metrics_df)
    corr_path = summary_dir / "descriptor_error_correlation.csv"
    corr_df.to_csv(corr_path, index=False)

    print(f"analysis_csv: {analysis_path}")
    print(f"correlation_csv: {corr_path}")


if __name__ == "__main__":
    main()
