import argparse
from pathlib import Path

import pandas as pd

from utils.config import DATASETS, MODELS, RESULTS_ROOT, SEEDS, TRAIN_RATIO_TAGS


DEFAULT_ML_MODELS = ["ECFP4_RF", "Desc_RF", "ECFP4_Desc_RF"]


def count_runs(results_root, datasets, models, ratios, seeds):
    rows = []
    for dataset in datasets:
        for ratio in ratios:
            expected = len(models) * len(seeds)
            completed = 0
            missing = []
            for model in models:
                for seed in seeds:
                    metrics_path = (
                        Path(results_root)
                        / dataset
                        / model
                        / f"train_{ratio}"
                        / f"seed_{seed}"
                        / "metrics.json"
                    )
                    if metrics_path.exists():
                        completed += 1
                    else:
                        missing.append(f"{model}:seed_{seed}")
            rows.append({
                "dataset": dataset,
                "train_ratio_tag": ratio,
                "completed": completed,
                "expected": expected,
                "percent": round(100 * completed / expected, 1) if expected else 0.0,
                "missing_examples": ", ".join(missing[:5]),
            })
    return pd.DataFrame(rows)


def print_table(title, df):
    print(f"\n{title}")
    if df.empty:
        print("(empty)")
        return
    print(df.to_string(index=False))
    completed = int(df["completed"].sum())
    expected = int(df["expected"].sum())
    pct = round(100 * completed / expected, 1) if expected else 0.0
    print(f"TOTAL: {completed}/{expected} ({pct}%)")


def parse_args():
    parser = argparse.ArgumentParser(description="Show ADMET/M2U experiment completion status.")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--ratios", nargs="+", type=int, default=TRAIN_RATIO_TAGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--ml-models", nargs="+", default=DEFAULT_ML_MODELS)
    return parser.parse_args()


def main():
    args = parse_args()
    neural_df = count_runs(
        results_root=args.results_root,
        datasets=args.datasets,
        models=MODELS,
        ratios=args.ratios,
        seeds=args.seeds,
    )
    ml_df = count_runs(
        results_root=args.results_root,
        datasets=args.datasets,
        models=args.ml_models,
        ratios=args.ratios,
        seeds=args.seeds,
    )

    print_table("Neural runs", neural_df)
    print_table("ML baseline runs", ml_df)


if __name__ == "__main__":
    main()
