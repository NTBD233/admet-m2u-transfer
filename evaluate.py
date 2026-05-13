import argparse

from utils.config import RESULTS_ROOT
from utils.summary import collect_metrics, save_summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Collect per-seed metrics and write mean±std summaries.")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_df = collect_metrics(args.results_root)

    if metrics_df.empty:
        print(f"No metrics.json files found under: {args.results_root}")
        return

    paths = save_summaries(metrics_df, args.results_root)
    print("Summary files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
