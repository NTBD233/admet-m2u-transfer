import argparse
import pickle
from pathlib import Path

import numpy as np

from train_ml_baselines import build_features, load_split, model_name_parts, predict_raw
from utils.config import DATASETS, RESULTS_ROOT, SEEDS, TRAIN_RATIO_TAG, PROJECT_ROOT
from utils.dataset import feature_paths
from utils.metrics import compute_metrics


TEACHER_ROOT = PROJECT_ROOT / "data" / "teacher_predictions"


def teacher_run_dir(results_root, dataset_name, teacher_model, train_ratio_tag, seed):
    return (
        Path(results_root)
        / dataset_name
        / teacher_model
        / f"train_{train_ratio_tag}"
        / f"seed_{seed}"
    )


def generate_one(dataset_name, task_type, teacher_model, train_ratio_tag, seed, results_root, output_root):
    run_dir = teacher_run_dir(results_root, dataset_name, teacher_model, train_ratio_tag, seed)
    model_path = run_dir / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing teacher model: {model_path}")

    with model_path.open("rb") as f:
        teacher = pickle.load(f)

    feature_set, _ = model_name_parts(teacher_model)
    train_path, valid_path, test_path = feature_paths(
        dataset_name=dataset_name,
        train_ratio_tag=train_ratio_tag,
    )
    split_paths = {
        f"train_{train_ratio_tag}": train_path,
        "valid": valid_path,
        "test": test_path,
    }

    out_dir = Path(output_root) / dataset_name / teacher_model / f"train_{train_ratio_tag}" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, npz_path in split_paths.items():
        split = load_split(npz_path)
        x = build_features(split, feature_set)
        pred_raw = predict_raw(teacher, x, task_type)
        metrics, pred = compute_metrics(split["y"], pred_raw, task_type)
        np.savez_compressed(
            out_dir / f"{split_name}_teacher_predictions.npz",
            pred=pred.astype(np.float32),
            pred_raw=pred_raw.astype(np.float32),
            y=split["y"].astype(np.float32),
            smiles=split["smiles"],
            metrics=np.array([metrics], dtype=object),
        )

    print(f"Saved teacher predictions to: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate per-split teacher predictions for distillation.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--teacher-model", default="ECFP4_Desc_RF")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--train-ratio-tags", nargs="+", type=int, default=[TRAIN_RATIO_TAG])
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(TEACHER_ROOT))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for dataset_name in args.datasets:
        task_type = DATASETS[dataset_name]["task_type"]
        for train_ratio_tag in args.train_ratio_tags:
            for seed in args.seeds:
                output_path = (
                    Path(args.output_root)
                    / dataset_name
                    / args.teacher_model
                    / f"train_{train_ratio_tag}"
                    / f"seed_{seed}"
                    / f"train_{train_ratio_tag}_teacher_predictions.npz"
                )
                if args.skip_existing and output_path.exists():
                    print(f"Skipping existing teacher predictions: {output_path.parent}")
                    continue
                generate_one(
                    dataset_name=dataset_name,
                    task_type=task_type,
                    teacher_model=args.teacher_model,
                    train_ratio_tag=train_ratio_tag,
                    seed=seed,
                    results_root=args.results_root,
                    output_root=args.output_root,
                )


if __name__ == "__main__":
    main()
