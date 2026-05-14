import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import build_model
from utils.config import (
    BATCH_SIZE,
    DATASETS,
    FEATURE_ROOT,
    LAMBDA_TRANSFER,
    LR,
    MAX_EPOCHS,
    MODELS,
    PATIENCE,
    RESULTS_ROOT,
    SEEDS,
    TRAIN_RATIO_TAG,
    TRAIN_RATIO_TAGS,
    WEIGHT_DECAY,
)
from utils.dataset import make_loaders
from utils.io import save_run_outputs
from utils.metrics import compute_metrics, prediction_frame
from utils.seed import set_seed
from utils.summary import collect_metrics, save_summaries


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_metric_from_json(path, metric):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data[f"valid_{metric}"])


def load_teacher_valid_metric(teacher_root, dataset_name, teacher_model, train_ratio_tag, seed, metric):
    teacher_path = (
        Path(teacher_root)
        / dataset_name
        / teacher_model
        / f"train_{train_ratio_tag}"
        / f"seed_{seed}"
        / "valid_teacher_predictions.npz"
    )
    if not teacher_path.exists():
        raise FileNotFoundError(f"Missing validation teacher predictions: {teacher_path}")
    teacher_data = np.load(teacher_path, allow_pickle=True)
    metrics = teacher_data["metrics"][0]
    return float(metrics[metric])


def load_teacher_valid_metrics(teacher_root, dataset_name, teacher_models, train_ratio_tag, seed, metric):
    return {
        teacher_model: load_teacher_valid_metric(
            teacher_root=teacher_root,
            dataset_name=dataset_name,
            teacher_model=teacher_model,
            train_ratio_tag=train_ratio_tag,
            seed=seed,
            metric=metric,
        )
        for teacher_model in teacher_models
    }


def build_multiteacher_weights(
    teacher_root,
    teacher_models,
    strategy,
    dataset_name,
    train_ratio_tag,
    seed,
    main_metric,
    higher_is_better,
):
    if not teacher_models:
        return None, None

    if strategy == "uniform":
        weights = np.full(len(teacher_models), 1.0 / len(teacher_models), dtype=np.float32)
        info = {
            "multiteacher_strategy": strategy,
            "teacher_models": teacher_models,
            "teacher_valid_metric_map": {},
            "teacher_weight_map": {name: float(weight) for name, weight in zip(teacher_models, weights)},
        }
        return torch.tensor(weights, dtype=torch.float32, device=DEVICE), info

    valid_metric_map = load_teacher_valid_metrics(
        teacher_root=teacher_root,
        dataset_name=dataset_name,
        teacher_models=teacher_models,
        train_ratio_tag=train_ratio_tag,
        seed=seed,
        metric=main_metric,
    )
    scores = np.asarray(
        [
            valid_metric_map[name] if higher_is_better else -valid_metric_map[name]
            for name in teacher_models
        ],
        dtype=np.float32,
    )

    if strategy == "validation_weighted":
        shifted = scores - scores.max()
        weights = np.exp(shifted)
        weights = weights / weights.sum()
    elif strategy == "top1_validation":
        weights = np.zeros(len(teacher_models), dtype=np.float32)
        weights[int(np.argmax(scores))] = 1.0
    else:
        raise ValueError(f"Unknown multiteacher strategy: {strategy}")

    info = {
        "multiteacher_strategy": strategy,
        "teacher_models": teacher_models,
        "teacher_valid_metric_map": {k: float(v) for k, v in valid_metric_map.items()},
        "teacher_weight_map": {name: float(weight) for name, weight in zip(teacher_models, weights)},
    }
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE), info


def build_batch_teacher_weights(
    strategy,
    teacher_uncertainty,
    global_teacher_weights=None,
):
    if teacher_uncertainty is None:
        raise ValueError(f"{strategy} requires teacher uncertainty inputs")

    uncertainty = teacher_uncertainty.to(DEVICE)
    if uncertainty.ndim != 2:
        raise ValueError(
            f"Expected teacher_uncertainty to have shape [batch, teachers], got {uncertainty.shape}"
        )

    finite_mask = torch.isfinite(uncertainty)
    if not finite_mask.any():
        raise ValueError(f"{strategy} requires at least one finite teacher uncertainty")

    fallback = torch.where(
        finite_mask,
        uncertainty,
        torch.full_like(uncertainty, float("inf")),
    ).amin(dim=1, keepdim=True)
    uncertainty = torch.where(finite_mask, uncertainty, fallback + 1.0)

    scores = -uncertainty
    if strategy == "uncertainty_validation_prior":
        if global_teacher_weights is None:
            raise ValueError("uncertainty_validation_prior requires global teacher weights")
        scores = scores + torch.log(global_teacher_weights.clamp_min(1e-8)).view(1, -1)
    elif strategy != "uncertainty_only":
        raise ValueError(f"Unknown sample-level multiteacher strategy: {strategy}")

    return torch.softmax(scores, dim=1)


def resolve_lambda_distill(
    lambda_distill,
    adaptive_distill_strategy,
    base_results_root,
    teacher_root,
    teacher_model,
    dataset_name,
    model_name,
    train_ratio_tag,
    seed,
    main_metric,
    higher_is_better,
):
    if adaptive_distill_strategy == "none":
        return lambda_distill, None

    if adaptive_distill_strategy != "teacher_valid_advantage":
        raise ValueError(f"Unknown adaptive distill strategy: {adaptive_distill_strategy}")

    if base_results_root is None:
        raise ValueError("--adaptive-base-results-root is required for teacher_valid_advantage")
    if teacher_root is None or teacher_model is None:
        raise ValueError("--teacher-root and --teacher-model are required for adaptive distillation")

    base_metrics_path = (
        Path(base_results_root)
        / dataset_name
        / model_name
        / f"train_{train_ratio_tag}"
        / f"seed_{seed}"
        / "metrics.json"
    )
    if not base_metrics_path.exists():
        raise FileNotFoundError(f"Missing base metrics for adaptive distillation: {base_metrics_path}")

    base_valid = load_metric_from_json(base_metrics_path, main_metric)
    teacher_valid = load_teacher_valid_metric(
        teacher_root=teacher_root,
        dataset_name=dataset_name,
        teacher_model=teacher_model,
        train_ratio_tag=train_ratio_tag,
        seed=seed,
        metric=main_metric,
    )

    if higher_is_better:
        raw_advantage = teacher_valid - base_valid
        scale = max(abs(base_valid), 1e-8)
    else:
        raw_advantage = base_valid - teacher_valid
        scale = max(abs(base_valid), 1e-8)
    advantage_ratio = max(raw_advantage / scale, 0.0)
    effective_lambda = lambda_distill * min(advantage_ratio, 1.0)
    info = {
        "adaptive_distill_strategy": adaptive_distill_strategy,
        "base_valid_metric": base_valid,
        "teacher_valid_metric": teacher_valid,
        "teacher_valid_advantage": raw_advantage,
        "teacher_valid_advantage_ratio": advantage_ratio,
        "lambda_distill_max": lambda_distill,
        "lambda_distill_effective": effective_lambda,
    }
    return effective_lambda, info


def compute_loss(outputs, y, desc, task_type, model_name, lambda_transfer=LAMBDA_TRANSFER):
    total_loss, task_loss, transfer_loss, _ = compute_total_loss(
        outputs=outputs,
        y=y,
        desc=desc,
        task_type=task_type,
        model_name=model_name,
        lambda_transfer=lambda_transfer,
    )
    return total_loss, task_loss, transfer_loss


def compute_total_loss(
    outputs,
    y,
    desc,
    task_type,
    model_name,
    lambda_transfer=LAMBDA_TRANSFER,
    teacher_pred=None,
    teacher_weights=None,
    lambda_distill=0.0,
):
    pred = outputs["pred"]

    if task_type == "classification":
        task_loss = nn.BCEWithLogitsLoss()(pred, y)
    elif task_type == "regression":
        task_loss = nn.MSELoss()(pred, y)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    transfer_loss = torch.tensor(0.0, device=pred.device)
    if model_name in {"ECFP4_MLP_DescPred", "ECFP4_MLP_DescAdapterFusion"}:
        transfer_loss = nn.MSELoss()(outputs["desc_hat"], desc)

    distill_loss = torch.tensor(0.0, device=pred.device)
    if teacher_pred is not None and lambda_distill > 0:
        if teacher_pred.ndim == 2 and teacher_pred.shape[1] > 1:
            if teacher_weights is None:
                teacher_weights = torch.full(
                    (teacher_pred.shape[1],),
                    1.0 / teacher_pred.shape[1],
                    device=pred.device,
                )
            per_teacher_losses = []
            for idx in range(teacher_pred.shape[1]):
                teacher_target = teacher_pred[:, idx].view_as(pred)
                if task_type == "classification":
                    teacher_loss = F.binary_cross_entropy_with_logits(
                        pred,
                        teacher_target,
                        reduction="none",
                    ).view(pred.shape[0], -1).mean(dim=1)
                elif task_type == "regression":
                    teacher_loss = F.mse_loss(
                        pred,
                        teacher_target,
                        reduction="none",
                    ).view(pred.shape[0], -1).mean(dim=1)
                else:
                    raise ValueError(f"Unknown task_type: {task_type}")
                per_teacher_losses.append(teacher_loss)
            per_teacher_losses = torch.stack(per_teacher_losses, dim=1)
            if teacher_weights.ndim == 1:
                distill_loss = torch.sum(per_teacher_losses * teacher_weights.view(1, -1), dim=1).mean()
            elif teacher_weights.ndim == 2:
                distill_loss = torch.sum(per_teacher_losses * teacher_weights, dim=1).mean()
            else:
                raise ValueError(f"Unexpected teacher_weights shape: {teacher_weights.shape}")
        elif task_type == "classification":
            distill_loss = F.binary_cross_entropy_with_logits(pred, teacher_pred)
        elif task_type == "regression":
            distill_loss = nn.MSELoss()(pred, teacher_pred)

    total_loss = task_loss + lambda_transfer * transfer_loss + lambda_distill * distill_loss
    return total_loss, task_loss, transfer_loss, distill_loss


@torch.no_grad()
def evaluate_model(model, loader, task_type):
    model.eval()

    all_y = []
    all_pred_raw = []
    all_smiles = []

    for batch in loader:
        fp = batch["fp"].to(DEVICE)
        desc = batch["desc"].to(DEVICE)
        y = batch["y"].to(DEVICE)

        outputs = model(fp, desc)
        pred_raw = outputs["pred"]

        all_y.append(y.cpu().numpy())
        all_pred_raw.append(pred_raw.cpu().numpy())
        all_smiles.extend(batch["smiles"])

    y_true = np.concatenate(all_y).reshape(-1)
    pred_raw = np.concatenate(all_pred_raw).reshape(-1)
    metrics, pred = compute_metrics(y_true, pred_raw, task_type)
    pred_df = prediction_frame(all_smiles, y_true, pred, pred_raw)

    return metrics, pred_df


def is_better(current, best, higher_is_better):
    if best is None:
        return True
    return current > best if higher_is_better else current < best


def train_one_model(
    dataset_name,
    task_type,
    model_name,
    seed,
    train_ratio_tag=TRAIN_RATIO_TAG,
    results_root=RESULTS_ROOT,
    lambda_transfer=LAMBDA_TRANSFER,
    teacher_root=None,
    teacher_model=None,
    teacher_models=None,
    multiteacher_strategy="uniform",
    lambda_distill=0.0,
    adaptive_distill_strategy="none",
    adaptive_base_results_root=None,
    skip_existing=False,
):
    set_seed(seed)

    save_dir = (
        Path(results_root)
        / dataset_name
        / model_name
        / f"train_{train_ratio_tag}"
        / f"seed_{seed}"
    )
    if skip_existing and (save_dir / "metrics.json").exists():
        print(f"Skipping existing run: {save_dir}")
        return None

    train_loader, valid_loader, test_loader = make_loaders(
        dataset_name=dataset_name,
        task_type=task_type,
        batch_size=BATCH_SIZE,
        train_ratio_tag=train_ratio_tag,
        teacher_root=teacher_root,
        teacher_model=teacher_model,
        teacher_models=teacher_models,
        seed=seed,
    )

    model = build_model(model_name).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    dataset_cfg = DATASETS[dataset_name]
    higher_is_better = dataset_cfg["higher_is_better"]
    main_metric = dataset_cfg["main_metric"]
    uses_sample_level_gate = multiteacher_strategy in {
        "uncertainty_only",
        "uncertainty_validation_prior",
    }
    teacher_weights = None
    multiteacher_info = None
    if teacher_models is not None:
        if teacher_root is None:
            raise ValueError("--teacher-root is required when using --teacher-models")
        global_weight_strategy = (
            "validation_weighted"
            if multiteacher_strategy == "uncertainty_validation_prior"
            else "uniform" if multiteacher_strategy == "uncertainty_only" else multiteacher_strategy
        )
        teacher_weights, multiteacher_info = build_multiteacher_weights(
            teacher_root=teacher_root,
            teacher_models=teacher_models,
            strategy=global_weight_strategy,
            dataset_name=dataset_name,
            train_ratio_tag=train_ratio_tag,
            seed=seed,
            main_metric=main_metric,
            higher_is_better=higher_is_better,
        )
        if uses_sample_level_gate and multiteacher_info is not None:
            multiteacher_info["sample_level_gate"] = multiteacher_strategy
    lambda_distill_effective, adaptive_distill_info = resolve_lambda_distill(
        lambda_distill=lambda_distill,
        adaptive_distill_strategy=adaptive_distill_strategy,
        base_results_root=adaptive_base_results_root,
        teacher_root=teacher_root,
        teacher_model=teacher_model,
        dataset_name=dataset_name,
        model_name=model_name,
        train_ratio_tag=train_ratio_tag,
        seed=seed,
        main_metric=main_metric,
        higher_is_better=higher_is_better,
    )
    if multiteacher_info is not None:
        print(
            f"{dataset_name} | {model_name} | seed={seed} | "
            f"multiteacher={multiteacher_strategy} | "
            f"weights={multiteacher_info['teacher_weight_map']}"
        )
    if adaptive_distill_info is not None:
        print(
            f"{dataset_name} | {model_name} | seed={seed} | "
            f"adaptive_distill={adaptive_distill_strategy} | "
            f"lambda_max={lambda_distill:.4f} | "
            f"lambda_effective={lambda_distill_effective:.4f} | "
            f"teacher_valid_advantage_ratio={adaptive_distill_info['teacher_valid_advantage_ratio']:.4f}"
        )

    best_metric = None
    best_epoch = -1
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        total_loss_sum = 0.0
        task_loss_sum = 0.0
        transfer_loss_sum = 0.0
        distill_loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            fp = batch["fp"].to(DEVICE)
            desc = batch["desc"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            teacher_pred = batch.get("teacher_pred")
            teacher_uncertainty = batch.get("teacher_uncertainty")
            if teacher_pred is not None:
                teacher_pred = teacher_pred.to(DEVICE)
            batch_teacher_weights = teacher_weights
            if uses_sample_level_gate:
                batch_teacher_weights = build_batch_teacher_weights(
                    strategy=multiteacher_strategy,
                    teacher_uncertainty=teacher_uncertainty,
                    global_teacher_weights=teacher_weights,
                )

            optimizer.zero_grad()
            outputs = model(fp, desc)
            total_loss, task_loss, transfer_loss, distill_loss = compute_total_loss(
                outputs=outputs,
                y=y,
                desc=desc,
                task_type=task_type,
                model_name=model_name,
                lambda_transfer=lambda_transfer,
                teacher_pred=teacher_pred,
                teacher_weights=batch_teacher_weights,
                lambda_distill=lambda_distill_effective,
            )
            total_loss.backward()
            optimizer.step()

            total_loss_sum += total_loss.item()
            task_loss_sum += task_loss.item()
            transfer_loss_sum += transfer_loss.item()
            distill_loss_sum += distill_loss.item()
            n_batches += 1

        train_total_loss = total_loss_sum / max(n_batches, 1)
        train_task_loss = task_loss_sum / max(n_batches, 1)
        train_transfer_loss = transfer_loss_sum / max(n_batches, 1)
        train_distill_loss = distill_loss_sum / max(n_batches, 1)

        valid_metrics, _ = evaluate_model(model, valid_loader, task_type)
        current_metric = valid_metrics[main_metric]

        history.append({
            "epoch": epoch,
            "train_total_loss": train_total_loss,
            "train_task_loss": train_task_loss,
            "train_transfer_loss": train_transfer_loss,
            "train_distill_loss": train_distill_loss,
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        })

        print(
            f"{dataset_name} | {model_name} | seed={seed} | "
            f"epoch={epoch:03d} | loss={train_total_loss:.6f} | "
            + " | ".join(f"valid_{k}={v:.6f}" for k, v in valid_metrics.items())
        )

        if is_better(current_metric, best_metric, higher_is_better):
            best_metric = current_metric
            best_epoch = epoch
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "best_epoch": best_epoch,
                "best_metric": float(best_metric),
                "dataset": dataset_name,
                "task_type": task_type,
                "model": model_name,
                "seed": seed,
                "lambda_transfer": lambda_transfer,
                "lambda_distill": lambda_distill,
                "lambda_distill_effective": lambda_distill_effective,
                "adaptive_distill_strategy": adaptive_distill_strategy,
                "teacher_model": teacher_model,
                "teacher_models": teacher_models,
                "multiteacher_strategy": multiteacher_strategy if teacher_models is not None else "single_teacher",
            }
            if adaptive_distill_info is not None:
                best_state.update(adaptive_distill_info)
            if multiteacher_info is not None:
                best_state.update(multiteacher_info)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    model.load_state_dict(best_state["model_state_dict"])
    valid_metrics, valid_pred_df = evaluate_model(model, valid_loader, task_type)
    test_metrics, test_pred_df = evaluate_model(model, test_loader, task_type)

    metrics = {
        "dataset": dataset_name,
        "task_type": task_type,
        "model": model_name,
        "train_ratio_tag": train_ratio_tag,
        "seed": seed,
        "lambda_transfer": lambda_transfer,
        "lambda_distill": lambda_distill,
        "lambda_distill_effective": lambda_distill_effective,
        "adaptive_distill_strategy": adaptive_distill_strategy,
        "teacher_model": teacher_model,
        "teacher_models": teacher_models,
        "multiteacher_strategy": multiteacher_strategy if teacher_models is not None else "single_teacher",
        "best_epoch": best_epoch,
        "best_metric": float(best_metric),
        "valid_roc_auc": np.nan,
        "valid_pr_auc": np.nan,
        "test_roc_auc": np.nan,
        "test_pr_auc": np.nan,
        "valid_mae": np.nan,
        "valid_rmse": np.nan,
        "test_mae": np.nan,
        "test_rmse": np.nan,
    }

    metrics.update({f"valid_{k}": float(v) for k, v in valid_metrics.items()})
    metrics.update({f"test_{k}": float(v) for k, v in test_metrics.items()})
    if adaptive_distill_info is not None:
        metrics.update(adaptive_distill_info)
    if multiteacher_info is not None:
        metrics.update(multiteacher_info)

    best_state["valid_metrics"] = valid_metrics
    best_state["test_metrics"] = test_metrics

    save_run_outputs(
        save_dir=save_dir,
        best_state=best_state,
        history=history,
        valid_pred_df=valid_pred_df,
        test_pred_df=test_pred_df,
        metrics=metrics,
    )

    print(f"Saved run outputs to: {save_dir}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train lightweight M2U models from the notebook.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--train-ratio-tags", nargs="+", type=int, default=[TRAIN_RATIO_TAG])
    parser.add_argument("--generate-features", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--lambda-transfer", type=float, default=LAMBDA_TRANSFER)
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--teacher-root", default=None)
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--teacher-models", nargs="+", default=None)
    parser.add_argument(
        "--multiteacher-strategy",
        choices=[
            "uniform",
            "validation_weighted",
            "top1_validation",
            "uncertainty_only",
            "uncertainty_validation_prior",
        ],
        default="uniform",
    )
    parser.add_argument("--lambda-distill", type=float, default=0.0)
    parser.add_argument(
        "--adaptive-distill-strategy",
        choices=["none", "teacher_valid_advantage"],
        default="none",
    )
    parser.add_argument("--adaptive-base-results-root", default=str(RESULTS_ROOT))
    return parser.parse_args()


def main():
    args = parse_args()

    if args.teacher_model is not None and args.teacher_models is not None:
        raise ValueError("Use either --teacher-model or --teacher-models, not both")
    if args.teacher_models is not None and args.adaptive_distill_strategy != "none":
        raise ValueError(
            "Adaptive single-teacher distillation does not support --teacher-models yet"
        )

    selected_datasets = {name: DATASETS[name] for name in args.datasets}

    if args.generate_features:
        from utils.features import generate_features_for_datasets

        print(f"Generating features into: {FEATURE_ROOT}")
        generate_features_for_datasets(
            selected_datasets,
            train_ratio_tags=args.train_ratio_tags,
        )
        if args.features_only:
            return

    for dataset_name, cfg in selected_datasets.items():
        for train_ratio_tag in args.train_ratio_tags:
            for model_name in args.models:
                for seed in args.seeds:
                    train_one_model(
                        dataset_name=dataset_name,
                        task_type=cfg["task_type"],
                        model_name=model_name,
                        seed=seed,
                        train_ratio_tag=train_ratio_tag,
                        results_root=args.results_root,
                        lambda_transfer=args.lambda_transfer,
                        teacher_root=args.teacher_root,
                        teacher_model=args.teacher_model,
                        teacher_models=args.teacher_models,
                        multiteacher_strategy=args.multiteacher_strategy,
                        lambda_distill=args.lambda_distill,
                        adaptive_distill_strategy=args.adaptive_distill_strategy,
                        adaptive_base_results_root=args.adaptive_base_results_root,
                        skip_existing=args.skip_existing,
                    )

    metrics_df = collect_metrics(args.results_root)
    paths = save_summaries(metrics_df, args.results_root)
    print("Summary files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
