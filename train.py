import argparse
import copy
import json
import random
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


class TeacherReliabilityGate(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        logits = self.scorer(features).squeeze(-1)
        return torch.softmax(logits, dim=1)


class TeacherReliabilityLinearGate(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.scorer = nn.Linear(input_dim, 1)

    def forward(self, features):
        logits = self.scorer(features).squeeze(-1)
        return torch.softmax(logits, dim=1)


class TeacherReliabilityPriorResidualGate(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8, residual_scale=0.5):
        super().__init__()
        self.residual_scale = residual_scale
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, features, prior_weights):
        residual_logits = self.scorer(features).squeeze(-1)
        residual_logits = self.residual_scale * torch.tanh(residual_logits)
        prior_logits = torch.log(prior_weights.clamp_min(1e-8)).view(1, -1)
        return torch.softmax(prior_logits + residual_logits, dim=1)


def straight_through_topk_weights(soft_weights, k=1):
    if soft_weights.ndim != 2:
        raise ValueError(f"Expected soft_weights to have shape [batch, teachers], got {soft_weights.shape}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    n_teachers = soft_weights.shape[1]
    k = min(k, n_teachers)
    topk_idx = torch.topk(soft_weights, k=k, dim=1).indices
    hard = torch.zeros_like(soft_weights)
    hard.scatter_(1, topk_idx, 1.0)
    if k > 1:
        masked_soft = soft_weights * hard
        hard = masked_soft / masked_soft.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return hard - soft_weights.detach() + soft_weights


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
    teacher_pred,
    selector_probs=None,
    student_pred=None,
    global_teacher_weights=None,
    gate_module=None,
    selector_confidence_threshold=0.6,
):
    if strategy in {
        "pretrained_selector_top1",
        "pretrained_selector_confidence_fallback_top1",
        "pretrained_selector_validation_blend_top1",
        "pretrained_selector_top2",
        "selector_filtered_uniform",
        "selector_filtered_validation_weighted",
    }:
        if selector_probs is None:
            raise ValueError(f"{strategy} requires selector probabilities")
        selector_probs = selector_probs.to(DEVICE)
        if selector_probs.ndim != 2:
            raise ValueError(f"Expected selector_probs to have shape [batch, teachers], got {selector_probs.shape}")
        if strategy == "pretrained_selector_top1":
            top1_idx = torch.argmax(selector_probs, dim=1, keepdim=True)
            weights = torch.zeros_like(selector_probs)
            weights.scatter_(1, top1_idx, 1.0)
            return weights
        if strategy == "pretrained_selector_validation_blend_top1":
            if global_teacher_weights is None:
                raise ValueError("pretrained_selector_validation_blend_top1 requires global teacher weights")
            blended_probs = selector_probs * global_teacher_weights.view(1, -1)
            top1_idx = torch.argmax(blended_probs, dim=1, keepdim=True)
            weights = torch.zeros_like(selector_probs)
            weights.scatter_(1, top1_idx, 1.0)
            return weights
        if strategy == "pretrained_selector_confidence_fallback_top1":
            if global_teacher_weights is None:
                raise ValueError("pretrained_selector_confidence_fallback_top1 requires global teacher weights")
            selector_top1_idx = torch.argmax(selector_probs, dim=1, keepdim=True)
            fallback_idx = torch.argmax(global_teacher_weights).view(1, 1).expand(selector_probs.shape[0], 1)
            selector_confidence = torch.max(selector_probs, dim=1, keepdim=True).values
            chosen_idx = torch.where(
                selector_confidence >= selector_confidence_threshold,
                selector_top1_idx,
                fallback_idx,
            )
            weights = torch.zeros_like(selector_probs)
            weights.scatter_(1, chosen_idx, 1.0)
            return weights
        if strategy == "pretrained_selector_top2":
            top2_idx = torch.topk(selector_probs, k=min(2, selector_probs.shape[1]), dim=1).indices
            weights = torch.zeros_like(selector_probs)
            weights.scatter_(1, top2_idx, 1.0)
            weights = selector_probs * weights
            return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        threshold = 1.0 / selector_probs.shape[1]
        mask = selector_probs >= threshold
        if strategy == "selector_filtered_uniform":
            weights = mask.float()
            return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if global_teacher_weights is None:
            raise ValueError("selector_filtered_validation_weighted requires global teacher weights")
        weights = mask.float() * global_teacher_weights.view(1, -1)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    if teacher_uncertainty is None:
        raise ValueError(f"{strategy} requires teacher uncertainty inputs")
    if teacher_pred is None:
        raise ValueError(f"{strategy} requires teacher predictions")

    uncertainty = teacher_uncertainty.to(DEVICE)
    teacher_pred = teacher_pred.to(DEVICE)
    if uncertainty.ndim != 2:
        raise ValueError(
            f"Expected teacher_uncertainty to have shape [batch, teachers], got {uncertainty.shape}"
        )
    if teacher_pred.ndim != 2:
        raise ValueError(
            f"Expected teacher_pred to have shape [batch, teachers], got {teacher_pred.shape}"
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
    teacher_consensus = teacher_pred.mean(dim=1, keepdim=True)
    teacher_disagreement = torch.abs(teacher_pred - teacher_consensus)
    student_disagreement = None
    if student_pred is not None:
        student_disagreement = torch.abs(teacher_pred - student_pred.view(-1, 1))

    if strategy in {
        "learned_reliability_gate",
        "learned_linear_gate",
        "learned_prior_residual_gate",
        "supervised_hard_top1_gate",
        "supervised_hard_top2_gate",
    }:
        if student_disagreement is None:
            raise ValueError(f"{strategy} requires student predictions")
        if global_teacher_weights is None:
            raise ValueError(f"{strategy} requires global teacher weights")
        if gate_module is None:
            raise ValueError(f"{strategy} requires a gate module")
        gate_features = torch.stack(
            [
                uncertainty,
                teacher_disagreement,
                student_disagreement,
                torch.log(global_teacher_weights.clamp_min(1e-8)).view(1, -1).expand_as(uncertainty),
            ],
            dim=-1,
        )
        if strategy in {
            "learned_prior_residual_gate",
            "supervised_hard_top1_gate",
            "supervised_hard_top2_gate",
        }:
            soft_weights = gate_module(gate_features, global_teacher_weights)
            if strategy == "supervised_hard_top1_gate":
                return straight_through_topk_weights(soft_weights, k=1)
            if strategy == "supervised_hard_top2_gate":
                return straight_through_topk_weights(soft_weights, k=2)
            return soft_weights
        return gate_module(gate_features)
    if strategy == "uncertainty_validation_prior":
        if global_teacher_weights is None:
            raise ValueError("uncertainty_validation_prior requires global teacher weights")
        scores = scores + torch.log(global_teacher_weights.clamp_min(1e-8)).view(1, -1)
    elif strategy == "uncertainty_teacher_disagreement":
        scores = scores - teacher_disagreement
    elif strategy == "uncertainty_teacher_student":
        if student_disagreement is None:
            raise ValueError("uncertainty_teacher_student requires student predictions")
        scores = scores - student_disagreement
    elif strategy == "uncertainty_student_prior":
        if student_disagreement is None:
            raise ValueError("uncertainty_student_prior requires student predictions")
        if global_teacher_weights is None:
            raise ValueError("uncertainty_student_prior requires global teacher weights")
        scores = (
            scores
            + torch.log(global_teacher_weights.clamp_min(1e-8)).view(1, -1)
            - student_disagreement
        )
    elif strategy == "uncertainty_composite":
        if student_disagreement is None:
            raise ValueError("uncertainty_composite requires student predictions")
        if global_teacher_weights is None:
            raise ValueError("uncertainty_composite requires global teacher weights")
        scores = (
            scores
            + torch.log(global_teacher_weights.clamp_min(1e-8)).view(1, -1)
            - teacher_disagreement
            - student_disagreement
        )
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
    distill_sample_weights=None,
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
                per_sample_distill = torch.sum(per_teacher_losses * teacher_weights.view(1, -1), dim=1)
            elif teacher_weights.ndim == 2:
                per_sample_distill = torch.sum(per_teacher_losses * teacher_weights, dim=1)
            else:
                raise ValueError(f"Unexpected teacher_weights shape: {teacher_weights.shape}")
            if distill_sample_weights is not None:
                per_sample_distill = per_sample_distill * distill_sample_weights.view(-1)
            distill_loss = per_sample_distill.mean()
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


@torch.no_grad()
def choose_selector_reweight_mode(valid_loader, task_type, teacher_weights):
    if teacher_weights is None:
        return "global_confidence", None

    global_top1_idx = int(torch.argmax(teacher_weights).item())
    total = 0
    selector_correct = 0
    global_correct = 0

    for batch in valid_loader:
        teacher_pred = batch.get("teacher_pred")
        selector_probs = batch.get("selector_probs")
        y = batch["y"]
        if teacher_pred is None or selector_probs is None:
            continue
        if task_type != "regression" or teacher_pred.ndim != 2 or teacher_pred.shape[1] <= 1:
            continue

        oracle_idx = torch.argmin(torch.abs(teacher_pred - y.view(-1, 1)), dim=1)
        selector_idx = torch.argmax(selector_probs, dim=1)
        selector_correct += int((selector_idx == oracle_idx).sum().item())
        global_correct += int((oracle_idx == global_top1_idx).sum().item())
        total += int(oracle_idx.numel())

    if total == 0:
        return "global_confidence", None

    selector_acc = selector_correct / total
    global_acc = global_correct / total
    chosen_mode = "global_confidence" if selector_acc > global_acc else "disagreement_confidence"
    return chosen_mode, {
        "selector_valid_oracle_acc": selector_acc,
        "top1_valid_oracle_acc": global_acc,
        "resolved_selector_reweight_mode": chosen_mode,
    }


def snapshot_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def distill_ratio_factor(schedule, train_ratio_tag):
    if schedule == "none":
        return 1.0
    if schedule == "low_resource_decay":
        return {
            10: 1.0,
            20: 0.7,
            50: 0.3,
        }.get(int(train_ratio_tag), 1.0)
    if schedule == "high_resource_decay":
        return {
            10: 1.0,
            20: 1.0,
            50: 0.3,
        }.get(int(train_ratio_tag), 1.0)
    if schedule == "sqrt_10_over_ratio":
        return min(1.0, float(np.sqrt(10.0 / max(float(train_ratio_tag), 1.0))))
    raise ValueError(f"Unknown distillation ratio schedule: {schedule}")


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
    selector_root=None,
    selector_model_name=None,
    multiteacher_strategy="uniform",
    selector_confidence_threshold=0.6,
    selector_distill_reweight=False,
    selector_distill_reweight_mode="global_confidence",
    lambda_distill=0.0,
    lambda_distill_ratio_schedule="none",
    lambda_gate_supervision=0.0,
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
        selector_root=selector_root,
        selector_model_name=selector_model_name,
        seed=seed,
    )

    model = build_model(model_name).to(DEVICE)

    dataset_cfg = DATASETS[dataset_name]
    higher_is_better = dataset_cfg["higher_is_better"]
    main_metric = dataset_cfg["main_metric"]
    uses_sample_level_gate = multiteacher_strategy in {
        "uncertainty_only",
        "uncertainty_validation_prior",
        "uncertainty_teacher_disagreement",
        "uncertainty_teacher_student",
        "uncertainty_student_prior",
        "uncertainty_composite",
        "learned_reliability_gate",
        "learned_linear_gate",
        "learned_prior_residual_gate",
        "supervised_hard_top1_gate",
        "supervised_hard_top2_gate",
        "pretrained_selector_top1",
        "pretrained_selector_confidence_fallback_top1",
        "pretrained_selector_validation_blend_top1",
        "pretrained_selector_top2",
        "selector_filtered_uniform",
        "selector_filtered_validation_weighted",
    }
    teacher_weights = None
    multiteacher_info = None
    gate_module = None
    resolved_selector_reweight_mode = selector_distill_reweight_mode
    selector_reweight_info = None
    if teacher_models is not None:
        if teacher_root is None:
            raise ValueError("--teacher-root is required when using --teacher-models")
        global_weight_strategy = (
            "validation_weighted"
            if multiteacher_strategy in {
                "uncertainty_validation_prior",
                "uncertainty_student_prior",
                "uncertainty_composite",
                "learned_reliability_gate",
                "learned_linear_gate",
                "learned_prior_residual_gate",
                "supervised_hard_top1_gate",
                "supervised_hard_top2_gate",
                "pretrained_selector_top1",
                "pretrained_selector_confidence_fallback_top1",
                "pretrained_selector_validation_blend_top1",
                "pretrained_selector_top2",
                "selector_filtered_validation_weighted",
            }
            else "uniform"
            if multiteacher_strategy in {
                "uncertainty_only",
                "uncertainty_teacher_disagreement",
                "uncertainty_teacher_student",
                "selector_filtered_uniform",
            }
            else multiteacher_strategy
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
    if selector_distill_reweight and selector_distill_reweight_mode == "auto":
        rng_state = snapshot_rng_state()
        resolved_selector_reweight_mode, selector_reweight_info = choose_selector_reweight_mode(
            valid_loader=valid_loader,
            task_type=task_type,
            teacher_weights=teacher_weights,
        )
        restore_rng_state(rng_state)
        if multiteacher_info is not None and selector_reweight_info is not None:
            multiteacher_info.update(selector_reweight_info)
    if multiteacher_strategy == "learned_reliability_gate":
        gate_module = TeacherReliabilityGate().to(DEVICE)
    elif multiteacher_strategy == "learned_linear_gate":
        gate_module = TeacherReliabilityLinearGate().to(DEVICE)
    elif multiteacher_strategy in {"learned_prior_residual_gate", "supervised_hard_top1_gate", "supervised_hard_top2_gate"}:
        gate_module = TeacherReliabilityPriorResidualGate().to(DEVICE)
    optim_params = list(model.parameters())
    if gate_module is not None:
        optim_params.extend(gate_module.parameters())
    optimizer = torch.optim.Adam(
        optim_params,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    lambda_distill_ratio_factor = distill_ratio_factor(lambda_distill_ratio_schedule, train_ratio_tag)
    lambda_distill_scheduled = lambda_distill * lambda_distill_ratio_factor
    lambda_distill_effective, adaptive_distill_info = resolve_lambda_distill(
        lambda_distill=lambda_distill_scheduled,
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
    if selector_reweight_info is not None:
        print(
            f"{dataset_name} | {model_name} | seed={seed} | "
            f"selector_reweight_mode={selector_reweight_info['resolved_selector_reweight_mode']} | "
            f"selector_valid_oracle_acc={selector_reweight_info['selector_valid_oracle_acc']:.4f} | "
            f"top1_valid_oracle_acc={selector_reweight_info['top1_valid_oracle_acc']:.4f}"
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
        gate_supervision_loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            fp = batch["fp"].to(DEVICE)
            desc = batch["desc"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            teacher_pred = batch.get("teacher_pred")
            teacher_uncertainty = batch.get("teacher_uncertainty")
            selector_probs = batch.get("selector_probs")
            if teacher_pred is not None:
                teacher_pred = teacher_pred.to(DEVICE)
            batch_teacher_weights = teacher_weights
            distill_sample_weights = None
            outputs = model(fp, desc)
            if uses_sample_level_gate:
                batch_teacher_weights = build_batch_teacher_weights(
                    strategy=multiteacher_strategy,
                    teacher_uncertainty=teacher_uncertainty,
                    teacher_pred=teacher_pred,
                    selector_probs=selector_probs,
                    student_pred=outputs["pred"].detach().view(-1),
                    global_teacher_weights=teacher_weights,
                    gate_module=gate_module,
                    selector_confidence_threshold=selector_confidence_threshold,
                )
            if (
                selector_distill_reweight
                and selector_probs is not None
                and teacher_models is not None
                and len(teacher_models) > 1
            ):
                selector_probs_device = selector_probs.to(DEVICE)
                selector_confidence = torch.max(selector_probs_device, dim=1).values
                if (
                    resolved_selector_reweight_mode == "disagreement_confidence"
                    and teacher_weights is not None
                ):
                    selector_top1_idx = torch.argmax(selector_probs_device, dim=1)
                    global_top1_idx = int(torch.argmax(teacher_weights).item())
                    disagreement_mask = selector_top1_idx != global_top1_idx
                    distill_sample_weights = torch.ones_like(selector_confidence)
                    distill_sample_weights[disagreement_mask] = (
                        selector_confidence[disagreement_mask] * selector_probs_device.shape[1]
                    )
                else:
                    distill_sample_weights = selector_confidence * selector_probs_device.shape[1]

            optimizer.zero_grad()
            total_loss, task_loss, transfer_loss, distill_loss = compute_total_loss(
                outputs=outputs,
                y=y,
                desc=desc,
                task_type=task_type,
                model_name=model_name,
                lambda_transfer=lambda_transfer,
                teacher_pred=teacher_pred,
                teacher_weights=batch_teacher_weights,
                distill_sample_weights=distill_sample_weights,
                lambda_distill=lambda_distill_effective,
            )
            gate_supervision_loss = torch.tensor(0.0, device=DEVICE)
            if (
                gate_module is not None
                and lambda_gate_supervision > 0
                and batch_teacher_weights is not None
                and teacher_pred is not None
                and teacher_pred.ndim == 2
                and teacher_pred.shape[1] > 1
            ):
                oracle_teacher_idx = torch.argmin(
                    torch.abs(teacher_pred - y.view(-1, 1)),
                    dim=1,
                )
                gate_supervision_loss = F.nll_loss(
                    torch.log(batch_teacher_weights.clamp_min(1e-8)),
                    oracle_teacher_idx,
                )
                total_loss = total_loss + lambda_gate_supervision * gate_supervision_loss
            total_loss.backward()
            optimizer.step()

            total_loss_sum += total_loss.item()
            task_loss_sum += task_loss.item()
            transfer_loss_sum += transfer_loss.item()
            distill_loss_sum += distill_loss.item()
            gate_supervision_loss_sum += gate_supervision_loss.item()
            n_batches += 1

        train_total_loss = total_loss_sum / max(n_batches, 1)
        train_task_loss = task_loss_sum / max(n_batches, 1)
        train_transfer_loss = transfer_loss_sum / max(n_batches, 1)
        train_distill_loss = distill_loss_sum / max(n_batches, 1)
        train_gate_supervision_loss = gate_supervision_loss_sum / max(n_batches, 1)

        valid_metrics, _ = evaluate_model(model, valid_loader, task_type)
        current_metric = valid_metrics[main_metric]

        history.append({
            "epoch": epoch,
            "train_total_loss": train_total_loss,
            "train_task_loss": train_task_loss,
            "train_transfer_loss": train_transfer_loss,
            "train_distill_loss": train_distill_loss,
            "train_gate_supervision_loss": train_gate_supervision_loss,
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
                "gate_state_dict": copy.deepcopy(gate_module.state_dict()) if gate_module is not None else None,
                "best_epoch": best_epoch,
                "best_metric": float(best_metric),
                "dataset": dataset_name,
                "task_type": task_type,
                "model": model_name,
                "seed": seed,
                "lambda_transfer": lambda_transfer,
                "lambda_distill": lambda_distill,
                "lambda_distill_ratio_schedule": lambda_distill_ratio_schedule,
                "lambda_distill_ratio_factor": lambda_distill_ratio_factor,
                "lambda_distill_scheduled": lambda_distill_scheduled,
                "lambda_distill_effective": lambda_distill_effective,
                "lambda_gate_supervision": lambda_gate_supervision,
                "adaptive_distill_strategy": adaptive_distill_strategy,
                "teacher_model": teacher_model,
                "teacher_models": teacher_models,
                "selector_model_name": selector_model_name,
                "multiteacher_strategy": multiteacher_strategy if teacher_models is not None else "single_teacher",
                "selector_distill_reweight": selector_distill_reweight,
                "selector_distill_reweight_mode": resolved_selector_reweight_mode if selector_distill_reweight else "none",
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
    if gate_module is not None and best_state.get("gate_state_dict") is not None:
        gate_module.load_state_dict(best_state["gate_state_dict"])
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
        "lambda_distill_ratio_schedule": lambda_distill_ratio_schedule,
        "lambda_distill_ratio_factor": lambda_distill_ratio_factor,
        "lambda_distill_scheduled": lambda_distill_scheduled,
        "lambda_distill_effective": lambda_distill_effective,
        "lambda_gate_supervision": lambda_gate_supervision,
        "adaptive_distill_strategy": adaptive_distill_strategy,
        "teacher_model": teacher_model,
        "teacher_models": teacher_models,
        "selector_model_name": selector_model_name,
        "multiteacher_strategy": multiteacher_strategy if teacher_models is not None else "single_teacher",
        "selector_distill_reweight": selector_distill_reweight,
        "selector_distill_reweight_mode": resolved_selector_reweight_mode if selector_distill_reweight else "none",
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
    parser.add_argument("--selector-root", default=None)
    parser.add_argument("--selector-model-name", default=None)
    parser.add_argument(
        "--multiteacher-strategy",
        choices=[
            "uniform",
            "validation_weighted",
            "top1_validation",
            "uncertainty_only",
            "uncertainty_validation_prior",
            "uncertainty_teacher_disagreement",
            "uncertainty_teacher_student",
            "uncertainty_student_prior",
            "uncertainty_composite",
            "learned_reliability_gate",
            "learned_linear_gate",
            "learned_prior_residual_gate",
            "supervised_hard_top1_gate",
            "supervised_hard_top2_gate",
            "pretrained_selector_top1",
            "pretrained_selector_confidence_fallback_top1",
            "pretrained_selector_validation_blend_top1",
            "pretrained_selector_top2",
            "selector_filtered_uniform",
            "selector_filtered_validation_weighted",
        ],
        default="uniform",
    )
    parser.add_argument("--selector-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--lambda-distill", type=float, default=0.0)
    parser.add_argument(
        "--lambda-distill-ratio-schedule",
        choices=["none", "low_resource_decay", "high_resource_decay", "sqrt_10_over_ratio"],
        default="none",
    )
    parser.add_argument("--lambda-gate-supervision", type=float, default=0.0)
    parser.add_argument("--selector-distill-reweight", action="store_true")
    parser.add_argument(
        "--selector-distill-reweight-mode",
        choices=["global_confidence", "disagreement_confidence", "auto"],
        default="global_confidence",
    )
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
    selector_strategies = {
        "pretrained_selector_top1",
        "pretrained_selector_confidence_fallback_top1",
        "pretrained_selector_validation_blend_top1",
        "pretrained_selector_top2",
        "selector_filtered_uniform",
        "selector_filtered_validation_weighted",
    }
    if args.multiteacher_strategy in selector_strategies:
        if args.teacher_models is None:
            raise ValueError(f"{args.multiteacher_strategy} requires --teacher-models")
        if args.selector_root is None or args.selector_model_name is None:
            raise ValueError(
                f"{args.multiteacher_strategy} requires --selector-root and --selector-model-name"
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
                        selector_root=args.selector_root,
                        selector_model_name=args.selector_model_name,
                        multiteacher_strategy=args.multiteacher_strategy,
                        selector_confidence_threshold=args.selector_confidence_threshold,
                        selector_distill_reweight=args.selector_distill_reweight,
                        selector_distill_reweight_mode=args.selector_distill_reweight_mode,
                        lambda_distill=args.lambda_distill,
                        lambda_distill_ratio_schedule=args.lambda_distill_ratio_schedule,
                        lambda_gate_supervision=args.lambda_gate_supervision,
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
