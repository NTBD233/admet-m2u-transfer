import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def binary_roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC requires both positive and negative samples.")

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)

    start = 0
    while start < len(y_score):
        end = start + 1
        while end < len(y_score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_score_binary(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = int((y_sorted == 1).sum())
    if n_pos == 0:
        raise ValueError("Average precision requires at least one positive sample.")

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[y_sorted == 1].sum() / n_pos)


def compute_metrics(y_true, pred_raw, task_type):
    if task_type == "classification":
        pred_prob = sigmoid(pred_raw)
        return {
            "roc_auc": binary_roc_auc_score(y_true, pred_prob),
            "pr_auc": average_precision_score_binary(y_true, pred_prob),
        }, pred_prob

    if task_type == "regression":
        y_true = np.asarray(y_true)
        pred_raw = np.asarray(pred_raw)
        return {
            "mae": float(np.mean(np.abs(y_true - pred_raw))),
            "rmse": float(np.sqrt(np.mean((y_true - pred_raw) ** 2))),
        }, pred_raw

    raise ValueError(f"Unknown task_type: {task_type}")


def prediction_frame(smiles, y_true, pred, pred_raw):
    return pd.DataFrame({
        "smiles": smiles,
        "y_true": y_true,
        "pred": pred,
        "pred_raw": pred_raw,
    })
