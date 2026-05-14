from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.config import BATCH_SIZE, FEATURE_ROOT, TRAIN_RATIO_TAG


class M2UFeatureDataset(Dataset):
    def __init__(self, npz_path, task_type, teacher_path=None, teacher_paths=None):
        data = np.load(npz_path, allow_pickle=True)

        self.X_fp = torch.tensor(data["X_fp"], dtype=torch.float32)
        self.X_desc = torch.tensor(data["X_desc"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.float32).view(-1, 1)
        self.smiles = data["smiles"]
        self.task_type = task_type
        self.teacher_pred = None
        self.teacher_uncertainty = None
        if teacher_path is not None:
            teacher_data = np.load(teacher_path, allow_pickle=True)
            self.teacher_pred = torch.tensor(
                teacher_data["pred"],
                dtype=torch.float32,
            ).view(-1, 1)
            if len(self.teacher_pred) != len(self.y):
                raise ValueError(
                    f"Teacher predictions length mismatch for {teacher_path}: "
                    f"{len(self.teacher_pred)} != {len(self.y)}"
                )
            if "pred_uncertainty" in teacher_data.files:
                self.teacher_uncertainty = torch.tensor(
                    teacher_data["pred_uncertainty"],
                    dtype=torch.float32,
                ).view(-1, 1)
        if teacher_paths is not None:
            preds = []
            uncertainties = []
            for path in teacher_paths:
                teacher_data = np.load(path, allow_pickle=True)
                pred = torch.tensor(
                    teacher_data["pred"],
                    dtype=torch.float32,
                ).view(-1, 1)
                if len(pred) != len(self.y):
                    raise ValueError(
                        f"Teacher predictions length mismatch for {path}: "
                        f"{len(pred)} != {len(self.y)}"
                    )
                preds.append(pred)
                if "pred_uncertainty" in teacher_data.files:
                    uncertainty = torch.tensor(
                        teacher_data["pred_uncertainty"],
                        dtype=torch.float32,
                    ).view(-1, 1)
                else:
                    uncertainty = torch.full_like(pred, float("nan"))
                uncertainties.append(uncertainty)
            self.teacher_pred = torch.cat(preds, dim=1)
            self.teacher_uncertainty = torch.cat(uncertainties, dim=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item = {
            "fp": self.X_fp[idx],
            "desc": self.X_desc[idx],
            "y": self.y[idx],
            "smiles": self.smiles[idx],
        }
        if self.teacher_pred is not None:
            item["teacher_pred"] = self.teacher_pred[idx]
        if self.teacher_uncertainty is not None:
            item["teacher_uncertainty"] = self.teacher_uncertainty[idx]
        return item


def feature_paths(dataset_name, feature_root=FEATURE_ROOT, train_ratio_tag=TRAIN_RATIO_TAG):
    feature_root = Path(feature_root)
    train_path = feature_root / dataset_name / f"train_{TRAIN_RATIO_TAG}_features.npz"
    if train_ratio_tag != TRAIN_RATIO_TAG:
        train_path = feature_root / dataset_name / f"train_{train_ratio_tag}_features.npz"

    valid_ratio_path = feature_root / dataset_name / f"valid_{train_ratio_tag}_features.npz"
    test_ratio_path = feature_root / dataset_name / f"test_{train_ratio_tag}_features.npz"
    valid_path = valid_ratio_path if valid_ratio_path.exists() else feature_root / dataset_name / "valid_features.npz"
    test_path = test_ratio_path if test_ratio_path.exists() else feature_root / dataset_name / "test_features.npz"

    return train_path, valid_path, test_path


def make_loaders(
    dataset_name,
    task_type,
    batch_size=BATCH_SIZE,
    feature_root=FEATURE_ROOT,
    train_ratio_tag=TRAIN_RATIO_TAG,
    teacher_root=None,
    teacher_model=None,
    teacher_models=None,
    seed=None,
):
    train_path, valid_path, test_path = feature_paths(
        dataset_name=dataset_name,
        feature_root=feature_root,
        train_ratio_tag=train_ratio_tag,
    )

    teacher_train_path = None
    teacher_train_paths = None
    if teacher_root is not None and teacher_model is not None and seed is not None:
        teacher_train_path = (
            Path(teacher_root)
            / dataset_name
            / teacher_model
            / f"train_{train_ratio_tag}"
            / f"seed_{seed}"
            / f"train_{train_ratio_tag}_teacher_predictions.npz"
        )
        if not teacher_train_path.exists():
            raise FileNotFoundError(f"Missing teacher predictions: {teacher_train_path}")
    if teacher_root is not None and teacher_models is not None and seed is not None:
        teacher_train_paths = []
        for teacher_name in teacher_models:
            teacher_path = (
                Path(teacher_root)
                / dataset_name
                / teacher_name
                / f"train_{train_ratio_tag}"
                / f"seed_{seed}"
                / f"train_{train_ratio_tag}_teacher_predictions.npz"
            )
            if not teacher_path.exists():
                raise FileNotFoundError(f"Missing teacher predictions: {teacher_path}")
            teacher_train_paths.append(teacher_path)

    train_ds = M2UFeatureDataset(
        train_path,
        task_type,
        teacher_path=teacher_train_path,
        teacher_paths=teacher_train_paths,
    )
    valid_ds = M2UFeatureDataset(valid_path, task_type)
    test_ds = M2UFeatureDataset(test_path, task_type)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
