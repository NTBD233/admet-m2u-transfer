import torch
import torch.nn as nn


class ECFP4MLP(nn.Module):
    def __init__(self, fp_dim=2048, hidden_dim=512, z_dim=128):
        super().__init__()

        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.pred_head = nn.Linear(z_dim, 1)

    def forward(self, fp, desc=None):
        z_fp = self.fp_encoder(fp)
        pred = self.pred_head(z_fp)

        return {
            "pred": pred,
            "z_fp": z_fp,
        }


class ECFP4MLPDescPred(nn.Module):
    """
    Corrected lightweight M2U prototype from the notebook:
    ECFP4 predicts the downstream target and also predicts standardized RDKit
    descriptors during training. Inference still uses only ECFP4.
    """

    def __init__(self, fp_dim=2048, hidden_dim=512, z_dim=128, desc_dim=9):
        super().__init__()

        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.pred_head = nn.Linear(z_dim, 1)

        self.desc_pred_head = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, desc_dim),
        )

    def forward(self, fp, desc=None):
        z_fp = self.fp_encoder(fp)
        pred = self.pred_head(z_fp)
        desc_hat = self.desc_pred_head(z_fp)

        return {
            "pred": pred,
            "z_fp": z_fp,
            "desc_hat": desc_hat,
        }


class ECFP4MLPDescConcat(nn.Module):
    """
    Direct descriptor baseline:
    ECFP4 and standardized RDKit descriptors are concatenated as model input.
    Unlike DescPred, this model uses descriptors at inference time.
    """

    def __init__(self, fp_dim=2048, desc_dim=9, hidden_dim=512, z_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(fp_dim + desc_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.pred_head = nn.Linear(z_dim, 1)

    def forward(self, fp, desc=None):
        if desc is None:
            raise ValueError("ECFP4_MLP_DescConcat requires descriptor input.")

        z = self.encoder(torch.cat([fp, desc], dim=1))
        pred = self.pred_head(z)

        return {
            "pred": pred,
            "z_fp": z,
        }


class ECFP4MLPDescAdapterFusion(nn.Module):
    """
    Lightweight multi-to-uni variant:
    ECFP4 generates a pseudo descriptor representation, fuses it with the main
    ECFP4 representation, and predicts descriptors as an auxiliary task.
    Inference still uses only ECFP4.
    """

    def __init__(self, fp_dim=2048, hidden_dim=512, z_dim=128, desc_dim=9):
        super().__init__()

        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.desc_adapter = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(z_dim * 2, z_dim),
            nn.Sigmoid(),
        )

        self.pred_head = nn.Linear(z_dim, 1)
        self.desc_pred_head = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, desc_dim),
        )

    def forward(self, fp, desc=None):
        z_fp = self.fp_encoder(fp)
        z_desc_pseudo = self.desc_adapter(z_fp)

        gate = self.fusion_gate(torch.cat([z_fp, z_desc_pseudo], dim=1))
        z_fused = gate * z_fp + (1.0 - gate) * z_desc_pseudo

        pred = self.pred_head(z_fused)
        desc_hat = self.desc_pred_head(z_desc_pseudo)

        return {
            "pred": pred,
            "z_fp": z_fp,
            "z_desc_pseudo": z_desc_pseudo,
            "z_fused": z_fused,
            "fusion_gate": gate,
            "desc_hat": desc_hat,
        }


def build_model(model_name):
    if model_name == "ECFP4_MLP":
        return ECFP4MLP()
    if model_name == "ECFP4_MLP_DescPred":
        return ECFP4MLPDescPred()
    if model_name == "ECFP4_MLP_DescConcat":
        return ECFP4MLPDescConcat()
    if model_name == "ECFP4_MLP_DescAdapterFusion":
        return ECFP4MLPDescAdapterFusion()
    raise ValueError(f"Unknown model_name: {model_name}")
