import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

from utils.config import (
    DESC_COLS,
    FEATURE_ROOT,
    FP_RADIUS,
    FP_SIZE,
    PREPARED_DATA_ROOT,
    TRAIN_RATIO_TAG,
)


RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS,
    fpSize=FP_SIZE,
)


class DescriptorStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before transform.")
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean_) / self.scale_


def mol_from_smiles(smiles):
    if pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def calc_ecfp4(mol):
    fp = morgan_generator.GetFingerprint(mol)
    arr = np.zeros((FP_SIZE,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def calc_rdkit_descriptors(mol):
    desc = np.array([
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        Lipinski.HeavyAtomCount(mol),
        rdMolDescriptors.CalcNumRings(mol),
    ], dtype=np.float32)

    if not np.all(np.isfinite(desc)):
        return None
    return desc


def featurize_dataframe_for_m2u(df, smiles_col="Drug", label_col="Y"):
    X_fp_list = []
    X_desc_list = []
    y_list = []
    smiles_list = []
    kept_idx = []
    dropped = 0

    for idx, row in df.iterrows():
        mol = mol_from_smiles(row[smiles_col])
        if mol is None:
            dropped += 1
            continue

        desc = calc_rdkit_descriptors(mol)
        if desc is None:
            dropped += 1
            continue

        X_fp_list.append(calc_ecfp4(mol))
        X_desc_list.append(desc)
        y_list.append(row[label_col])
        smiles_list.append(str(row[smiles_col]))
        kept_idx.append(idx)

    return (
        np.asarray(X_fp_list, dtype=np.float32),
        np.asarray(X_desc_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(smiles_list, dtype=object),
        np.asarray(kept_idx, dtype=np.int64),
        dropped,
    )


def process_one_dataset(
    dataset_name,
    task_type,
    input_root=PREPARED_DATA_ROOT,
    output_root=FEATURE_ROOT,
    train_ratio_tag=TRAIN_RATIO_TAG,
):
    input_dir = Path(input_root) / dataset_name
    output_dir = Path(output_root) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    split_files = {
        f"train_{train_ratio_tag}": f"train_{train_ratio_tag}.csv",
        "valid": "valid.csv",
        "test": "test.csv",
    }

    raw_features = {}
    summary_rows = []

    for split_name, filename in split_files.items():
        csv_path = input_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "Drug" not in df.columns or "Y" not in df.columns:
            raise ValueError(f"{csv_path} must contain columns ['Drug', 'Y']")

        X_fp, X_desc_raw, y, smiles, kept_idx, dropped = featurize_dataframe_for_m2u(df)
        raw_features[split_name] = {
            "X_fp": X_fp,
            "X_desc_raw": X_desc_raw,
            "y": y,
            "smiles": smiles,
            "kept_idx": kept_idx,
            "dropped": dropped,
            "n_original": len(df),
            "n_valid": len(y),
        }

        summary_rows.append({
            "dataset": dataset_name,
            "task_type": task_type,
            "split": split_name,
            "input_file": str(csv_path),
            "n_original": len(df),
            "n_valid": len(y),
            "n_dropped": dropped,
            "fp_dim": X_fp.shape[1] if len(X_fp) > 0 else FP_SIZE,
            "desc_dim": X_desc_raw.shape[1] if len(X_desc_raw) > 0 else len(DESC_COLS),
        })

    train_split = f"train_{train_ratio_tag}"
    scaler = DescriptorStandardScaler()
    scaler.fit(raw_features[train_split]["X_desc_raw"])

    with (output_dir / f"desc_scaler_{train_ratio_tag}.pkl").open("wb") as f:
        pickle.dump(scaler, f)

    for split_name in [train_split, "valid", "test"]:
        features = raw_features[split_name]
        X_desc = scaler.transform(features["X_desc_raw"]).astype(np.float32)
        if split_name == train_split:
            save_path = output_dir / f"train_{train_ratio_tag}_features.npz"
        else:
            save_path = output_dir / f"{split_name}_{train_ratio_tag}_features.npz"

        np.savez_compressed(
            save_path,
            X_fp=features["X_fp"].astype(np.float32),
            X_desc=X_desc,
            X_desc_raw=features["X_desc_raw"].astype(np.float32),
            y=features["y"].astype(np.float32),
            smiles=features["smiles"],
            kept_idx=features["kept_idx"],
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / f"feature_summary_{train_ratio_tag}.csv", index=False)

    desc_stats_df = pd.DataFrame(
        raw_features[train_split]["X_desc_raw"],
        columns=DESC_COLS,
    ).describe().T
    desc_stats_df.to_csv(output_dir / f"train_{train_ratio_tag}_descriptor_stats.csv")

    if train_ratio_tag == TRAIN_RATIO_TAG:
        with (output_dir / "desc_scaler.pkl").open("wb") as f:
            pickle.dump(scaler, f)
        for legacy_name, ratio_name in [
            ("valid_features.npz", f"valid_{train_ratio_tag}_features.npz"),
            ("test_features.npz", f"test_{train_ratio_tag}_features.npz"),
        ]:
            legacy_path = output_dir / legacy_name
            ratio_path = output_dir / ratio_name
            if ratio_path.exists():
                legacy_path.write_bytes(ratio_path.read_bytes())
        legacy_summary = output_dir / "feature_summary.csv"
        legacy_summary.write_text(summary_df.to_csv(index=False), encoding="utf-8")

    return summary_df


def generate_features_for_datasets(
    datasets,
    input_root=PREPARED_DATA_ROOT,
    output_root=FEATURE_ROOT,
    train_ratio_tags=None,
):
    if train_ratio_tags is None:
        train_ratio_tags = [TRAIN_RATIO_TAG]

    summaries = []
    for dataset_name, cfg in datasets.items():
        for train_ratio_tag in train_ratio_tags:
            summaries.append(
                process_one_dataset(
                    dataset_name=dataset_name,
                    task_type=cfg["task_type"],
                    input_root=input_root,
                    output_root=output_root,
                    train_ratio_tag=train_ratio_tag,
                )
            )

    all_summary_df = pd.concat(summaries, ignore_index=True)
    Path(output_root).mkdir(parents=True, exist_ok=True)
    all_summary_df.to_csv(Path(output_root) / "all_feature_summary.csv", index=False)
    return all_summary_df
