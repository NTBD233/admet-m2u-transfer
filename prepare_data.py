import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from utils.config import DATASETS, PROJECT_ROOT


DATA_URL = "https://dataverse.harvard.edu/api/access/datafile/4426004"
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PREPARED_ROOT = PROJECT_ROOT / "data" / "prepared_data"
SUBSAMPLE_ROOT = PROJECT_ROOT / "data" / "prepared_data_subsample"


def download_file(url, output_path):
    import requests

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Using existing file: {output_path}")
        return output_path

    print(f"Downloading: {url}")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    print(f"Saved: {output_path}")
    return output_path


def extract_zip(zip_path, extract_dir):
    marker_dir = extract_dir / "admet_group"
    if marker_dir.exists():
        print(f"Using existing extracted directory: {extract_dir}")
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"Extracted to: {extract_dir}")
    return extract_dir


def split_train_valid(df, task_type, test_size=0.2, random_state=42):
    if task_type == "classification":
        valid_df = (
            df.groupby("Y", group_keys=False)
            .sample(frac=test_size, random_state=random_state)
            .sort_index()
        )
        train_df = df.drop(valid_df.index)
    else:
        valid_df = df.sample(frac=test_size, random_state=random_state).sort_index()
        train_df = df.drop(valid_df.index)

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def make_subsample_train(df, ratio, task_type, seed=42):
    if ratio == 1.0:
        return df.reset_index(drop=True)

    if task_type == "classification":
        sub_df = (
            df.groupby("Y", group_keys=False)
            .sample(frac=ratio, random_state=seed)
            .sort_index()
        )
    else:
        sub_df = df.sample(frac=ratio, random_state=seed).sort_index()

    return sub_df.reset_index(drop=True)


def prepare_dataset(dataset_name, task_type, extracted_root):
    source_dir = extracted_root / "admet_group" / dataset_name
    train_val_path = source_dir / "train_val.csv"
    test_path = source_dir / "test.csv"

    if not train_val_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing source CSV files under: {source_dir}")

    train_val_df = pd.read_csv(train_val_path)
    test_df = pd.read_csv(test_path)
    train_df, valid_df = split_train_valid(train_val_df, task_type=task_type)

    prepared_dir = PREPARED_ROOT / dataset_name
    prepared_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(prepared_dir / "train.csv", index=False)
    valid_df.to_csv(prepared_dir / "valid.csv", index=False)
    test_df.to_csv(prepared_dir / "test.csv", index=False)

    subsample_dir = SUBSAMPLE_ROOT / dataset_name
    subsample_dir.mkdir(parents=True, exist_ok=True)
    valid_df.to_csv(subsample_dir / "valid.csv", index=False)
    test_df.to_csv(subsample_dir / "test.csv", index=False)

    for ratio in [1.0, 0.5, 0.2, 0.1]:
        ratio_tag = int(ratio * 100)
        sub_df = make_subsample_train(train_df, ratio=ratio, task_type=task_type, seed=42)
        sub_df.to_csv(subsample_dir / f"train_{ratio_tag}.csv", index=False)

    print(
        f"{dataset_name}: train={len(train_df)}, valid={len(valid_df)}, "
        f"test={len(test_df)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ADMET data from the original notebook logic.")
    parser.add_argument("--zip-path", default=None, help="Path to an existing admet_group.zip file.")
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    zip_path = RAW_ROOT / "admet_group.zip"
    extract_dir = RAW_ROOT

    if args.zip_path is not None:
        source_zip = Path(args.zip_path).expanduser()
        if not source_zip.exists():
            raise FileNotFoundError(f"Zip file not found: {source_zip}")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if source_zip.resolve() != zip_path.resolve():
            shutil.copy2(source_zip, zip_path)
        print(f"Using zip file: {zip_path}")
    elif not args.skip_download:
        download_file(DATA_URL, zip_path)
    elif not zip_path.exists():
        raise FileNotFoundError(f"Expected existing zip file: {zip_path}")

    extracted_root = extract_zip(zip_path, extract_dir)

    for dataset_name, cfg in DATASETS.items():
        prepare_dataset(
            dataset_name=dataset_name,
            task_type=cfg["task_type"],
            extracted_root=extracted_root,
        )


if __name__ == "__main__":
    main()
