import json
import math
from pathlib import Path

import pandas as pd


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, path):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=4)
    tmp_path.replace(path)


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value) and not isinstance(value, (str, bytes, bool)):
        return None
    return value


def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_run_outputs(save_dir, best_state, history, valid_pred_df, test_pred_df, metrics):
    import torch

    save_dir = ensure_dir(save_dir)
    torch.save(best_state, save_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)
    valid_pred_df.to_csv(save_dir / "valid_predictions.csv", index=False)
    test_pred_df.to_csv(save_dir / "test_predictions.csv", index=False)
    save_json(metrics, save_dir / "metrics.json")
