import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.summary import dataframe_to_markdown


def teacher_npz_path(teacher_root, dataset, teacher, ratio, seed, split):
    filename = f"train_{ratio}_teacher_predictions.npz" if split == "train" else f"{split}_teacher_predictions.npz"
    return (
        Path(teacher_root)
        / dataset
        / teacher
        / f"train_{ratio}"
        / f"seed_{seed}"
        / filename
    )


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_selector_npz(selector_root, dataset, selector_model_name, ratio, seed, split):
    path = (
        Path(selector_root)
        / dataset
        / selector_model_name
        / f"train_{ratio}"
        / f"seed_{seed}"
        / f"{split}_selector_predictions.npz"
    )
    z = np.load(path, allow_pickle=True)
    return {
        "path": path,
        "probs": z["probs"].astype(float),
        "pred_idx": z["pred_idx"].astype(int),
        "oracle_idx": z["oracle_idx"].astype(int),
        "smiles": z["smiles"].astype(str),
        "y": z["y"].astype(float),
        "teachers": z["teachers"].astype(str),
    }


def load_teacher_split(teacher_root, dataset, teacher, ratio, seed, split):
    path = teacher_npz_path(teacher_root, dataset, teacher, ratio, seed, split)
    z = np.load(path, allow_pickle=True)
    return {
        "path": path,
        "pred": z["pred"].astype(float),
        "y": z["y"].astype(float),
        "smiles": z["smiles"].astype(str),
    }


def load_student_predictions(results_root, dataset, ratio, seed, split):
    path = (
        Path(results_root)
        / dataset
        / "ECFP4_MLP_DescAdapterFusion"
        / f"train_{ratio}"
        / f"seed_{seed}"
        / f"{split}_predictions.csv"
    )
    df = pd.read_csv(path)
    out = df[["smiles", "y_true", "pred"]].copy()
    out["smiles"] = out["smiles"].astype(str)
    out = out.rename(columns={"y_true": "y_student", "pred": "student_pred"})
    return {
        "path": path,
        "df": out,
    }


def infer_top1_teacher(metrics_path, teachers):
    metrics = load_json(metrics_path)
    weight_map = metrics["teacher_weight_map"]
    weights = np.asarray([weight_map[t] for t in teachers], dtype=float)
    return int(np.argmax(weights)), metrics


def build_split_frame(
    dataset,
    ratio,
    seed,
    split,
    teachers,
    selector_root,
    selector_model_name,
    teacher_root,
    selector_student_root,
    top1_student_root,
):
    selector = load_selector_npz(selector_root, dataset, selector_model_name, ratio, seed, split)
    teacher_preds = []
    for teacher in teachers:
        teacher_preds.append(load_teacher_split(teacher_root, dataset, teacher, ratio, seed, split))

    first = teacher_preds[0]
    for rec in teacher_preds[1:]:
        if not np.array_equal(first["smiles"], rec["smiles"]):
            raise ValueError(f"Teacher smiles misaligned for {dataset} {split}")
        if not np.allclose(first["y"], rec["y"]):
            raise ValueError(f"Teacher y misaligned for {dataset} {split}")
    if not np.array_equal(first["smiles"], selector["smiles"]):
        raise ValueError(f"Selector smiles misaligned for {dataset} {split}")

    teacher_matrix = np.column_stack([rec["pred"] for rec in teacher_preds])
    teacher_abs_err = np.abs(teacher_matrix - selector["y"].reshape(-1, 1))

    df = pd.DataFrame(
        {
            "smiles": selector["smiles"],
            "y": selector["y"],
            "selector_pred_idx": selector["pred_idx"],
            "oracle_idx": selector["oracle_idx"],
            "selector_confidence": selector["probs"].max(axis=1),
        }
    )
    sorted_probs = np.sort(selector["probs"], axis=1)
    df["selector_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    for idx, teacher in enumerate(teachers):
        df[f"{teacher}_pred"] = teacher_matrix[:, idx]
        df[f"{teacher}_abs_err"] = teacher_abs_err[:, idx]

    selector_student = load_student_predictions(selector_student_root, dataset, ratio, seed, split)["df"]
    top1_student = load_student_predictions(top1_student_root, dataset, ratio, seed, split)["df"]
    selector_student = selector_student.rename(columns={"student_pred": "selector_student_pred"})
    top1_student = top1_student.rename(columns={"student_pred": "top1_student_pred"})

    for name, student_df in [("selector", selector_student), ("top1", top1_student)]:
        if len(student_df) != len(df):
            raise ValueError(f"{name} student length mismatch for {dataset} {split}")
        if not np.array_equal(student_df["smiles"].to_numpy(), df["smiles"].to_numpy()):
            raise ValueError(f"{name} student smiles misaligned for {dataset} {split}")
        if not np.allclose(student_df["y_student"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float), equal_nan=True):
            raise ValueError(f"{name} student y mismatch for {dataset} {split}")

    top1_teacher_idx, top1_metrics = infer_top1_teacher(
        Path(top1_student_root) / dataset / "ECFP4_MLP_DescAdapterFusion" / f"train_{ratio}" / f"seed_{seed}" / "metrics.json",
        teachers,
    )
    df["top1_validation_idx"] = top1_teacher_idx
    df["top1_validation_teacher"] = teachers[top1_teacher_idx]
    df["selector_teacher"] = [teachers[i] for i in df["selector_pred_idx"]]
    df["oracle_teacher"] = [teachers[i] for i in df["oracle_idx"]]
    df["selector_teacher_abs_err"] = teacher_abs_err[np.arange(len(df)), df["selector_pred_idx"]]
    df["top1_validation_abs_err"] = teacher_abs_err[:, top1_teacher_idx]
    df["selector_beats_top1_teacher"] = df["selector_teacher_abs_err"] < df["top1_validation_abs_err"]
    df["selector_matches_top1_teacher"] = df["selector_pred_idx"] == top1_teacher_idx
    df["selector_hits_oracle"] = df["selector_pred_idx"] == df["oracle_idx"]
    df["top1_hits_oracle"] = df["top1_validation_idx"] == df["oracle_idx"]

    df["selector_student_pred"] = selector_student["selector_student_pred"].to_numpy(dtype=float)
    df["top1_student_pred"] = top1_student["top1_student_pred"].to_numpy(dtype=float)
    df["selector_student_abs_err"] = np.abs(df["selector_student_pred"] - df["y"])
    df["top1_student_abs_err"] = np.abs(df["top1_student_pred"] - df["y"])
    df["selector_student_beats_top1_student"] = df["selector_student_abs_err"] < df["top1_student_abs_err"]
    df["student_abs_err_delta"] = df["selector_student_abs_err"] - df["top1_student_abs_err"]

    return df, top1_metrics


def summarize_subset(name, subdf):
    if len(subdf) == 0:
        return {
            "subset": name,
            "n": 0,
            "selector_oracle_acc": np.nan,
            "top1_oracle_acc": np.nan,
            "selector_teacher_beats_top1_rate": np.nan,
            "selector_student_beats_top1_rate": np.nan,
            "mean_student_abs_err_delta": np.nan,
        }
    return {
        "subset": name,
        "n": int(len(subdf)),
        "selector_oracle_acc": round(float(subdf["selector_hits_oracle"].mean()), 4),
        "top1_oracle_acc": round(float(subdf["top1_hits_oracle"].mean()), 4),
        "selector_teacher_beats_top1_rate": round(float(subdf["selector_beats_top1_teacher"].mean()), 4),
        "selector_student_beats_top1_rate": round(float(subdf["selector_student_beats_top1_student"].mean()), 4),
        "mean_student_abs_err_delta": round(float(subdf["student_abs_err_delta"].mean()), 4),
    }


def analyze_setting(args):
    teachers = args.teachers
    all_rows = []
    per_split_details = {}
    for split in ["valid", "test"]:
        split_df, top1_metrics = build_split_frame(
            dataset=args.dataset,
            ratio=args.train_ratio_tag,
            seed=args.seed,
            split=split,
            teachers=teachers,
            selector_root=args.selector_root,
            selector_model_name=args.selector_model_name,
            teacher_root=args.teacher_root,
            selector_student_root=args.selector_student_root,
            top1_student_root=args.top1_student_root,
        )
        per_split_details[split] = {
            "top1_metrics": top1_metrics,
            "df": split_df,
        }
        subsets = {
            "all": split_df,
            "agree": split_df[split_df["selector_matches_top1_teacher"]],
            "disagree": split_df[~split_df["selector_matches_top1_teacher"]],
            "disagree_selector_better_teacher": split_df[
                (~split_df["selector_matches_top1_teacher"]) & split_df["selector_beats_top1_teacher"]
            ],
            "disagree_selector_worse_teacher": split_df[
                (~split_df["selector_matches_top1_teacher"]) & (~split_df["selector_beats_top1_teacher"])
            ],
        }
        for subset_name, subset_df in subsets.items():
            row = summarize_subset(f"{split}:{subset_name}", subset_df)
            all_rows.append(row)
    return pd.DataFrame(all_rows), per_split_details


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze selector failure modes for a specific setting.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-ratio-tag", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--teachers", nargs="+", default=["ECFP4_RF", "Desc_RF", "ECFP4_Desc_RF"])
    parser.add_argument("--selector-root", default="data/selector_predictions")
    parser.add_argument("--selector-model-name", default="rf_crossfit_train_pseudo_oracle")
    parser.add_argument("--teacher-root", default="data/teacher_predictions")
    parser.add_argument("--selector-student-root", default="results_pretrained_selector_top1_regression")
    parser.add_argument("--top1-student-root", default="results_multiteacher_top1")
    parser.add_argument("--output-root", default="results_selector_failure_analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    summary_df, detail = analyze_setting(args)

    out_dir = (
        Path(args.output_root)
        / args.dataset
        / f"train_{args.train_ratio_tag}"
        / f"seed_{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print("## Selector Failure Summary")
    print(dataframe_to_markdown(summary_df))

    for split, rec in detail.items():
        df = rec["df"]
        df.to_csv(out_dir / f"{split}_sample_analysis.csv", index=False)
        top_disagreements = df.loc[
            ~df["selector_matches_top1_teacher"],
            [
                "smiles",
                "y",
                "selector_teacher",
                "oracle_teacher",
                "top1_validation_teacher",
                "selector_confidence",
                "selector_margin",
                "selector_teacher_abs_err",
                "top1_validation_abs_err",
                "selector_student_abs_err",
                "top1_student_abs_err",
                "student_abs_err_delta",
            ],
        ].copy()
        top_disagreements["teacher_abs_err_delta"] = (
            top_disagreements["selector_teacher_abs_err"] - top_disagreements["top1_validation_abs_err"]
        )
        top_disagreements = top_disagreements.sort_values("student_abs_err_delta", ascending=False)
        top_disagreements.head(25).to_csv(out_dir / f"{split}_top_disagreements.csv", index=False)


if __name__ == "__main__":
    main()
