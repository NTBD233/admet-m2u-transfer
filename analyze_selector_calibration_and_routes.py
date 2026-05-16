import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_selector_failure_modes import load_selector_npz
from utils.config import SEEDS, TRAIN_RATIO_TAGS
from utils.summary import dataframe_to_markdown


DEFAULT_TEACHERS = ["ECFP4_RF", "Desc_RF", "ECFP4_Desc_RF"]


def load_metrics(results_root, dataset, ratio, seed):
    path = (
        Path(results_root)
        / dataset
        / "ECFP4_MLP_DescAdapterFusion"
        / f"train_{ratio}"
        / f"seed_{seed}"
        / "metrics.json"
    )
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def top1_teacher_idx(metrics, teachers):
    weights = metrics.get("teacher_weight_map")
    if weights is None:
        return None
    return int(np.argmax([weights[teacher] for teacher in teachers]))


def split_selector_frame(selector_root, selector_model_name, dataset, ratio, seed, split, teachers, top1_idx):
    selector = load_selector_npz(selector_root, dataset, selector_model_name, ratio, seed, split)
    probs = selector["probs"]
    pred_idx = selector["pred_idx"]
    oracle_idx = selector["oracle_idx"]
    sorted_probs = np.sort(probs, axis=1)
    df = pd.DataFrame(
        {
            "dataset": dataset,
            "train_ratio_tag": ratio,
            "seed": seed,
            "split": split,
            "sample_idx": np.arange(len(pred_idx)),
            "selector_pred_idx": pred_idx,
            "oracle_idx": oracle_idx,
            "top1_idx": top1_idx,
            "selector_confidence": probs.max(axis=1),
            "selector_margin": sorted_probs[:, -1] - sorted_probs[:, -2],
            "selector_hits_oracle": pred_idx == oracle_idx,
            "top1_hits_oracle": oracle_idx == top1_idx,
        }
    )
    df["selector_teacher"] = [teachers[idx] for idx in df["selector_pred_idx"]]
    df["oracle_teacher"] = [teachers[idx] for idx in df["oracle_idx"]]
    df["top1_teacher"] = teachers[top1_idx]
    return df


def calibration_bins(sample_df):
    if sample_df.empty:
        return pd.DataFrame()
    bins = [0.0, 1 / 3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]
    labels = ["<=0.33", "0.33-0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]
    df = sample_df.copy()
    df["confidence_bin"] = pd.cut(
        df["selector_confidence"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    rows = []
    group_cols = ["split", "confidence_bin"]
    for keys, sub in df.groupby(group_cols, observed=False):
        split, confidence_bin = keys
        if sub.empty:
            continue
        rows.append(
            {
                "split": split,
                "confidence_bin": str(confidence_bin),
                "n": int(len(sub)),
                "mean_confidence": float(sub["selector_confidence"].mean()),
                "oracle_accuracy": float(sub["selector_hits_oracle"].mean()),
                "calibration_gap": float(sub["selector_confidence"].mean() - sub["selector_hits_oracle"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    for col in ["mean_confidence", "oracle_accuracy", "calibration_gap"]:
        out[col] = out[col].round(4)
    return out


def calibration_summary(sample_df):
    rows = []
    for keys, sub in sample_df.groupby(["dataset", "split"]):
        dataset, split = keys
        n = len(sub)
        if n == 0:
            continue
        bins = calibration_bins(sub)
        if bins.empty:
            ece = np.nan
        else:
            split_bins = bins[bins["split"] == split]
            ece = float((split_bins["n"] * split_bins["calibration_gap"].abs()).sum() / max(split_bins["n"].sum(), 1))
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "n": int(n),
                "mean_confidence": float(sub["selector_confidence"].mean()),
                "oracle_accuracy": float(sub["selector_hits_oracle"].mean()),
                "mean_margin": float(sub["selector_margin"].mean()),
                "ece": ece,
            }
        )
    out = pd.DataFrame(rows)
    for col in ["mean_confidence", "oracle_accuracy", "mean_margin", "ece"]:
        out[col] = out[col].round(4)
    return out.sort_values(["dataset", "split"], kind="stable")


def route_mode(selector_acc, top1_acc):
    return "global_confidence" if selector_acc > top1_acc else "disagreement_confidence"


def build_setting_rows(sample_df, auto_root, plain_root, top1_root, fixed_root, teachers):
    rows = []
    for keys, setting_df in sample_df.groupby(["dataset", "train_ratio_tag", "seed"]):
        dataset, ratio, seed = keys
        valid_df = setting_df[setting_df["split"] == "valid"]
        test_df = setting_df[setting_df["split"] == "test"]
        if valid_df.empty or test_df.empty:
            continue

        valid_selector_acc = float(valid_df["selector_hits_oracle"].mean())
        valid_top1_acc = float(valid_df["top1_hits_oracle"].mean())
        test_selector_acc = float(test_df["selector_hits_oracle"].mean())
        test_top1_acc = float(test_df["top1_hits_oracle"].mean())
        valid_mode = route_mode(valid_selector_acc, valid_top1_acc)
        test_oracle_mode = route_mode(test_selector_acc, test_top1_acc)

        auto_metrics = load_metrics(auto_root, dataset, ratio, seed)
        plain_metrics = load_metrics(plain_root, dataset, ratio, seed)
        top1_metrics = load_metrics(top1_root, dataset, ratio, seed)
        fixed_metrics = load_metrics(fixed_root, dataset, ratio, seed)
        top1_idx = top1_teacher_idx(top1_metrics, teachers) if top1_metrics is not None else None

        row = {
            "dataset": dataset,
            "train_ratio_tag": int(ratio),
            "seed": int(seed),
            "top1_teacher": teachers[top1_idx] if top1_idx is not None else None,
            "valid_selector_acc": valid_selector_acc,
            "valid_top1_acc": valid_top1_acc,
            "test_selector_acc": test_selector_acc,
            "test_top1_acc": test_top1_acc,
            "valid_selected_mode": valid_mode,
            "test_oracle_mode": test_oracle_mode,
            "mode_matches_test_oracle": valid_mode == test_oracle_mode,
        }
        if auto_metrics is not None:
            row["auto_mode_recorded"] = auto_metrics.get("selector_distill_reweight_mode")
            row["auto_test_rmse"] = auto_metrics.get("test_rmse")
        if plain_metrics is not None:
            row["plain_test_rmse"] = plain_metrics.get("test_rmse")
        if top1_metrics is not None:
            row["top1_test_rmse"] = top1_metrics.get("test_rmse")
        if fixed_metrics is not None:
            row["fixed_test_rmse"] = fixed_metrics.get("test_rmse")
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for baseline in ["plain", "top1", "fixed"]:
        if f"{baseline}_test_rmse" in out.columns and "auto_test_rmse" in out.columns:
            out[f"auto_delta_vs_{baseline}"] = out["auto_test_rmse"] - out[f"{baseline}_test_rmse"]
    numeric_cols = [col for col in out.columns if col.endswith("_acc") or col.endswith("_rmse") or col.startswith("auto_delta")]
    for col in numeric_cols:
        out[col] = out[col].round(4)
    return out.sort_values(["dataset", "train_ratio_tag", "seed"], kind="stable")


def summarize_routes(setting_df):
    if setting_df.empty:
        return pd.DataFrame()
    rows = []
    for keys, sub in setting_df.groupby(["dataset"]):
        dataset = keys[0] if isinstance(keys, tuple) else keys
        row = {
            "dataset": dataset,
            "settings": int(len(sub)),
            "mode_match_rate": float(sub["mode_matches_test_oracle"].mean()),
            "auto_wins_vs_plain": int((sub["auto_delta_vs_plain"] < 0).sum()) if "auto_delta_vs_plain" in sub else np.nan,
            "auto_wins_vs_top1": int((sub["auto_delta_vs_top1"] < 0).sum()) if "auto_delta_vs_top1" in sub else np.nan,
            "mean_delta_vs_plain": float(sub["auto_delta_vs_plain"].mean()) if "auto_delta_vs_plain" in sub else np.nan,
            "mean_delta_vs_top1": float(sub["auto_delta_vs_top1"].mean()) if "auto_delta_vs_top1" in sub else np.nan,
        }
        rows.append(row)
    rows.append(
        {
            "dataset": "ALL",
            "settings": int(len(setting_df)),
            "mode_match_rate": float(setting_df["mode_matches_test_oracle"].mean()),
            "auto_wins_vs_plain": int((setting_df["auto_delta_vs_plain"] < 0).sum()) if "auto_delta_vs_plain" in setting_df else np.nan,
            "auto_wins_vs_top1": int((setting_df["auto_delta_vs_top1"] < 0).sum()) if "auto_delta_vs_top1" in setting_df else np.nan,
            "mean_delta_vs_plain": float(setting_df["auto_delta_vs_plain"].mean()) if "auto_delta_vs_plain" in setting_df else np.nan,
            "mean_delta_vs_top1": float(setting_df["auto_delta_vs_top1"].mean()) if "auto_delta_vs_top1" in setting_df else np.nan,
        }
    )
    out = pd.DataFrame(rows)
    for col in ["mode_match_rate", "mean_delta_vs_plain", "mean_delta_vs_top1"]:
        out[col] = out[col].round(4)
    return out


def collect_samples(args):
    rows = []
    for dataset in args.datasets:
        for ratio in args.train_ratio_tags:
            for seed in args.seeds:
                top1_metrics = load_metrics(args.top1_root, dataset, ratio, seed)
                if top1_metrics is None:
                    continue
                top1_idx = top1_teacher_idx(top1_metrics, args.teachers)
                for split in ["valid", "test"]:
                    try:
                        rows.append(
                            split_selector_frame(
                                selector_root=args.selector_root,
                                selector_model_name=args.selector_model_name,
                                dataset=dataset,
                                ratio=ratio,
                                seed=seed,
                                split=split,
                                teachers=args.teachers,
                                top1_idx=top1_idx,
                            )
                        )
                    except FileNotFoundError:
                        continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def write_report(output_root, calibration_summary_df, calibration_bins_df, route_summary_df, setting_df):
    lines = [
        "# Selector Calibration And Route Audit",
        "",
        "## Calibration Summary",
        "",
        dataframe_to_markdown(calibration_summary_df),
        "",
        "## Route Audit Summary",
        "",
        dataframe_to_markdown(route_summary_df),
        "",
        "## Key Reading",
        "",
        "- `ece` is the expected calibration error using fixed confidence bins.",
        "- `mode_match_rate` measures whether validation-split mode selection matches the test-split teacher-selection advantage.",
        "- Positive `mean_delta_vs_*` means auto reweighting is worse on RMSE.",
        "",
        "## Per-Setting Route Audit",
        "",
        dataframe_to_markdown(setting_df),
    ]
    (output_root / "selector_calibration_route_audit.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze selector confidence calibration and validation route decisions.")
    parser.add_argument("--datasets", nargs="+", default=["caco2_wang", "ppbr_az"])
    parser.add_argument("--teachers", nargs="+", default=DEFAULT_TEACHERS)
    parser.add_argument("--train-ratio-tags", nargs="+", type=int, default=TRAIN_RATIO_TAGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--selector-root", default="data/selector_predictions")
    parser.add_argument("--selector-model-name", default="rf_crossfit_train_pseudo_oracle")
    parser.add_argument("--auto-root", default="results_pretrained_selector_top1_auto_reweight_partial_regression")
    parser.add_argument("--plain-root", default="results_pretrained_selector_top1_regression")
    parser.add_argument("--top1-root", default="results_multiteacher_top1")
    parser.add_argument("--fixed-root", default="results_distill_regression")
    parser.add_argument("--output-root", default="results_selector_calibration_route_audit")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sample_df = collect_samples(args)
    if sample_df.empty:
        raise SystemExit("No selector samples found.")

    calibration_summary_df = calibration_summary(sample_df)
    calibration_bins_df = calibration_bins(sample_df)
    setting_df = build_setting_rows(
        sample_df=sample_df,
        auto_root=args.auto_root,
        plain_root=args.plain_root,
        top1_root=args.top1_root,
        fixed_root=args.fixed_root,
        teachers=args.teachers,
    )
    route_summary_df = summarize_routes(setting_df)

    sample_df.to_csv(output_root / "selector_sample_calibration.csv", index=False)
    calibration_summary_df.to_csv(output_root / "selector_calibration_summary.csv", index=False)
    calibration_bins_df.to_csv(output_root / "selector_calibration_bins.csv", index=False)
    setting_df.to_csv(output_root / "selector_route_audit_by_setting.csv", index=False)
    route_summary_df.to_csv(output_root / "selector_route_audit_summary.csv", index=False)
    write_report(output_root, calibration_summary_df, calibration_bins_df, route_summary_df, setting_df)

    print("## Calibration Summary")
    print(dataframe_to_markdown(calibration_summary_df))
    print("\n## Route Audit Summary")
    print(dataframe_to_markdown(route_summary_df))
    print(f"\nWrote report: {output_root / 'selector_calibration_route_audit.md'}")


if __name__ == "__main__":
    main()
