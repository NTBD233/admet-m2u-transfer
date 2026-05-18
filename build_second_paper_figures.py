import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_figures_second"
TABLES = ROOT / "paper_tables_second"


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_method_diagram():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_axis_off()

    boxes = {
        "ecfp": (0.05, 0.58, 0.18, 0.18, "ECFP4\nfingerprint"),
        "student": (0.33, 0.58, 0.22, 0.18, "ECFP-only\nAdapterFusion student"),
        "pred": (0.72, 0.58, 0.2, 0.18, "ADMET\nprediction"),
        "teachers": (0.05, 0.15, 0.24, 0.2, "RF teachers\nECFP4 / Desc /\nECFP4+Desc"),
        "signals": (0.37, 0.15, 0.22, 0.2, "Reliability signals\nuncertainty, gap,\nprior, OOD"),
        "selector": (0.72, 0.15, 0.2, 0.2, "Pretrained selector\nfrozen top-1 routing"),
    }
    colors = {
        "ecfp": "#e8f1fb",
        "student": "#f0f5ea",
        "pred": "#f7eadf",
        "teachers": "#f4edf7",
        "signals": "#edf3f2",
        "selector": "#fff4d8",
    }
    for key, (x, y, w, h, text) in boxes.items():
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.0,
            facecolor=colors[key],
            edgecolor="#34404a",
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(a, b, text=None, dy=0):
        x1, y1 = a
        x2, y2 = b
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1.0, color="#34404a")
        ax.add_patch(arr)
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center", va="center", fontsize=8, color="#34404a")

    arrow((0.23, 0.67), (0.33, 0.67), "inference input", 0.05)
    arrow((0.55, 0.67), (0.72, 0.67), "ECFP-only", 0.05)
    arrow((0.29, 0.25), (0.37, 0.25))
    arrow((0.59, 0.25), (0.72, 0.25))
    arrow((0.82, 0.35), (0.45, 0.58))
    ax.text(
        0.64,
        0.50,
        "training-time\nteacher choice",
        ha="center",
        va="center",
        fontsize=8,
        color="#34404a",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    arrow((0.17, 0.58), (0.17, 0.35), "train only:\ndescriptors", 0.0)

    ax.text(0.5, 0.92, "Selector pretraining + frozen routing distillation", ha="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.03, "At deployment, descriptors and teachers are removed; the student consumes ECFP4 only.", ha="center", fontsize=8)
    savefig(fig, "fig1_selector_method")


def figure_main_rmse():
    df = pd.read_csv(TABLES / "table1_main_aggregate_rmse.csv")
    order = [
        "Base AdapterFusion",
        "Fixed ECFP4+Desc RF",
        "Uniform multi-teacher",
        "Validation-weighted",
        "Pretrained selector",
        "Selector + high-resource decay",
        "Top-1 validation",
    ]
    df = df.set_index("method").loc[order].reset_index()
    y = np.arange(len(df))
    vals = df["mean_test_rmse"].astype(float).to_numpy()
    colors = ["#9aa6b2", "#9aa6b2", "#b5c7df", "#93b5d8", "#5b8cc0", "#2f6f9f", "#d8a05e"]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.barh(y, vals, color=colors, edgecolor="#334", linewidth=0.5)
    ax.set_yticks(y, df["method"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean test RMSE across 45 regression settings")
    ax.set_xlim(5.06, 5.20)
    ax.axvline(5.1841, color="#7f8c8d", lw=1, ls="--", label="Base")
    ax.axvline(5.0792, color="#d28b35", lw=1, ls=":", label="Top-1 validation")
    for yi, val in zip(y, vals):
        ax.text(val + 0.002, yi, f"{val:.4f}", va="center", fontsize=8)
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, "fig2_main_rmse")


def figure_setting_heatmap():
    df = pd.read_csv(TABLES / "table2_main_by_setting_rmse.csv")
    datasets = ["caco2_wang", "lipophilicity_astrazeneca", "solubility_aqsoldb", "vdss_lombardo", "ppbr_az"]
    ratios = [10, 20, 50]
    mat = np.zeros((len(datasets), len(ratios)))
    labels = []
    for i, dataset in enumerate(datasets):
        row_labels = []
        for j, ratio in enumerate(ratios):
            sub = df[(df["dataset"] == dataset) & (df["train_ratio"] == ratio)].iloc[0]
            mat[i, j] = float(sub["delta_selector_decay_vs_top1"])
            short = {
                "Pretrained selector": "Selector",
                "Selector + high-resource decay": "Sel.+decay",
                "Validation-weighted": "Val-wt",
                "Uniform multi-teacher": "Uniform",
                "Fixed ECFP4+Desc RF": "Fixed",
                "Top-1 validation": "Top1",
                "Base AdapterFusion": "Base",
            }
            row_labels.append(short.get(sub["best_method"], sub["best_method"]))
        labels.append(row_labels)
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    vmax = max(abs(mat.min()), abs(mat.max()))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(ratios)), [str(r) for r in ratios])
    ax.set_yticks(np.arange(len(datasets)), datasets)
    ax.set_xlabel("Train ratio (%)")
    ax.set_title("Selector + decay delta vs top-1 validation")
    for i in range(len(datasets)):
        for j in range(len(ratios)):
            ax.text(j, i, f"{mat[i,j]:+.3f}\n{labels[i][j]}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RMSE delta")
    savefig(fig, "fig3_setting_delta_heatmap")


def figure_selector_quality():
    reg = pd.read_csv(TABLES / "table4_selector_quality_by_dataset.csv")
    class_note = ROOT / "paper_notes" / "second_paper_classification_selector_supplement.md"
    class_vals = None
    if class_note.exists():
        # Values are fixed by the generated note; keeping them explicit avoids parsing markdown.
        class_vals = {"test_accuracy": 0.2162, "test_majority_accuracy": 0.2958}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [2.4, 1.0]})
    ax = axes[0]
    x = np.arange(len(reg))
    ax.bar(x - 0.18, reg["test_accuracy"].astype(float), width=0.36, label="Accuracy", color="#5b8cc0")
    ax.bar(x + 0.18, reg["test_macro_f1"].astype(float), width=0.36, label="Macro-F1", color="#9abf88")
    ax.set_xticks(x, reg["dataset"], rotation=30, ha="right")
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Regression selector quality")
    ax.legend(frameon=False)

    ax = axes[1]
    if class_vals:
        ax.bar([0, 1], [class_vals["test_accuracy"], class_vals["test_majority_accuracy"]], color=["#c65f5f", "#9aa6b2"])
        ax.set_xticks([0, 1], ["Selector", "Majority"], rotation=20, ha="right")
        ax.set_ylim(0, 0.65)
        ax.set_title("Classification diagnostic")
        ax.set_ylabel("Test oracle accuracy")
        for i, val in enumerate([class_vals["test_accuracy"], class_vals["test_majority_accuracy"]]):
            ax.text(i, val + 0.015, f"{val:.3f}", ha="center", fontsize=8)
    savefig(fig, "fig4_selector_quality")


def main():
    setup()
    figure_method_diagram()
    figure_main_rmse()
    figure_setting_heatmap()
    figure_selector_quality()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
