#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeArgument=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualization.styles import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
    FIGURE_SIZES,
    FONT_SIZES,
    PUBLICATION_DPI,
    apply_publication_style,
    save_fig,
    CKA_CMAP,
    ERROR_ALPHA,
    LINEWIDTH,
    MARKERSIZE,
)

RD5_AGG = "results/rd5_fullscale/aggregated/aggregated_results.json"
RD6_AGG = "results/rd6_fullscale/aggregated/aggregated_results.json"
TOPK_AGG = "results/rd6/sae_topk_fullscale/aggregated/aggregated_results.json"

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "paper" / "figures"


def _read_json(path: str) -> dict[str, Any]:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_size(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(n_bytes)
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.1f} {units[idx]}"


def _save_and_collect(
    fig: plt.Figure, base_path: Path, rows: list[tuple[str, str, str]]
) -> None:
    save_fig(fig, str(base_path), close=True)
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    rows.append(
        (
            base_path.name,
            _format_size(png_path.stat().st_size),
            _format_size(pdf_path.stat().st_size),
        )
    )


def _plot_fig_a1(rd5: dict[str, Any], rows: list[tuple[str, str, str]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES["full_tall"])
    target_specs = [
        ("a", "a_r2_mean", "a_r2_std", "a"),
        ("b", "b_r2_mean", "b_r2_std", "b"),
        ("c", "c_r2_mean", "c_r2_std", "c"),
        ("a*b", "ab_r2_mean", "ab_r2_std", "a*b"),
    ]

    for ax, (title, mean_key, std_key, ylab) in zip(axes.flat, target_specs):
        for model in ["tabpfn", "tabicl", "iltm"]:
            mean = np.array(rd5["copy_mechanism"][model][mean_key], dtype=float)
            std = np.array(rd5["copy_mechanism"][model][std_key], dtype=float)
            layers = np.arange(len(mean))
            ax.plot(
                layers,
                mean,
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
            )
            ax.fill_between(
                layers,
                mean - std,
                mean + std,
                color=MODEL_COLORS[model],
                alpha=ERROR_ALPHA,
            )
        ax.set_title(f"Target {title}", fontsize=FONT_SIZES["title"], fontweight="bold")
        ax.set_xlabel("Layer", fontsize=FONT_SIZES["label"])
        ax.set_ylabel(f"R2 ({ylab})", fontsize=FONT_SIZES["label"])
        ax.set_ylim(-0.05, 1.05)

    axes.flat[0].legend(fontsize=FONT_SIZES["legend"])
    fig.tight_layout()
    _save_and_collect(fig, OUT_DIR / "fig_a1_copy_mechanism", rows)


def _plot_fig_a2(rd6: dict[str, Any], rows: list[tuple[str, str, str]]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES["full"])

    tabpfn = rd6["attention_comparison"]["tabpfn"]
    if "entropy_mean" in tabpfn and "entropy_std" in tabpfn:
        tabpfn_mean = np.array(tabpfn["entropy_mean"], dtype=float)
        tabpfn_std = np.array(tabpfn["entropy_std"], dtype=float)
    else:
        tabpfn_mean = np.array(tabpfn["sample_entropy_mean"], dtype=float)
        tabpfn_std = np.array(tabpfn["sample_entropy_std"], dtype=float)

    tabicl = rd6["attention_comparison"]["tabicl"]
    if "entropy_mean" in tabicl and "entropy_std" in tabicl:
        tabicl_mean = np.array(tabicl["entropy_mean"], dtype=float)
        tabicl_std = np.array(tabicl["entropy_std"], dtype=float)
    else:
        tabicl_mean = np.array(tabicl["sample_entropy_mean"], dtype=float)
        tabicl_std = np.array(tabicl["sample_entropy_std"], dtype=float)

    x1 = np.arange(len(tabpfn_mean))
    ax1.bar(
        x1,
        tabpfn_mean,
        yerr=tabpfn_std,
        color=MODEL_COLORS["tabpfn"],
        alpha=0.85,
        capsize=3,
    )
    ax1.set_title(
        MODEL_LABELS["tabpfn"], fontsize=FONT_SIZES["title"], fontweight="bold"
    )
    ax1.set_xlabel("Layer", fontsize=FONT_SIZES["label"])
    ax1.set_ylabel("Entropy", fontsize=FONT_SIZES["label"])
    ax1.set_xticks(x1)

    x2 = np.arange(len(tabicl_mean))
    ax2.bar(
        x2,
        tabicl_mean,
        yerr=tabicl_std,
        color=MODEL_COLORS["tabicl"],
        alpha=0.85,
        capsize=3,
    )
    ax2.set_title(
        MODEL_LABELS["tabicl"], fontsize=FONT_SIZES["title"], fontweight="bold"
    )
    ax2.set_xlabel("Layer", fontsize=FONT_SIZES["label"])
    ax2.set_ylabel("Entropy", fontsize=FONT_SIZES["label"])
    ax2.set_xticks(x2)

    fig.tight_layout()
    _save_and_collect(fig, OUT_DIR / "fig_a2_attention_entropy", rows)


def _plot_fig_a3(rd6: dict[str, Any], rows: list[tuple[str, str, str]]) -> None:
    cls_data = rd6["classification_probing"]
    preferred = ["breast_cancer", "iris_binary", "adult_income"]
    datasets = [d for d in preferred if d in cls_data]
    if not datasets:
        datasets = [d for d in cls_data if d != "n_seeds"]

    models = ["tabpfn", "tabicl", "iltm"]
    x = np.arange(len(datasets))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZES["half"])
    for i, model in enumerate(models):
        vals: list[float] = []
        errs: list[float] = []
        for dataset in datasets:
            entry = cls_data[dataset][model]
            if "best_accuracy_mean" in entry:
                vals.append(float(entry["best_accuracy_mean"]))
                if "accuracy_std" in entry and "best_layer" in entry:
                    best_layer = int(entry["best_layer"])
                    errs.append(float(entry["accuracy_std"][best_layer]))
                else:
                    errs.append(0.0)
            else:
                acc_mean = np.array(entry["accuracy_mean"], dtype=float)
                acc_std = np.array(
                    entry.get("accuracy_std", np.zeros_like(acc_mean)), dtype=float
                )
                best_layer = int(np.argmax(acc_mean))
                vals.append(float(acc_mean[best_layer]))
                errs.append(float(acc_std[best_layer]))

        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width * 0.9,
            yerr=errs,
            capsize=3,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in datasets])
    ax.set_ylabel("Accuracy", fontsize=FONT_SIZES["label"])
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=FONT_SIZES["legend"])
    fig.tight_layout()
    _save_and_collect(fig, OUT_DIR / "fig_a3_classification", rows)


def _plot_fig_a4(rd6: dict[str, Any], rows: list[tuple[str, str, str]]) -> None:
    rw_data = rd6["realworld_expanded"]
    datasets = [d for d in rw_data if d != "n_seeds"]
    models = ["tabpfn", "tabicl", "iltm"]

    x = np.arange(len(datasets))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZES["full_tall"])
    for i, model in enumerate(models):
        vals: list[float] = []
        errs: list[float] = []
        for dataset in datasets:
            entry = rw_data[dataset][model]
            vals.append(float(entry["best_r2"]))
            if "r2_std" in entry and "best_layer" in entry:
                best_layer = int(entry["best_layer"])
                errs.append(float(entry["r2_std"][best_layer]))
            else:
                errs.append(0.0)

        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width * 0.9,
            yerr=errs,
            capsize=3,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel("Best R2", fontsize=FONT_SIZES["label"])
    ax.legend(fontsize=FONT_SIZES["legend"], ncol=3)
    fig.tight_layout()
    _save_and_collect(fig, OUT_DIR / "fig_a4_realworld", rows)


def _plot_fig_a5(rd5: dict[str, Any], rows: list[tuple[str, str, str]]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES["full"])

    for model in ["tabpfn", "tabicl", "iltm"]:
        alpha_mean = np.array(
            rd5["coefficient_probing"][model]["alpha_r2_mean"], dtype=float
        )
        alpha_std = np.array(
            rd5["coefficient_probing"][model]["alpha_r2_std"], dtype=float
        )
        alpha_layers = np.arange(len(alpha_mean))
        ax1.plot(
            alpha_layers,
            alpha_mean,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )
        ax1.fill_between(
            alpha_layers,
            alpha_mean - alpha_std,
            alpha_mean + alpha_std,
            color=MODEL_COLORS[model],
            alpha=ERROR_ALPHA,
        )

        beta_mean = np.array(
            rd5["coefficient_probing"][model]["beta_r2_mean"], dtype=float
        )
        beta_std = np.array(
            rd5["coefficient_probing"][model]["beta_r2_std"], dtype=float
        )
        beta_layers = np.arange(len(beta_mean))
        ax2.plot(
            beta_layers,
            beta_mean,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )
        ax2.fill_between(
            beta_layers,
            beta_mean - beta_std,
            beta_mean + beta_std,
            color=MODEL_COLORS[model],
            alpha=ERROR_ALPHA,
        )

    ax1.set_title("Alpha R2", fontsize=FONT_SIZES["title"], fontweight="bold")
    ax1.set_xlabel("Layer", fontsize=FONT_SIZES["label"])
    ax1.set_ylabel("R2", fontsize=FONT_SIZES["label"])
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=FONT_SIZES["legend"])

    ax2.set_title("Beta R2", fontsize=FONT_SIZES["title"], fontweight="bold")
    ax2.set_xlabel("Layer", fontsize=FONT_SIZES["label"])
    ax2.set_ylabel("R2", fontsize=FONT_SIZES["label"])
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=FONT_SIZES["legend"])

    fig.tight_layout()
    _save_and_collect(fig, OUT_DIR / "fig_a5_coefficient_probing", rows)


def _print_summary(rows: list[tuple[str, str, str]]) -> None:
    print("\nGenerated Figure Files")
    print("-" * 64)
    print(f"{'Figure':<28} {'PNG':>14} {'PDF':>14}")
    print("-" * 64)
    for name, png_size, pdf_size in rows:
        print(f"{name:<28} {png_size:>14} {pdf_size:>14}")
    print("-" * 64)


def main() -> int:
    apply_publication_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _ = (PUBLICATION_DPI, CKA_CMAP)

    rd5 = _read_json(RD5_AGG)
    rd6 = _read_json(RD6_AGG)
    topk_data = _read_json(TOPK_AGG)

    _ = topk_data

    rows: list[tuple[str, str, str]] = []
    _plot_fig_a1(rd5, rows)
    _plot_fig_a2(rd6, rows)
    _plot_fig_a3(rd6, rows)
    _plot_fig_a4(rd6, rows)
    _plot_fig_a5(rd5, rows)

    _print_summary(rows)
    plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
