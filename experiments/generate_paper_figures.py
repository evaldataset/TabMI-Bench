#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeArgument=false
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
    CKA_CMAP,
    ERROR_ALPHA,
    FIGURE_SIZES,
    FONT_SIZES,
    LINEWIDTH,
    MARKERSIZE,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
    PUBLICATION_DPI,
    SAE_COLORS,
    SAE_HATCHES,
    SAE_LABELS,
    apply_publication_style,
    save_fig,
)
from src.visualization.plots import (
    plot_cka_heatmaps,
    plot_multi_model_r2,
    plot_sae_grouped_bar,
    plot_sensitivity_profiles,
    plot_steering_scatter,
)

RD5_AGG = "results/rd5_fullscale/aggregated/aggregated_results.json"
RD6_AGG = "results/rd6_fullscale/aggregated/aggregated_results.json"
V25_AGG = "results/rd6/tabpfn25_fullscale/aggregated/aggregated_results.json"
TOPK_AGG = "results/rd6/sae_topk_fullscale/aggregated/aggregated_results.json"
L5L6_DATA = "results/rd6/tabicl_l5l6_zoom/results.json"

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


def _plot_fig1(rd5: dict[str, Any], rows: list[tuple[str, str, str]]) -> None:
    model_data = {
        "tabpfn": {
            "mean": rd5["intermediary_probing"]["tabpfn"]["intermediary_r2_mean"],
            "std": rd5["intermediary_probing"]["tabpfn"]["intermediary_r2_std"],
        },
        "tabicl": {
            "mean": rd5["intermediary_probing"]["tabicl"]["intermediary_r2_mean"],
            "std": rd5["intermediary_probing"]["tabicl"]["intermediary_r2_std"],
        },
        "iltm": {
            "mean": rd5["intermediary_probing"]["iltm"]["intermediary_r2_mean"],
            "std": rd5["intermediary_probing"]["iltm"]["intermediary_r2_std"],
        },
    }
    fig = plot_multi_model_r2(
        model_data=model_data,
        title="Intermediary R² Across Layers (5-seed)",
        figsize=FIGURE_SIZES["full"],
    )
    _save_and_collect(fig, OUT_DIR / "fig1_intermediary_r2", rows)


def _plot_fig2(rd5: dict[str, Any], rows: list[tuple[str, str, str]]) -> None:
    cka_matrices = {
        "tabpfn": np.array(rd5["cka"]["tabpfn"]["cka_matrix_mean"]),
        "tabicl": np.array(rd5["cka"]["tabicl"]["cka_matrix_mean"]),
        "iltm": np.array(rd5["cka"]["iltm"]["cka_matrix_mean"]),
    }
    plt.set_cmap(CKA_CMAP)
    fig = plot_cka_heatmaps(
        cka_matrices=cka_matrices,
        title="CKA Similarity Matrices",
        figsize=FIGURE_SIZES["full"],
    )
    _save_and_collect(fig, OUT_DIR / "fig2_cka_heatmaps", rows)


def _plot_fig3(
    rd5: dict[str, Any], v25_data: dict[str, Any], rows: list[tuple[str, str, str]]
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES["full"])

    layers = np.arange(len(rd5["patching"]["tabpfn"]["sensitivity_mean"]))
    for model in ["tabpfn", "tabicl"]:
        mean = np.array(rd5["patching"][model]["sensitivity_mean"])
        std = np.array(rd5["patching"][model]["sensitivity_std"])
        ax1.plot(
            layers,
            mean,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )
        ax1.fill_between(
            layers, mean - std, mean + std, color=MODEL_COLORS[model], alpha=ERROR_ALPHA
        )
    ax1.set_title(
        "Noising-Based Causal Tracing", fontsize=FONT_SIZES["title"], fontweight="bold"
    )
    ax1.set_xlabel("Layer", fontsize=FONT_SIZES["label"])
    ax1.set_ylabel("Sensitivity", fontsize=FONT_SIZES["label"])
    ax1.legend(fontsize=FONT_SIZES["legend"])

    bar_labels = ["Logit Lens\n(first R²>0.5)", "Logit Lens\n(first R²>0.5)", "CKA Mean\n(adjacent)", "CKA Mean\n(adjacent)"]
    bar_values = [
        float(v25_data["logit_lens"]["v2"]["first_above_05"]["mean"]),
        float(v25_data["logit_lens"]["v2.5"]["first_above_05"]["mean"]),
        float(v25_data["cka_mean"]["v2"]["mean"]),
        float(v25_data["cka_mean"]["v2.5"]["mean"]),
    ]
    bar_stds = [
        float(v25_data["logit_lens"]["v2"]["first_above_05"]["std"]),
        float(v25_data["logit_lens"]["v2.5"]["first_above_05"]["std"]),
        float(v25_data["cka_mean"]["v2"]["std"]),
        float(v25_data["cka_mean"]["v2.5"]["std"]),
    ]
    bar_colors = [
        MODEL_COLORS["tabpfn"],
        MODEL_COLORS["tabpfn25"],
        MODEL_COLORS["tabpfn"],
        MODEL_COLORS["tabpfn25"],
    ]
    x = np.arange(len(bar_labels))
    ax2.bar(x, bar_values, yerr=bar_stds, color=bar_colors, capsize=3, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["v2", "v2.5", "v2", "v2.5"], fontsize=FONT_SIZES["tick"])
    # Add group labels underneath
    ax2.text(0.5, -0.18, "Logit Lens", ha="center", va="top", fontsize=FONT_SIZES["annotation"], transform=ax2.get_xaxis_transform())
    ax2.text(2.5, -0.18, "CKA Mean", ha="center", va="top", fontsize=FONT_SIZES["annotation"], transform=ax2.get_xaxis_transform())
    ax2.set_title("TabPFN v2 vs v2.5", fontsize=FONT_SIZES["title"], fontweight="bold")
    ax2.set_ylabel("Value", fontsize=FONT_SIZES["label"])

    fig.tight_layout()
    _save_and_collect(fig, OUT_DIR / "fig3_causal_and_v25", rows)


def _plot_fig4(
    rd6: dict[str, Any], l5l6_data: dict[str, Any], rows: list[tuple[str, str, str]]
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES["full"])

    for model in ["tabpfn", "tabicl"]:
        layers = rd6["robust_steering"][model]["layers"]
        mean = rd6["robust_steering"][model]["abs_r_mean"]
        std = rd6["robust_steering"][model]["abs_r_std"]
        ax1.errorbar(
            layers,
            mean,
            yerr=std,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            capsize=3,
        )
    ax1.set_title(
        "Robust Steering Effectiveness", fontsize=FONT_SIZES["title"], fontweight="bold"
    )
    ax1.set_xlabel("Layer", fontsize=FONT_SIZES["label"])
    ax1.set_ylabel("|r| (Pearson)", fontsize=FONT_SIZES["label"])
    ax1.set_xticks(rd6["robust_steering"]["tabpfn"]["layers"])
    ax1.legend(fontsize=FONT_SIZES["legend"])
    ax1.set_ylim(-0.05, 1.15)

    zoom_layers = [4, 5, 6, 7]
    per_seed = l5l6_data["per_seed"]
    means = []
    stds = []
    for layer in zoom_layers:
        vals = [float(seed["per_layer"][str(layer)]["abs_r_mean"]) for seed in per_seed]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    ax2.bar(
        np.arange(len(zoom_layers)),
        means,
        yerr=stds,
        color=MODEL_COLORS["tabicl"],
        capsize=3,
        alpha=0.85,
    )
    ax2.set_xticks(np.arange(len(zoom_layers)))
    ax2.set_xticklabels([str(layer) for layer in zoom_layers])
    ax2.set_title(
        "TabICL L5-L6 Anomaly", fontsize=FONT_SIZES["title"], fontweight="bold"
    )
    ax2.set_xlabel("Layer", fontsize=FONT_SIZES["label"])
    ax2.set_ylabel("|r|", fontsize=FONT_SIZES["label"])
    ax2.set_ylim(-0.05, 1.1)

    fig.tight_layout()
    _save_and_collect(fig, OUT_DIR / "fig4_steering_and_anomaly", rows)


def _plot_fig5(topk_data: dict[str, Any], rows: list[tuple[str, str, str]]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES["full"])

    variants = ["relu_16x", "jumprelu_16x", "topk_16x_6p25"]
    models = ["tabpfn", "tabicl", "iltm"]
    x = np.arange(len(models))
    width = 0.8 / len(variants)

    for i, variant in enumerate(variants):
        offset = (i - (len(variants) - 1) / 2) * width
        alpha_means = [
            topk_data["aggregated"][variant][model]["max_alpha_corr"]["mean"]
            for model in models
        ]
        alpha_stds = [
            topk_data["aggregated"][variant][model]["max_alpha_corr"]["std"]
            for model in models
        ]
        sparsity_means = [
            topk_data["aggregated"][variant][model]["sparsity"]["mean"]
            for model in models
        ]
        sparsity_stds = [
            topk_data["aggregated"][variant][model]["sparsity"]["std"]
            for model in models
        ]

        ax1.bar(
            x + offset,
            alpha_means,
            width * 0.9,
            yerr=alpha_stds,
            capsize=3,
            color=SAE_COLORS[variant],
            hatch=SAE_HATCHES[variant],
            label=SAE_LABELS[variant],
            alpha=0.85,
        )
        ax2.bar(
            x + offset,
            sparsity_means,
            width * 0.9,
            yerr=sparsity_stds,
            capsize=3,
            color=SAE_COLORS[variant],
            hatch=SAE_HATCHES[variant],
            label=SAE_LABELS[variant],
            alpha=0.85,
        )

    ax1.set_title("Max |r_α|", fontsize=FONT_SIZES["title"], fontweight="bold")
    ax1.set_ylabel("Max |r_α|", fontsize=FONT_SIZES["label"])
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax1.legend(fontsize=FONT_SIZES["legend"])

    ax2.set_title("Sparsity", fontsize=FONT_SIZES["title"], fontweight="bold")
    ax2.set_ylabel("Sparsity", fontsize=FONT_SIZES["label"])
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax2.legend(fontsize=FONT_SIZES["legend"])

    fig.tight_layout()
    _save_and_collect(fig, OUT_DIR / "fig5_sae_comparison", rows)


def _plot_fig6(v25_data: dict[str, Any], rows: list[tuple[str, str, str]]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES["full"])

    labels = ["v2", "v2.5"]
    colors = [MODEL_COLORS["tabpfn"], MODEL_COLORS["tabpfn25"]]

    alpha_vals = [
        float(v25_data["coefficient_alpha"]["v2"]["max_r2"]["mean"]),
        float(v25_data["coefficient_alpha"]["v2.5"]["max_r2"]["mean"]),
    ]
    alpha_err = [
        float(v25_data["coefficient_alpha"]["v2"]["max_r2"]["std"]),
        float(v25_data["coefficient_alpha"]["v2.5"]["max_r2"]["std"]),
    ]
    x = np.arange(2)
    ax1.bar(x, alpha_vals, yerr=alpha_err, color=colors, capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title(
        "Coefficient α max R²", fontsize=FONT_SIZES["title"], fontweight="bold"
    )
    ax1.set_ylabel("R²", fontsize=FONT_SIZES["label"])

    cka_vals = [
        float(v25_data["cka_mean"]["v2"]["mean"]),
        float(v25_data["cka_mean"]["v2.5"]["mean"]),
    ]
    cka_err = [
        float(v25_data["cka_mean"]["v2"]["std"]),
        float(v25_data["cka_mean"]["v2.5"]["std"]),
    ]
    ax2.bar(x, cka_vals, yerr=cka_err, color=colors, capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title("CKA mean adjacency", fontsize=FONT_SIZES["title"], fontweight="bold")
    ax2.set_ylabel("CKA", fontsize=FONT_SIZES["label"])

    fig.tight_layout()
    _save_and_collect(fig, OUT_DIR / "fig6_v2_vs_v25", rows)


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

    _ = (
        plot_sensitivity_profiles,
        plot_steering_scatter,
        plot_sae_grouped_bar,
        PUBLICATION_DPI,
    )

    rd5 = _read_json(RD5_AGG)
    rd6 = _read_json(RD6_AGG)
    v25_data = _read_json(V25_AGG)
    topk_data = _read_json(TOPK_AGG)
    l5l6_data = _read_json(L5L6_DATA)

    rows: list[tuple[str, str, str]] = []
    _plot_fig1(rd5, rows)
    _plot_fig2(rd5, rows)
    _plot_fig3(rd5, v25_data, rows)
    _plot_fig4(rd6, l5l6_data, rows)
    _plot_fig5(topk_data, rows)
    _plot_fig6(v25_data, rows)

    _print_summary(rows)
    plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
