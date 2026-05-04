# pyright: reportMissingImports=false
"""Generate paper figures from Phase 8 experiment results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "paper" / "figures"


def fig_function_invariance() -> None:
    """Generate Figure: Non-linear function invariance overlay (Phase 8A)."""
    seeds = []
    for p in sorted((ROOT / "results" / "phase8a" / "nonlinear_probing").glob("results_seed*.json")):
        seeds.append(json.loads(p.read_text()))

    if not seeds:
        print("  SKIP: no Phase 8A probing results found")
        return

    funcs = ["quadratic", "sinusoidal", "polynomial", "mixed"]
    colors = {"quadratic": "#555555", "sinusoidal": "#0072B2", "polynomial": "#009E73", "mixed": "#D55E00"}
    models = ["tabpfn", "tabicl", "iltm"]
    titles = {"tabpfn": "TabPFN (Staged)", "tabicl": "TabICL (Distributed)", "iltm": "iLTM (Preproc.-dom.)"}

    # Per-subplot y-axis ranges (no sharey) so each model's line spread is visible.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    # Track per-axis data range to set tight per-subplot ylim with small margin.
    for ax, model in zip(axes, models, strict=True):
        all_y = []
        for func in funcs:
            profiles = []
            for seed_data in seeds:
                if model in seed_data and func in seed_data[model]:
                    r2_list = seed_data[model][func]["results_by_complexity"]["0"]["intermediary_r2_by_layer"]
                    profiles.append(r2_list)

            if not profiles:
                continue

            arr = np.array(profiles)
            mean = arr.mean(axis=0)
            layers = np.arange(len(mean))
            all_y.extend(mean.tolist())

            if arr.shape[0] > 1:
                std = arr.std(axis=0, ddof=1)
                ax.fill_between(layers, mean - std, mean + std, alpha=0.15, color=colors[func])
                all_y.extend((mean - std).tolist())
                all_y.extend((mean + std).tolist())

            lw = 1.5 if func == "quadratic" else 2.0
            ls = "--" if func == "quadratic" else "-"
            ax.plot(layers, mean, marker="o", linewidth=lw, linestyle=ls,
                    color=colors[func], label=func, markersize=4, alpha=0.9)

        # Tight per-subplot ylim with 5% padding so each model's line spread is maximally visible.
        if all_y:
            y_min, y_max = float(np.nanmin(all_y)), float(np.nanmax(all_y))
            pad = max(0.02, 0.08 * (y_max - y_min))
            ax.set_ylim(y_min - pad, min(1.02, y_max + pad))

        ax.set_title(titles[model], fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_xticks(layers if model != "iltm" else [0, 1, 2])
        ax.grid(alpha=0.2)
        ax.set_ylabel("Intermediary $R^2$")
        ax.legend(fontsize=8, loc="lower right" if model == "tabpfn" else "best")

    fig.tight_layout()
    out_pdf = FIGURES_DIR / "fig8_function_invariance.pdf"
    out_png = FIGURES_DIR / "fig8_function_invariance.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_pdf}")


def fig_feature_complexity() -> None:
    """Generate Figure: Peak layer vs feature count (Phase 8D)."""
    path = ROOT / "results" / "phase8d" / "feature_complexity" / "results_seed42.json"
    if not path.exists():
        print("  SKIP: no Phase 8D results found")
        return

    data = json.loads(path.read_text())
    feature_counts = [2, 4, 8, 16]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = {"tabpfn": "#0072B2", "tabicl": "#D55E00", "iltm": "#009E73"}
    markers = {"tabpfn": "o", "tabicl": "s", "iltm": "^"}

    # Left: probing peak layer
    for model in ["tabpfn", "tabicl", "iltm"]:
        if model not in data:
            continue
        peaks = [data[model][str(d)]["probing"]["peak_layer"] for d in feature_counts]
        ax1.plot(feature_counts, peaks, marker=markers[model], linewidth=2,
                 color=colors[model], label=model.upper(), markersize=8)

    ax1.set_xlabel("Number of Features ($d$)", fontsize=11)
    ax1.set_ylabel("Probing Peak Layer", fontsize=11)
    ax1.set_title("Probing: Peak Layer vs Feature Count", fontsize=9)
    ax1.set_xticks(feature_counts)
    ax1.grid(alpha=0.2)
    ax1.legend(fontsize=10)

    # Right: causal peak layer (TabPFN and TabICL only)
    for model in ["tabpfn", "tabicl"]:
        if model not in data:
            continue
        causal_peaks = []
        for d in feature_counts:
            entry = data[model][str(d)].get("causal", {})
            if entry.get("skipped"):
                causal_peaks.append(None)
            else:
                causal_peaks.append(entry.get("most_sensitive_layer"))
        valid_d = [d for d, p in zip(feature_counts, causal_peaks) if p is not None]
        valid_p = [p for p in causal_peaks if p is not None]
        ax2.plot(valid_d, valid_p, marker=markers[model], linewidth=2,
                 color=colors[model], label=model.upper(), markersize=8)

    ax2.set_xlabel("Number of Features ($d$)", fontsize=11)
    ax2.set_ylabel("Causal Peak Layer", fontsize=11)
    ax2.set_title("Causal Tracing: Peak Layer vs Feature Count", fontsize=9)
    ax2.set_xticks(feature_counts)
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    out_pdf = FIGURES_DIR / "fig9_feature_complexity.pdf"
    out_png = FIGURES_DIR / "fig9_feature_complexity.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_pdf}")


def fig_realworld_steering() -> None:
    """Generate Figure: Real-world steering scatter (Phase 8B)."""
    path = ROOT / "results" / "phase8b" / "realworld_steering" / "results_seed42.json"
    if not path.exists():
        print("  SKIP: no Phase 8B results found")
        return

    data = json.loads(path.read_text())
    datasets = [d for d in data if data[d].get("tabpfn")]

    if not datasets:
        print("  SKIP: no valid steering results")
        return

    n = len(datasets)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)

    for col, ds_name in enumerate(datasets):
        for row, model in enumerate(["tabpfn", "tabicl"]):
            ax = axes[row, col]
            result = data[ds_name].get(model)
            if not result or "mean_preds" not in result:
                ax.set_visible(False)
                continue

            lambdas = sorted([float(k) for k in result["mean_preds"]])
            preds = [result["mean_preds"][str(l)] for l in lambdas]

            color = "#0072B2" if model == "tabpfn" else "#D55E00"
            ax.plot(lambdas, preds, marker="o", linewidth=2, color=color, markersize=6)
            ax.set_title(f"{ds_name}\n{model.upper()} $|r|$={abs(result['pearson_r']):.3f}", fontsize=8)
            ax.set_xlabel("$\\lambda$")
            if col == 0:
                ax.set_ylabel("Mean Prediction")
            ax.grid(alpha=0.2)

    fig.suptitle("Real-World Steering Validation", fontsize=10, y=1.01)
    fig.tight_layout()
    out_pdf = FIGURES_DIR / "fig10_realworld_steering.pdf"
    out_png = FIGURES_DIR / "fig10_realworld_steering.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_pdf}")


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating Phase 8 figures...")
    print("\n[1/3] Function invariance (Phase 8A)")
    fig_function_invariance()

    print("\n[2/3] Feature complexity (Phase 8D)")
    fig_feature_complexity()

    print("\n[3/3] Real-world steering (Phase 8B)")
    fig_realworld_steering()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
