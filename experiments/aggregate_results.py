#!/usr/bin/env python3
"""Aggregate multi-seed RD-5 results into mean±std with error-bar plots.

Reads per-seed results from results/rd5_fullscale/seed_*/
and produces aggregated statistics + publication-quality plots.

Usage:
    .venv/bin/python experiments/aggregate_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FULLSCALE_DIR = ROOT / "results" / "rd5_fullscale"
OUTPUT_DIR = FULLSCALE_DIR / "aggregated"

MODEL_ORDER = ["tabpfn", "tabicl", "iltm"]
MODEL_LABELS = {"tabpfn": "TabPFN", "tabicl": "TabICL", "iltm": "iLTM"}
MODEL_COLORS = {"tabpfn": "#1f77b4", "tabicl": "#ff7f0e", "iltm": "#2ca02c"}


def find_seed_dirs() -> list[Path]:
    """Find all seed_* directories."""
    if not FULLSCALE_DIR.exists():
        print(f"Error: {FULLSCALE_DIR} not found. Run run_fullscale.py first.")
        sys.exit(1)
    dirs = sorted(FULLSCALE_DIR.glob("seed_*"))
    if not dirs:
        print(f"Error: No seed_* directories found in {FULLSCALE_DIR}")
        sys.exit(1)
    return dirs


def load_seed_results(seed_dirs: list[Path], experiment: str) -> list[dict[str, Any]]:
    """Load results.json from each seed directory for a given experiment."""
    results = []
    for sd in seed_dirs:
        json_path = sd / experiment / "results.json"
        if json_path.exists():
            with json_path.open() as f:
                results.append(json.load(f))
    return results


def aggregate_layer_metric(
    seed_results: list[dict[str, Any]],
    model: str,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a per-layer metric across seeds and compute mean±std.

    Returns (mean_array, std_array) both of shape [n_layers].
    """
    arrays = []
    for result in seed_results:
        if model in result and key in result[model]:
            arrays.append(np.asarray(result[model][key], dtype=np.float64))
    if not arrays:
        return np.array([]), np.array([])
    stacked = np.stack(arrays, axis=0)  # [n_seeds, n_layers]
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)


def aggregate_intermediary_probing(seed_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate intermediary probing results."""
    results = load_seed_results(seed_dirs, "intermediary_probing")
    if not results:
        return {}

    agg: dict[str, Any] = {"n_seeds": len(results)}
    for model in MODEL_ORDER:
        mean, std = aggregate_layer_metric(results, model, "intermediary_r2_by_layer")
        if mean.size > 0:
            agg[model] = {
                "intermediary_r2_mean": mean.tolist(),
                "intermediary_r2_std": std.tolist(),
                "peak_layer": int(np.argmax(mean)),
                "peak_r2": float(np.max(mean)),
            }
    return agg


def aggregate_copy_mechanism(seed_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate copy mechanism results."""
    results = load_seed_results(seed_dirs, "copy_mechanism")
    if not results:
        return {}

    agg: dict[str, Any] = {"n_seeds": len(results)}
    for model in MODEL_ORDER:
        model_agg: dict[str, Any] = {}
        for target in ["a", "b", "c", "ab"]:
            key = f"{target}_r2_by_layer"
            mean, std = aggregate_layer_metric(results, model, key)
            if mean.size > 0:
                model_agg[f"{target}_r2_mean"] = mean.tolist()
                model_agg[f"{target}_r2_std"] = std.tolist()
        if model_agg:
            agg[model] = model_agg
    return agg


def aggregate_coefficient_probing(seed_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate coefficient probing results."""
    results = load_seed_results(seed_dirs, "coefficient_probing")
    if not results:
        return {}

    agg: dict[str, Any] = {"n_seeds": len(results)}
    for model in MODEL_ORDER:
        model_agg: dict[str, Any] = {}
        for coef in ["alpha", "beta"]:
            key = f"{coef}_r2_by_layer"
            mean, std = aggregate_layer_metric(results, model, key)
            if mean.size > 0:
                model_agg[f"{coef}_r2_mean"] = mean.tolist()
                model_agg[f"{coef}_r2_std"] = std.tolist()
                model_agg[f"peak_{coef}_layer"] = int(np.argmax(mean))
                model_agg[f"peak_{coef}_r2"] = float(np.max(mean))
        if model_agg:
            agg[model] = model_agg
    return agg


def aggregate_cka(seed_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate CKA results."""
    results = load_seed_results(seed_dirs, "cka")
    if not results:
        return {}

    agg: dict[str, Any] = {"n_seeds": len(results)}
    for model in MODEL_ORDER:
        if model in results[0] and "cka_matrix" in results[0][model]:
            matrices = []
            for r in results:
                if model in r and "cka_matrix" in r[model]:
                    matrices.append(
                        np.asarray(r[model]["cka_matrix"], dtype=np.float64)
                    )
            if matrices:
                stacked = np.stack(matrices, axis=0)
                agg[model] = {
                    "cka_matrix_mean": np.mean(stacked, axis=0).tolist(),
                    "cka_matrix_std": np.std(stacked, axis=0).tolist(),
                }
    return agg


def aggregate_steering(seed_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate steering results (scalar metrics)."""
    results = load_seed_results(seed_dirs, "steering")
    if not results:
        return {}

    agg: dict[str, Any] = {"n_seeds": len(results)}

    # TabPFN
    tabpfn_rs = [
        float(r["tabpfn"]["effect"]["pearson_r"])
        for r in results
        if "tabpfn" in r and "effect" in r["tabpfn"]
    ]
    if tabpfn_rs:
        agg["tabpfn"] = {
            "pearson_r_mean": float(np.mean(tabpfn_rs)),
            "pearson_r_std": float(np.std(tabpfn_rs)),
        }

    # TabICL
    tabicl_rs = [
        float(r["tabicl"]["best_effect"]["pearson_r"])
        for r in results
        if "tabicl" in r and "best_effect" in r["tabicl"]
    ]
    if tabicl_rs:
        agg["tabicl"] = {
            "pearson_r_mean": float(np.mean(tabicl_rs)),
            "pearson_r_std": float(np.std(tabicl_rs)),
        }

    return agg


def aggregate_sae(seed_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate SAE results."""
    results = load_seed_results(seed_dirs, "sae")
    if not results:
        return {}

    agg: dict[str, Any] = {"n_seeds": len(results)}
    for model in MODEL_ORDER:
        alpha_corrs = [
            float(r[model]["max_alpha_corr"])
            for r in results
            if model in r and "max_alpha_corr" in r[model]
        ]
        beta_corrs = [
            float(r[model]["max_beta_corr"])
            for r in results
            if model in r and "max_beta_corr" in r[model]
        ]
        recon_losses = [
            float(r[model]["recon_loss"])
            for r in results
            if model in r and "recon_loss" in r[model]
        ]
        if alpha_corrs:
            agg[model] = {
                "max_alpha_corr_mean": float(np.mean(alpha_corrs)),
                "max_alpha_corr_std": float(np.std(alpha_corrs)),
                "max_beta_corr_mean": float(np.mean(beta_corrs)),
                "max_beta_corr_std": float(np.std(beta_corrs)),
                "recon_loss_mean": float(np.mean(recon_losses)),
                "recon_loss_std": float(np.std(recon_losses)),
            }
    return agg


def aggregate_patching(seed_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate noising-based causal tracing results."""
    results = load_seed_results(seed_dirs, "patching")
    if not results:
        return {}

    agg: dict[str, Any] = {"n_seeds": len(results)}
    for model in ["tabpfn", "tabicl"]:
        sensitivities = []
        for r in results:
            if model in r and "summary_sensitivity" in r[model]:
                sensitivities.append(
                    np.asarray(r[model]["summary_sensitivity"], dtype=np.float64)
                )
        if sensitivities:
            stacked = np.stack(sensitivities, axis=0)
            mean = np.mean(stacked, axis=0)
            std = np.std(stacked, axis=0)
            agg[model] = {
                "sensitivity_mean": mean.tolist(),
                "sensitivity_std": std.tolist(),
                "most_sensitive_layer": int(np.argmax(mean)),
                "peak_sensitivity": float(np.max(mean)),
            }
    # Store the finding text from the first result
    if results and "finding" in results[0]:
        agg["finding"] = results[0]["finding"]
    return agg


def aggregate_realworld_probing(seed_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate real-world probing results across seeds."""
    results = load_seed_results(seed_dirs, "realworld_probing")
    if not results:
        return {}

    agg: dict[str, Any] = {"n_seeds": len(results)}
    # Collect per-dataset, per-model, per-layer R² across seeds
    datasets = set()
    for r in results:
        if "datasets" in r:
            datasets.update(r["datasets"].keys())

    for ds_name in sorted(datasets):
        ds_agg: dict[str, Any] = {}
        for model in MODEL_ORDER:
            r2_arrays = []
            for r in results:
                if (
                    "datasets" in r
                    and ds_name in r["datasets"]
                    and model in r["datasets"][ds_name]
                    and "r2_by_layer" in r["datasets"][ds_name][model]
                ):
                    r2_arrays.append(
                        np.asarray(
                            r["datasets"][ds_name][model]["r2_by_layer"],
                            dtype=np.float64,
                        )
                    )
            if r2_arrays:
                stacked = np.stack(r2_arrays, axis=0)
                mean = np.mean(stacked, axis=0)
                std = np.std(stacked, axis=0)
                ds_agg[model] = {
                    "r2_mean": mean.tolist(),
                    "r2_std": std.tolist(),
                    "best_layer": int(np.argmax(mean)),
                    "best_r2": float(np.max(mean)),
                }
        if ds_agg:
            agg[ds_name] = ds_agg
    return agg


def aggregate_realworld_cka(seed_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate real-world CKA results across seeds."""
    results = load_seed_results(seed_dirs, "realworld_cka")
    if not results:
        return {}

    agg: dict[str, Any] = {"n_seeds": len(results)}
    # Use averaged_by_model from each seed
    for model in MODEL_ORDER:
        matrices = []
        for r in results:
            if (
                "averaged_by_model" in r
                and model in r["averaged_by_model"]
                and "avg_cka_matrix" in r["averaged_by_model"][model]
            ):
                matrices.append(
                    np.asarray(
                        r["averaged_by_model"][model]["avg_cka_matrix"],
                        dtype=np.float64,
                    )
                )
        if matrices:
            stacked = np.stack(matrices, axis=0)
            agg[model] = {
                "avg_cka_matrix_mean": np.mean(stacked, axis=0).tolist(),
                "avg_cka_matrix_std": np.std(stacked, axis=0).tolist(),
            }
    return agg

# ─── Plotting ────────────────────────────────────────────────────────


def plot_intermediary_with_errorbars(agg: dict[str, Any], save_path: Path) -> None:
    """Publication-quality intermediary probing with error bars."""
    if not agg:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODEL_ORDER:
        if model not in agg:
            continue
        mean = np.asarray(agg[model]["intermediary_r2_mean"])
        std = np.asarray(agg[model]["intermediary_r2_std"])
        layers = np.arange(len(mean))
        ax.plot(
            layers,
            mean,
            marker="o",
            linewidth=2.0,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
        ax.fill_between(
            layers, mean - std, mean + std, alpha=0.2, color=MODEL_COLORS[model]
        )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("R² (intermediary a·b)", fontsize=12)
    ax.set_title(
        "Intermediary Probing: R² by Layer (mean ± std across seeds)", fontsize=13
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_copy_mechanism_with_errorbars(agg: dict[str, Any], save_path: Path) -> None:
    """Publication-quality copy mechanism with error bars."""
    if not agg:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    target_styles = {
        "a": {"color": "#1f77b4", "marker": "o"},
        "b": {"color": "#ff7f0e", "marker": "s"},
        "c": {"color": "#2ca02c", "marker": "^"},
        "ab": {"color": "#d62728", "marker": "D"},
    }

    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        if model not in agg:
            continue
        for target in ["a", "b", "c", "ab"]:
            mean_key = f"{target}_r2_mean"
            std_key = f"{target}_r2_std"
            if mean_key not in agg[model]:
                continue
            mean = np.asarray(agg[model][mean_key])
            std = np.asarray(agg[model][std_key])
            layers = np.arange(len(mean))
            style = target_styles[target]
            label = f"R²({target})" if target != "ab" else "R²(a·b)"
            ax.plot(
                layers,
                mean,
                marker=style["marker"],
                linewidth=2.0,
                label=label,
                color=style["color"],
            )
            ax.fill_between(
                layers, mean - std, mean + std, alpha=0.15, color=style["color"]
            )

        ax.set_title(MODEL_LABELS[model], fontsize=12)
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.25)
        if model == "tabpfn":
            ax.set_ylabel("R²")
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("Copy Mechanism: R² by Layer (mean ± std across seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_coefficient_probing_with_errorbars(
    agg: dict[str, Any], save_path: Path
) -> None:
    """Publication-quality coefficient probing with error bars."""
    if not agg:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        if model not in agg:
            continue
        for coef, color, marker in [
            ("alpha", "#1f77b4", "o"),
            ("beta", "#ff7f0e", "s"),
        ]:
            mean_key = f"{coef}_r2_mean"
            std_key = f"{coef}_r2_std"
            if mean_key not in agg[model]:
                continue
            mean = np.asarray(agg[model][mean_key])
            std = np.asarray(agg[model][std_key])
            layers = np.arange(len(mean))
            ax.plot(
                layers,
                mean,
                marker=marker,
                linewidth=2.0,
                label=f"R²({coef})",
                color=color,
            )
            ax.fill_between(layers, mean - std, mean + std, alpha=0.2, color=color)

        ax.set_title(MODEL_LABELS[model], fontsize=12)
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.25)
        if model == "tabpfn":
            ax.set_ylabel("R²")
        ax.legend(loc="best")

    fig.suptitle(
        "Coefficient Probing: R² by Layer (mean ± std across seeds)", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

def plot_patching_with_errorbars(agg: dict[str, Any], save_path: Path) -> None:
    """Publication-quality noising-based causal tracing with error bars."""
    if not agg:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in ["tabpfn", "tabicl"]:
        if model not in agg:
            continue
        mean = np.asarray(agg[model]["sensitivity_mean"])
        std = np.asarray(agg[model]["sensitivity_std"])
        layers = np.arange(len(mean))
        ax.plot(
            layers,
            mean,
            marker="o",
            linewidth=2.0,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
        ax.fill_between(
            layers, mean - std, mean + std, alpha=0.2, color=MODEL_COLORS[model]
        )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Normalized Sensitivity", fontsize=12)
    ax.set_title(
        "Noising-Based Causal Tracing: Sensitivity by Layer (mean \u00b1 std)", fontsize=13
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_realworld_probing_with_errorbars(
    agg: dict[str, Any], save_path: Path
) -> None:
    """Publication-quality real-world probing across datasets."""
    if not agg:
        return

    datasets = [k for k in agg if k != "n_seeds"]
    n_ds = len(datasets)
    if n_ds == 0:
        return

    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5), sharey=False)
    if n_ds == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, datasets):
        for model in MODEL_ORDER:
            if model not in agg[ds_name]:
                continue
            mean = np.asarray(agg[ds_name][model]["r2_mean"])
            std = np.asarray(agg[ds_name][model]["r2_std"])
            layers = np.arange(len(mean))
            ax.plot(
                layers,
                mean,
                marker="o",
                linewidth=2.0,
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )
            ax.fill_between(
                layers, mean - std, mean + std, alpha=0.2, color=MODEL_COLORS[model]
            )

        title = ds_name.replace("_", " ").title()
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.25)
        if ds_name == datasets[0]:
            ax.set_ylabel("R\u00b2")
        ax.legend(loc="best", fontsize=9)
        # Clip y-axis to reasonable range for visibility
        ax.set_ylim(bottom=max(ax.get_ylim()[0], -3.0), top=min(ax.get_ylim()[1], 1.5))

    fig.suptitle(
        "Real-World Probing: R\u00b2 by Layer (mean \u00b1 std across seeds)", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_dirs = find_seed_dirs()
    seeds = [d.name for d in seed_dirs]
    print(f"Found {len(seed_dirs)} seed directories: {seeds}")

    # Aggregate each experiment type
    print("\n─── Aggregating intermediary probing ───")
    intermed_agg = aggregate_intermediary_probing(seed_dirs)

    print("─── Aggregating copy mechanism ───")
    copy_agg = aggregate_copy_mechanism(seed_dirs)

    print("─── Aggregating coefficient probing ───")
    coeff_agg = aggregate_coefficient_probing(seed_dirs)

    print("─── Aggregating CKA ───")
    cka_agg = aggregate_cka(seed_dirs)

    print("─── Aggregating steering ───")
    steering_agg = aggregate_steering(seed_dirs)

    print("─── Aggregating SAE ───")
    sae_agg = aggregate_sae(seed_dirs)

    print("─── Aggregating patching (causal tracing) ───")
    patching_agg = aggregate_patching(seed_dirs)

    print("─── Aggregating real-world probing ───")
    realworld_probing_agg = aggregate_realworld_probing(seed_dirs)

    print("─── Aggregating real-world CKA ───")
    realworld_cka_agg = aggregate_realworld_cka(seed_dirs)

    # Save aggregated results
    all_agg = {
        "seeds": seeds,
        "intermediary_probing": intermed_agg,
        "copy_mechanism": copy_agg,
        "coefficient_probing": coeff_agg,
        "cka": cka_agg,
        "steering": steering_agg,
        "sae": sae_agg,
        "patching": patching_agg,
        "realworld_probing": realworld_probing_agg,
        "realworld_cka": realworld_cka_agg,
    }
    agg_path = OUTPUT_DIR / "aggregated_results.json"
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump(all_agg, f, indent=2)
    print(f"\nSaved aggregated results: {agg_path}")

    # Generate publication-quality plots
    print("\n─── Generating plots ───")
    plot_intermediary_with_errorbars(
        intermed_agg, OUTPUT_DIR / "intermediary_probing_errorbars.png"
    )
    plot_copy_mechanism_with_errorbars(
        copy_agg, OUTPUT_DIR / "copy_mechanism_errorbars.png"
    )
    plot_coefficient_probing_with_errorbars(
        coeff_agg, OUTPUT_DIR / "coefficient_probing_errorbars.png"
    )
    plot_patching_with_errorbars(
        patching_agg, OUTPUT_DIR / "patching_causal_tracing_errorbars.png"
    )
    plot_realworld_probing_with_errorbars(
        realworld_probing_agg, OUTPUT_DIR / "realworld_probing_errorbars.png"
    )
    print(f"\nAll outputs in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
