#!/usr/bin/env python3
"""Aggregate multi-seed Phase 6 results into mean±std with error-bar plots.

Reads per-seed results from results/rd6_fullscale/seed_*/
and produces aggregated statistics + publication-quality plots.

Usage:
    .venv/bin/python experiments/aggregate_phase6.py
    .venv/bin/python experiments/aggregate_phase6.py --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FULLSCALE_DIR = ROOT / "results" / "rd6_fullscale"
OUTPUT_DIR = FULLSCALE_DIR / "aggregated"
RUN_METADATA_PATH = FULLSCALE_DIR / "run_metadata.json"

MODEL_ORDER = ["tabpfn", "tabicl", "iltm"]
MODEL_LABELS = {"tabpfn": "TabPFN", "tabicl": "TabICL", "iltm": "iLTM"}
MODEL_COLORS = {"tabpfn": "#1f77b4", "tabicl": "#ff7f0e", "iltm": "#2ca02c"}


def _seed_dir(seed: int) -> Path:
    return FULLSCALE_DIR / f"seed_{seed}"


def _seed_dirs_from_metadata() -> list[Path]:
    if not RUN_METADATA_PATH.exists():
        return []

    with RUN_METADATA_PATH.open(encoding="utf-8") as f:
        metadata = json.load(f)

    seeds = metadata.get("seeds", [])
    if not isinstance(seeds, list):
        return []

    dirs: list[Path] = []
    for seed in seeds:
        try:
            seed_int = int(seed)
        except (TypeError, ValueError):
            continue
        path = _seed_dir(seed_int)
        if path.exists():
            dirs.append(path)
    return dirs


def find_seed_dirs(explicit_seeds: list[int] | None = None) -> list[Path]:
    if not FULLSCALE_DIR.exists():
        print(f"Error: {FULLSCALE_DIR} not found. Run run_fullscale_phase6.py first.")
        sys.exit(1)

    if explicit_seeds:
        dirs = [path for seed in explicit_seeds if (path := _seed_dir(seed)).exists()]
    else:
        # Always discover from filesystem to avoid stale metadata mismatch.
        # run_metadata.json records runner provenance only; aggregation
        # should include all available seed directories.
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


# ─── Robust Steering Aggregation ─────────────────────────────────────


def aggregate_robust_steering(seed_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate robust steering results across seeds."""
    results = load_seed_results(seed_dirs, "robust_steering")
    if not results:
        return {}

    agg: dict[str, Any] = {"n_seeds": len(results)}

    for model in ["tabpfn", "tabicl"]:
        # Collect per-layer abs_pearson_r across seeds
        per_layer_rs: dict[str, list[float]] = {}
        for r in results:
            if model not in r:
                continue
            robust = r[model].get("robust", {})
            per_layer = robust.get("per_layer", {})
            for layer_key, layer_data in per_layer.items():
                val = layer_data.get("abs_pearson_r", float("nan"))
                per_layer_rs.setdefault(layer_key, []).append(float(val))

        if per_layer_rs:
            layers_sorted = sorted(per_layer_rs.keys(), key=lambda x: int(x))
            means = [float(np.mean(per_layer_rs[k])) for k in layers_sorted]
            stds = [float(np.std(per_layer_rs[k])) for k in layers_sorted]
            layer_ints = [int(k) for k in layers_sorted]
            best_idx = int(np.argmax(means))
            agg[model] = {
                "layers": layer_ints,
                "abs_r_mean": means,
                "abs_r_std": stds,
                "best_layer": layer_ints[best_idx],
                "best_abs_r_mean": means[best_idx],
                "best_abs_r_std": stds[best_idx],
            }

    return agg


# ─── Improved SAE Aggregation ────────────────────────────────────────


def aggregate_improved_sae(seed_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate improved SAE results across seeds."""
    results = load_seed_results(seed_dirs, "improved_sae")
    if not results:
        return {}

    agg: dict[str, Any] = {"n_seeds": len(results)}
    variants = ["relu_16x", "jumprelu_16x"]
    metrics = ["reconstruction_r2", "sparsity", "max_alpha_corr", "max_beta_corr"]

    for variant in variants:
        variant_agg: dict[str, Any] = {}
        for model in MODEL_ORDER:
            model_metrics: dict[str, float] = {}
            for metric in metrics:
                vals = []
                for r in results:
                    res = r.get("results", {}).get(variant, {}).get(model, {})
                    if metric in res:
                        vals.append(float(res[metric]))
                if vals:
                    model_metrics[f"{metric}_mean"] = float(np.mean(vals))
                    model_metrics[f"{metric}_std"] = float(np.std(vals))
            if model_metrics:
                variant_agg[model] = model_metrics
        if variant_agg:
            agg[variant] = variant_agg

    return agg


# ─── Real-World Expanded Aggregation ─────────────────────────────────


def aggregate_realworld_expanded(seed_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate expanded real-world probing results across seeds."""
    results = load_seed_results(seed_dirs, "realworld_expanded")
    if not results:
        return {}

    agg: dict[str, Any] = {"n_seeds": len(results)}

    # Collect all dataset names
    datasets: set[str] = set()
    for r in results:
        if "datasets" in r:
            datasets.update(r["datasets"].keys())

    for ds_name in sorted(datasets):
        ds_agg: dict[str, Any] = {}
        for model in MODEL_ORDER:
            r2_arrays = []
            for r in results:
                model_data = (
                    r.get("datasets", {})
                    .get(ds_name, {})
                    .get("models", {})
                    .get(model, {})
                )
                if "r2_by_layer" in model_data:
                    r2_arrays.append(
                        np.asarray(model_data["r2_by_layer"], dtype=np.float64)
                    )
            if r2_arrays:
                # Pad arrays to same length (iltm has 3 layers vs 12)
                max_len = max(a.size for a in r2_arrays)
                padded = []
                for a in r2_arrays:
                    if a.size < max_len:
                        padded.append(
                            np.pad(a, (0, max_len - a.size), constant_values=np.nan)
                        )
                    else:
                        padded.append(a)
                stacked = np.stack(padded, axis=0)
                mean = np.nanmean(stacked, axis=0)
                std = np.nanstd(stacked, axis=0)

                # Best layer from mean
                valid = np.isfinite(mean)
                if valid.any():
                    best_idx = int(np.nanargmax(mean))
                    best_r2 = float(mean[best_idx])
                else:
                    best_idx = 0
                    best_r2 = float("nan")

                ds_agg[model] = {
                    "r2_mean": mean.tolist(),
                    "r2_std": std.tolist(),
                    "best_layer": best_idx,
                    "best_r2": best_r2,
                }
        if ds_agg:
            agg[ds_name] = ds_agg

    return agg


# ─── Classification Probing Aggregation ──────────────────────────────


def aggregate_classification_probing(seed_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate classification probing results across seeds."""
    results = load_seed_results(seed_dirs, "classification_probing")
    if not results:
        return {}

    agg: dict[str, Any] = {"n_seeds": len(results)}

    datasets: set[str] = set()
    for r in results:
        if "datasets" in r:
            datasets.update(r["datasets"].keys())

    metric_keys = [
        "accuracy_by_layer",
        "f1_macro_by_layer",
        "pseudo_r2_mcfadden_by_layer",
    ]

    for ds_name in sorted(datasets):
        ds_agg: dict[str, Any] = {}
        for model in MODEL_ORDER:
            model_agg: dict[str, Any] = {}
            for metric_key in metric_keys:
                arrays = []
                for r in results:
                    model_data = (
                        r.get("datasets", {})
                        .get(ds_name, {})
                        .get("models", {})
                        .get(model, {})
                    )
                    if metric_key in model_data:
                        arrays.append(
                            np.asarray(model_data[metric_key], dtype=np.float64)
                        )
                if arrays:
                    max_len = max(a.size for a in arrays)
                    padded = []
                    for a in arrays:
                        if a.size < max_len:
                            padded.append(
                                np.pad(a, (0, max_len - a.size), constant_values=np.nan)
                            )
                        else:
                            padded.append(a)
                    stacked = np.stack(padded, axis=0)
                    short_key = metric_key.replace("_by_layer", "")
                    model_agg[f"{short_key}_mean"] = np.nanmean(
                        stacked, axis=0
                    ).tolist()
                    model_agg[f"{short_key}_std"] = np.nanstd(stacked, axis=0).tolist()

            # Best accuracy
            if "accuracy_mean" in model_agg:
                acc_mean = np.asarray(model_agg["accuracy_mean"])
                valid = np.isfinite(acc_mean)
                if valid.any():
                    best_idx = int(np.nanargmax(acc_mean))
                    model_agg["best_layer"] = best_idx
                    model_agg["best_accuracy_mean"] = float(acc_mean[best_idx])

            if model_agg:
                ds_agg[model] = model_agg
        if ds_agg:
            agg[ds_name] = ds_agg

    return agg


# ─── Attention Comparison Aggregation ────────────────────────────────


def aggregate_attention_comparison(seed_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate attention comparison results across seeds."""
    results = load_seed_results(seed_dirs, "attention_comparison")
    if not results:
        return {}

    agg: dict[str, Any] = {"n_seeds": len(results)}

    # TabPFN: sample_entropy, feature_entropy, sample_head_jsd, feature_head_jsd
    tabpfn_keys = [
        "sample_entropy",
        "feature_entropy",
        "sample_head_jsd",
        "feature_head_jsd",
    ]
    tabpfn_agg: dict[str, Any] = {}
    for key in tabpfn_keys:
        arrays = []
        for r in results:
            val = r.get("models", {}).get("tabpfn", {}).get(key, [])
            if val:
                arrays.append(np.asarray(val, dtype=np.float64))
        if arrays:
            stacked = np.stack(arrays, axis=0)
            tabpfn_agg[f"{key}_mean"] = np.mean(stacked, axis=0).tolist()
            tabpfn_agg[f"{key}_std"] = np.std(stacked, axis=0).tolist()
    if tabpfn_agg:
        agg["tabpfn"] = tabpfn_agg

    # TabICL: entropy, head_jsd
    tabicl_keys = ["entropy", "head_jsd"]
    tabicl_agg: dict[str, Any] = {}
    for key in tabicl_keys:
        arrays = []
        for r in results:
            val = r.get("models", {}).get("tabicl", {}).get(key, [])
            if val:
                arrays.append(np.asarray(val, dtype=np.float64))
        if arrays:
            stacked = np.stack(arrays, axis=0)
            tabicl_agg[f"{key}_mean"] = np.mean(stacked, axis=0).tolist()
            tabicl_agg[f"{key}_std"] = np.std(stacked, axis=0).tolist()
    if tabicl_agg:
        agg["tabicl"] = tabicl_agg

    return agg


# ─── Plotting ────────────────────────────────────────────────────────


def plot_robust_steering(agg: dict[str, Any], save_path: Path) -> None:
    """Error-bar plot of robust steering |r| by layer."""
    if not agg or ("tabpfn" not in agg and "tabicl" not in agg):
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in ["tabpfn", "tabicl"]:
        if model not in agg:
            continue
        layers = np.asarray(agg[model]["layers"])
        mean = np.asarray(agg[model]["abs_r_mean"])
        std = np.asarray(agg[model]["abs_r_std"])
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
    ax.set_ylabel("|Pearson r|", fontsize=12)
    ax.set_title("Robust Multi-Pair Steering: |r| by Layer (mean ± std)", fontsize=13)
    ax.set_ylim(0.0, 1.1)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_improved_sae(agg: dict[str, Any], save_path: Path) -> None:
    """Bar chart of SAE max_alpha_corr by model/variant with error bars."""
    if not agg:
        return

    variants = [v for v in ["relu_16x", "jumprelu_16x"] if v in agg]
    if not variants:
        return

    models = [m for m in MODEL_ORDER if any(m in agg.get(v, {}) for v in variants)]
    n_models = len(models)
    n_variants = len(variants)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bar_width = 0.35
    variant_colors = {"relu_16x": "#4C72B0", "jumprelu_16x": "#DD8452"}
    variant_labels = {"relu_16x": "ReLU 16×", "jumprelu_16x": "JumpReLU 16×"}

    for ax_idx, metric_stem in enumerate(["max_alpha_corr", "max_beta_corr"]):
        ax = axes[ax_idx]
        x = np.arange(n_models)
        for v_idx, variant in enumerate(variants):
            means = []
            stds = []
            for model in models:
                m_data = agg.get(variant, {}).get(model, {})
                means.append(m_data.get(f"{metric_stem}_mean", 0.0))
                stds.append(m_data.get(f"{metric_stem}_std", 0.0))
            offset = (v_idx - (n_variants - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                means,
                bar_width * 0.9,
                yerr=stds,
                label=variant_labels[variant],
                color=variant_colors[variant],
                capsize=4,
                alpha=0.85,
            )

        label = "α" if "alpha" in metric_stem else "β"
        ax.set_ylabel(f"Max {label} Correlation", fontsize=11)
        ax.set_title(f"SAE Feature-{label} Correlation", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models])
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend(fontsize=10)

    fig.suptitle(
        "Improved SAE (16× Expansion): Feature Correlations (mean ± std)", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_realworld_expanded(agg: dict[str, Any], save_path: Path) -> None:
    """Grid plot of R² by layer per dataset with error bars."""
    if not agg:
        return

    datasets = [k for k in agg if k != "n_seeds"]
    n_ds = len(datasets)
    if n_ds == 0:
        return

    n_cols = min(3, n_ds)
    n_rows = int(math.ceil(n_ds / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False
    )

    for idx, ds_name in enumerate(datasets):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        for model in MODEL_ORDER:
            if model not in agg[ds_name]:
                continue
            mean = np.asarray(agg[ds_name][model]["r2_mean"])
            std = np.asarray(agg[ds_name][model]["r2_std"])
            layers = np.arange(len(mean))
            # Filter NaN for iLTM (3 layers padded)
            valid = np.isfinite(mean)
            ax.plot(
                layers[valid],
                mean[valid],
                marker="o",
                linewidth=1.8,
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )
            ax.fill_between(
                layers[valid],
                (mean - std)[valid],
                (mean + std)[valid],
                alpha=0.2,
                color=MODEL_COLORS[model],
            )

        title = ds_name.replace("_", " ").title()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Layer")
        ax.set_ylabel("R²")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim(bottom=max(ax.get_ylim()[0], -3.0), top=min(ax.get_ylim()[1], 1.5))

    # Hide empty axes
    for idx in range(n_ds, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    fig.suptitle("Real-World Probing (Expanded): R² by Layer (mean ± std)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_classification_probing(agg: dict[str, Any], save_path: Path) -> None:
    """Grid plot of accuracy by layer per dataset with error bars."""
    if not agg:
        return

    datasets = [k for k in agg if k != "n_seeds"]
    n_ds = len(datasets)
    if n_ds == 0:
        return

    n_cols = min(3, n_ds)
    n_rows = int(math.ceil(n_ds / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False
    )

    for idx, ds_name in enumerate(datasets):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        for model in MODEL_ORDER:
            m_data = agg.get(ds_name, {}).get(model, {})
            if "accuracy_mean" not in m_data:
                continue
            mean = np.asarray(m_data["accuracy_mean"])
            std = np.asarray(m_data["accuracy_std"])
            layers = np.arange(len(mean))
            valid = np.isfinite(mean)
            ax.plot(
                layers[valid],
                mean[valid],
                marker="o",
                linewidth=1.8,
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )
            ax.fill_between(
                layers[valid],
                (mean - std)[valid],
                (mean + std)[valid],
                alpha=0.2,
                color=MODEL_COLORS[model],
            )

        title = ds_name.replace("_", " ").title()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.1)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    for idx in range(n_ds, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    fig.suptitle("Classification Probing: Accuracy by Layer (mean ± std)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_attention_comparison(agg: dict[str, Any], save_path: Path) -> None:
    """Error-bar plots for entropy + JSD curves across models."""
    if not agg:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Entropy
    ax = axes[0]
    line_specs = [
        ("tabpfn", "sample_entropy", "TabPFN (sample)", "#1f77b4", "o"),
        ("tabpfn", "feature_entropy", "TabPFN (feature)", "#aec7e8", "s"),
        ("tabicl", "entropy", "TabICL", "#ff7f0e", "^"),
    ]
    for model, key, label, color, marker in line_specs:
        m_data = agg.get(model, {})
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        if mean_key not in m_data:
            continue
        mean = np.asarray(m_data[mean_key])
        std = np.asarray(m_data[std_key])
        layers = np.arange(len(mean))
        ax.plot(layers, mean, marker=marker, linewidth=1.8, label=label, color=color)
        ax.fill_between(layers, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Entropy", fontsize=12)
    ax.set_title("Attention Entropy by Layer", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10)

    # (b) Head Specialization (JSD)
    ax = axes[1]
    jsd_specs = [
        ("tabpfn", "sample_head_jsd", "TabPFN (sample)", "#1f77b4", "o"),
        ("tabpfn", "feature_head_jsd", "TabPFN (feature)", "#aec7e8", "s"),
        ("tabicl", "head_jsd", "TabICL", "#ff7f0e", "^"),
    ]
    for model, key, label, color, marker in jsd_specs:
        m_data = agg.get(model, {})
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        if mean_key not in m_data:
            continue
        mean = np.asarray(m_data[mean_key])
        std = np.asarray(m_data[std_key])
        layers = np.arange(len(mean))
        ax.plot(layers, mean, marker=marker, linewidth=1.8, label=label, color=color)
        ax.fill_between(layers, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean Pairwise JSD", fontsize=12)
    ax.set_title("Head Specialization (JSD) by Layer", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10)

    fig.suptitle(
        "Cross-Model Attention Comparison (mean ± std across seeds)", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─── Main ────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Phase 6 results")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Explicit seed list to aggregate. Defaults to run_metadata.json seeds if available.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_dirs = find_seed_dirs(args.seeds)
    seeds = [d.name for d in seed_dirs]
    print(f"Found {len(seed_dirs)} seed directories: {seeds}")

    # Aggregate each experiment
    print("\n─── Aggregating robust steering ───")
    steering_agg = aggregate_robust_steering(seed_dirs)
    print(f"  Models: {[k for k in steering_agg if k != 'n_seeds']}")

    print("─── Aggregating improved SAE ───")
    sae_agg = aggregate_improved_sae(seed_dirs)
    print(f"  Variants: {[k for k in sae_agg if k != 'n_seeds']}")

    print("─── Aggregating real-world expanded ───")
    realworld_agg = aggregate_realworld_expanded(seed_dirs)
    print(f"  Datasets: {[k for k in realworld_agg if k != 'n_seeds']}")

    print("─── Aggregating classification probing ───")
    cls_agg = aggregate_classification_probing(seed_dirs)
    print(f"  Datasets: {[k for k in cls_agg if k != 'n_seeds']}")

    print("─── Aggregating attention comparison ───")
    attn_agg = aggregate_attention_comparison(seed_dirs)
    print(f"  Models: {[k for k in attn_agg if k != 'n_seeds']}")

    # Save aggregated results
    all_agg = {
        "seeds": seeds,
        "robust_steering": steering_agg,
        "improved_sae": sae_agg,
        "realworld_expanded": realworld_agg,
        "classification_probing": cls_agg,
        "attention_comparison": attn_agg,
    }
    agg_path = OUTPUT_DIR / "aggregated_results.json"
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump(all_agg, f, indent=2)
    print(f"\nSaved aggregated results: {agg_path}")

    # Generate publication-quality plots
    print("\n─── Generating plots ───")
    plot_robust_steering(steering_agg, OUTPUT_DIR / "robust_steering_errorbars.png")
    plot_improved_sae(sae_agg, OUTPUT_DIR / "improved_sae_bars.png")
    plot_realworld_expanded(
        realworld_agg, OUTPUT_DIR / "realworld_expanded_errorbars.png"
    )
    plot_classification_probing(
        cls_agg, OUTPUT_DIR / "classification_probing_errorbars.png"
    )
    plot_attention_comparison(
        attn_agg, OUTPUT_DIR / "attention_comparison_errorbars.png"
    )

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
