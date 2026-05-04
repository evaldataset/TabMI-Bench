# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Aggregate M28 multi-seed results and generate error-bar plots."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEEDS = [42, 123, 456]
RESULTS_DIR = ROOT / "results" / "rd6" / "tabpfn25_fullscale"


def load_seed_results(seed: int) -> dict[str, Any]:
    path = RESULTS_DIR / f"seed_{seed}" / "results.json"
    with open(path) as f:
        return json.load(f)


def aggregate_metric(values: list[float]) -> dict[str, float]:
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> int:
    print("M28 Aggregation: Loading 3-seed results...")

    results_by_seed = {seed: load_seed_results(seed) for seed in SEEDS}

    # Collect per-layer R² curves
    v2_alpha_curves = []
    v25_alpha_curves = []
    v2_beta_curves = []
    v25_beta_curves = []
    v2_ll_curves = []
    v25_ll_curves = []
    v2_inter_curves = []
    v25_inter_curves = []

    for seed in SEEDS:
        r = results_by_seed[seed]
        v2_alpha_curves.append(r["coefficient_probing"]["v2"]["alpha_r2_by_layer"])
        v25_alpha_curves.append(r["coefficient_probing"]["v2.5"]["alpha_r2_by_layer"])
        v2_beta_curves.append(r["coefficient_probing"]["v2"]["beta_r2_by_layer"])
        v25_beta_curves.append(r["coefficient_probing"]["v2.5"]["beta_r2_by_layer"])
        v2_ll_curves.append(r["logit_lens"]["v2"]["r2_by_layer"])
        v25_ll_curves.append(r["logit_lens"]["v2.5"]["r2_by_layer"])
        v2_inter_curves.append(r["intermediary_probing"]["v2"]["r2_by_layer"])
        v25_inter_curves.append(r["intermediary_probing"]["v2.5"]["r2_by_layer"])

    # Stack and compute stats
    v2_alpha_arr = np.array(v2_alpha_curves)  # [3, 12]
    v25_alpha_arr = np.array(v25_alpha_curves)  # [3, 18]
    v2_beta_arr = np.array(v2_beta_curves)
    v25_beta_arr = np.array(v25_beta_curves)
    v2_ll_arr = np.array(v2_ll_curves)
    v25_ll_arr = np.array(v25_ll_curves)
    v2_inter_arr = np.array(v2_inter_curves)
    v25_inter_arr = np.array(v25_inter_curves)

    # Aggregate CKA
    v2_cka_means = [
        np.mean(results_by_seed[s]["cka"]["v2_adjacent_cka"]) for s in SEEDS
    ]
    v25_cka_means = [
        np.mean(results_by_seed[s]["cka"]["v25_adjacent_cka"]) for s in SEEDS
    ]

    # Summary stats
    summary = {
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "coefficient_alpha": {
            "v2": {
                "max_r2": aggregate_metric([float(np.max(c)) for c in v2_alpha_curves]),
                "peak_layer": aggregate_metric(
                    [int(np.argmax(c)) for c in v2_alpha_curves]
                ),
            },
            "v2.5": {
                "max_r2": aggregate_metric(
                    [float(np.max(c)) for c in v25_alpha_curves]
                ),
                "peak_layer": aggregate_metric(
                    [int(np.argmax(c)) for c in v25_alpha_curves]
                ),
            },
        },
        "logit_lens": {
            "v2": {
                "first_above_05": aggregate_metric(
                    [
                        next((i for i, v in enumerate(c) if v > 0.5), None)
                        for c in v2_ll_curves
                    ]
                ),
                "max_r2": aggregate_metric([float(np.max(c)) for c in v2_ll_curves]),
            },
            "v2.5": {
                "first_above_05": aggregate_metric(
                    [
                        next((i for i, v in enumerate(c) if v > 0.5), None)
                        for c in v25_ll_curves
                    ]
                ),
                "max_r2": aggregate_metric([float(np.max(c)) for c in v25_ll_curves]),
            },
        },
        "intermediary": {
            "v2": {
                "max_r2": aggregate_metric([float(np.max(c)) for c in v2_inter_curves]),
                "peak_layer": aggregate_metric(
                    [int(np.argmax(c)) for c in v2_inter_curves]
                ),
            },
            "v2.5": {
                "max_r2": aggregate_metric(
                    [float(np.max(c)) for c in v25_inter_curves]
                ),
                "peak_layer": aggregate_metric(
                    [int(np.argmax(c)) for c in v25_inter_curves]
                ),
            },
        },
        "cka_mean": {
            "v2": aggregate_metric(v2_cka_means),
            "v2.5": aggregate_metric(v25_cka_means),
        },
    }

    # Save aggregated JSON
    agg_path = RESULTS_DIR / "aggregated" / "aggregated_results.json"
    agg_path.parent.mkdir(exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {agg_path}")

    # Generate error-bar plots
    print("Generating plots...")

    # Plot 1: Coefficient Alpha with error bars
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw layers
    ax = axes[0]
    x_v2 = np.arange(v2_alpha_arr.shape[1])
    x_v25 = np.arange(v25_alpha_arr.shape[1])
    ax.errorbar(
        x_v2,
        v2_alpha_arr.mean(axis=0),
        yerr=v2_alpha_arr.std(axis=0),
        fmt="o-",
        capsize=3,
        label=f"v2 ({v2_alpha_arr.shape[1]}L)",
        lw=2,
    )
    ax.errorbar(
        x_v25,
        v25_alpha_arr.mean(axis=0),
        yerr=v25_alpha_arr.std(axis=0),
        fmt="s-",
        capsize=3,
        label=f"v2.5 ({v25_alpha_arr.shape[1]}L)",
        lw=2,
    )
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("R²(α)")
    ax.set_title("Coefficient α Probing (mean±std, n=3)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: normalized
    ax = axes[1]
    x_v2_norm = np.linspace(0, 1, v2_alpha_arr.shape[1])
    x_v25_norm = np.linspace(0, 1, v25_alpha_arr.shape[1])
    ax.errorbar(
        x_v2_norm,
        v2_alpha_arr.mean(axis=0),
        yerr=v2_alpha_arr.std(axis=0),
        fmt="o-",
        capsize=3,
        label=f"v2 ({v2_alpha_arr.shape[1]}L)",
        lw=2,
    )
    ax.errorbar(
        x_v25_norm,
        v25_alpha_arr.mean(axis=0),
        yerr=v25_alpha_arr.std(axis=0),
        fmt="s-",
        capsize=3,
        label=f"v2.5 ({v25_alpha_arr.shape[1]}L)",
        lw=2,
    )
    ax.set_xlabel("Normalized Layer Position")
    ax.set_ylabel("R²(α)")
    ax.set_title("Coefficient α Probing (normalized)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "aggregated" / "coefficient_alpha_errorbars.png", dpi=180)
    plt.close(fig)

    # Plot 2: Logit Lens with error bars
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x_v2 = np.arange(v2_ll_arr.shape[1])
    x_v25 = np.arange(v25_ll_arr.shape[1])
    ax.errorbar(
        x_v2,
        v2_ll_arr.mean(axis=0),
        yerr=v2_ll_arr.std(axis=0),
        fmt="o-",
        capsize=3,
        label=f"v2 ({v2_ll_arr.shape[1]}L)",
        lw=2,
    )
    ax.errorbar(
        x_v25,
        v25_ll_arr.mean(axis=0),
        yerr=v25_ll_arr.std(axis=0),
        fmt="s-",
        capsize=3,
        label=f"v2.5 ({v25_ll_arr.shape[1]}L)",
        lw=2,
    )
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="R²=0.5")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("R²(predicted vs actual)")
    ax.set_title("Logit Lens (mean±std, n=3)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    x_v2_norm = np.linspace(0, 1, v2_ll_arr.shape[1])
    x_v25_norm = np.linspace(0, 1, v25_ll_arr.shape[1])
    ax.errorbar(
        x_v2_norm,
        v2_ll_arr.mean(axis=0),
        yerr=v2_ll_arr.std(axis=0),
        fmt="o-",
        capsize=3,
        label=f"v2 ({v2_ll_arr.shape[1]}L)",
        lw=2,
    )
    ax.errorbar(
        x_v25_norm,
        v25_ll_arr.mean(axis=0),
        yerr=v25_ll_arr.std(axis=0),
        fmt="s-",
        capsize=3,
        label=f"v2.5 ({v25_ll_arr.shape[1]}L)",
        lw=2,
    )
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="R²=0.5")
    ax.set_xlabel("Normalized Layer Position")
    ax.set_ylabel("R²(predicted vs actual)")
    ax.set_title("Logit Lens (normalized)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "aggregated" / "logit_lens_errorbars.png", dpi=180)
    plt.close(fig)

    # Plot 3: Intermediary with error bars
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x_v2 = np.arange(v2_inter_arr.shape[1])
    x_v25 = np.arange(v25_inter_arr.shape[1])
    ax.errorbar(
        x_v2,
        v2_inter_arr.mean(axis=0),
        yerr=v2_inter_arr.std(axis=0),
        fmt="o-",
        capsize=3,
        label=f"v2 ({v2_inter_arr.shape[1]}L)",
        lw=2,
    )
    ax.errorbar(
        x_v25,
        v25_inter_arr.mean(axis=0),
        yerr=v25_inter_arr.std(axis=0),
        fmt="s-",
        capsize=3,
        label=f"v2.5 ({v25_inter_arr.shape[1]}L)",
        lw=2,
    )
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("R²(a·b)")
    ax.set_title("Intermediary Probing (mean±std, n=3)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    x_v2_norm = np.linspace(0, 1, v2_inter_arr.shape[1])
    x_v25_norm = np.linspace(0, 1, v25_inter_arr.shape[1])
    ax.errorbar(
        x_v2_norm,
        v2_inter_arr.mean(axis=0),
        yerr=v2_inter_arr.std(axis=0),
        fmt="o-",
        capsize=3,
        label=f"v2 ({v2_inter_arr.shape[1]}L)",
        lw=2,
    )
    ax.errorbar(
        x_v25_norm,
        v25_inter_arr.mean(axis=0),
        yerr=v25_inter_arr.std(axis=0),
        fmt="s-",
        capsize=3,
        label=f"v2.5 ({v25_inter_arr.shape[1]}L)",
        lw=2,
    )
    ax.set_xlabel("Normalized Layer Position")
    ax.set_ylabel("R²(a·b)")
    ax.set_title("Intermediary Probing (normalized)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "aggregated" / "intermediary_errorbars.png", dpi=180)
    plt.close(fig)

    # Print summary
    print("\n" + "=" * 60)
    print("M28 3-SEED AGGREGATION SUMMARY")
    print("=" * 60)
    print(f"\nCoefficient Alpha Max R²:")
    print(
        f"  v2:   {summary['coefficient_alpha']['v2']['max_r2']['mean']:.3f} ± {summary['coefficient_alpha']['v2']['max_r2']['std']:.3f}"
    )
    print(
        f"  v2.5: {summary['coefficient_alpha']['v2.5']['max_r2']['mean']:.3f} ± {summary['coefficient_alpha']['v2.5']['max_r2']['std']:.3f}"
    )

    print(f"\nLogit Lens First R²>0.5:")
    print(
        f"  v2:   L{summary['logit_lens']['v2']['first_above_05']['mean']:.1f} (std={summary['logit_lens']['v2']['first_above_05']['std']:.1f})"
    )
    print(
        f"  v2.5: L{summary['logit_lens']['v2.5']['first_above_05']['mean']:.1f} (std={summary['logit_lens']['v2.5']['first_above_05']['std']:.1f})"
    )

    print(f"\nCKA Mean (adjacent layers):")
    print(
        f"  v2:   {summary['cka_mean']['v2']['mean']:.3f} ± {summary['cka_mean']['v2']['std']:.3f}"
    )
    print(
        f"  v2.5: {summary['cka_mean']['v2.5']['mean']:.3f} ± {summary['cka_mean']['v2.5']['std']:.3f}"
    )

    print(f"\nIntermediary Max R²:")
    print(
        f"  v2:   {summary['intermediary']['v2']['max_r2']['mean']:.3f} ± {summary['intermediary']['v2']['max_r2']['std']:.3f}"
    )
    print(
        f"  v2.5: {summary['intermediary']['v2.5']['max_r2']['mean']:.3f} ± {summary['intermediary']['v2.5']['max_r2']['std']:.3f}"
    )

    print(f"\nPlots saved: {RESULTS_DIR / 'aggregated'}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
