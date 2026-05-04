# pyright: reportMissingImports=false
"""RD-1 M5-T3: All-Layer Patching Sweep on z = a·b + c Intermediary Data.

Sweeps activation patching across all 12 layers for 3 corruption conditions
on z = a·b + c data to identify which layers are causally important for
encoding each component (a, b, c) of the intermediary computation.

Corruption conditions:
    1. a-corrupted: a scaled 4× (a·b changes from 2.0 to 8.0)
    2. b-corrupted: b scaled 2.5× (a·b changes from 2.0 to 5.0)
    3. c-corrupted: c shifted to 4.0 (additive offset changes)

For each condition, shared X features are generated with the same rng seed,
then y_train is computed from different parameter values to create clean
vs corrupted models.

Reference:
    - Gupta et al. "TabPFN Through The Looking Glass" (2026)
      arXiv:2601.08181 — Layer 5-8 identified as key for intermediary encoding
    - Meng et al. "Locating and Editing Factual Associations in GPT" (2022)
      arXiv:2202.05262 — Activation patching methodology
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rd5_config import cfg

from src.hooks.activation_patcher import (  # noqa: E402
    TabPFNActivationPatcher,
    compute_patch_effect,
)

QUICK_RUN = True
RANDOM_SEED = 42
N_LAYERS = 12

# Dataset sizes
N_TRAIN = 50
N_TEST = 10

# Clean parameter values for z = a·b + c
A_CLEAN = 1.0
B_CLEAN = 2.0
C_CLEAN = 1.5

# Corruption conditions: (name, column_index, clean_value, corrupt_value)
CONDITIONS: list[dict[str, Any]] = [
    {
        "name": "a-corrupted",
        "label": "a corrupted (a: 1.0→4.0)",
        "col": 0,
        "clean_val": A_CLEAN,
        "corrupt_val": 4.0,
        "color": "#d62728",
    },
    {
        "name": "b-corrupted",
        "label": "b corrupted (b: 2.0→5.0)",
        "col": 1,
        "clean_val": B_CLEAN,
        "corrupt_val": 5.0,
        "color": "#1f77b4",
    },
    {
        "name": "c-corrupted",
        "label": "c corrupted (c: 1.5→4.0)",
        "col": 2,
        "clean_val": C_CLEAN,
        "corrupt_val": 4.0,
        "color": "#2ca02c",
    },
]


def _build_model() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _generate_condition_data(
    condition: dict[str, Any],
) -> dict[str, Any]:
    """Generate clean and corrupted datasets for a single corruption condition.

    Both datasets share identical X features (same rng seed). The corrupted
    dataset replaces one column's values to change the y_train computation.

    Args:
        condition: Dict with keys 'col', 'corrupt_val' describing the corruption.

    Returns:
        Dict with X_train, X_test, y_train_clean, y_train_corrupt.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_total = N_TRAIN + N_TEST

    # Features: 3 columns (a, b, c) ~ Uniform[0.5, 3.0]
    X = rng.uniform(0.5, 3.0, (n_total, 3))
    X_train, X_test = X[:N_TRAIN], X[N_TRAIN:]

    # Clean y: z = a·b + c
    X_train_clean = X_train.copy()
    y_train_clean = X_train_clean[:, 0] * X_train_clean[:, 1] + X_train_clean[:, 2]

    # Corrupted y: scale/shift the target column
    X_train_corrupt = X_train.copy()
    col = condition["col"]
    scale = condition["corrupt_val"] / condition["clean_val"]
    X_train_corrupt[:, col] = X_train_corrupt[:, col] * scale
    y_train_corrupt = (
        X_train_corrupt[:, 0] * X_train_corrupt[:, 1] + X_train_corrupt[:, 2]
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_clean": y_train_clean,
        "y_train_corrupt": y_train_corrupt,
    }


def _run_patching_sweep(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Run activation patching sweep across all 12 layers for one condition.

    Args:
        data: Dict with X_train, X_test, y_train_clean, y_train_corrupt.

    Returns:
        Dict with patch_effects list, peak_layer, and prediction arrays.
    """
    X_train = data["X_train"]
    X_test = data["X_test"]

    # Fit clean model
    model_clean = _build_model()
    model_clean.fit(X_train, data["y_train_clean"])
    patcher_clean = TabPFNActivationPatcher(model_clean)
    preds_clean, clean_cache = patcher_clean.run_with_cache(X_test)

    # Fit corrupted model
    model_corrupt = _build_model()
    model_corrupt.fit(X_train, data["y_train_corrupt"])
    patcher_corrupt = TabPFNActivationPatcher(model_corrupt)
    preds_corrupt, _ = patcher_corrupt.run_with_cache(X_test)

    # Sweep all layers
    patch_effects: list[float] = []
    for layer_idx in range(N_LAYERS):
        preds_patched = patcher_corrupt.patched_run(
            X_test, clean_cache, patch_layer=layer_idx
        )
        effect = compute_patch_effect(preds_clean, preds_corrupt, preds_patched)
        patch_effects.append(effect["mean"])

    peak_layer = int(np.argmax(np.abs(patch_effects)))

    return {
        "patch_effects": patch_effects,
        "peak_layer": peak_layer,
        "peak_effect": patch_effects[peak_layer],
        "preds_clean_mean": float(np.mean(preds_clean)),
        "preds_corrupt_mean": float(np.mean(preds_corrupt)),
    }


def _plot_condition_sweep(
    all_results: list[dict[str, Any]],
    save_path: Path,
) -> None:
    """Plot 3 subplots side by side, one per corruption condition.

    Args:
        all_results: List of sweep result dicts, one per condition.
        save_path: Path to save the figure.
    """
    layers = np.arange(N_LAYERS, dtype=np.int32)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for idx, (cond, result) in enumerate(zip(CONDITIONS, all_results)):
        ax = axes[idx]
        effects = result["patch_effects"]

        # Highlight Layer 5-8 region
        ax.axvspan(4.5, 8.5, color="#FFFACD", alpha=0.6, label="Layer 5–8 region")

        # Reference lines
        ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        # Patch effect curve
        ax.plot(
            layers,
            effects,
            marker="o",
            linewidth=2.2,
            color=cond["color"],
            zorder=5,
            label="Patch effect",
        )

        # Mark peak layer
        peak = result["peak_layer"]
        ax.annotate(
            f"Peak: L{peak} ({effects[peak]:.3f})",
            xy=(peak, effects[peak]),
            xytext=(peak + 1.0, effects[peak] + 0.08),
            fontsize=8,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": cond["color"]},
            color=cond["color"],
        )

        ax.set_xlabel("Layer", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Mean Patch Effect", fontsize=10)
        ax.set_title(cond["label"], fontweight="bold", fontsize=11)
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "RD-1 Layer Sweep: Activation Patching per Condition (z = a·b + c)",
        fontweight="bold",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_sweep(
    all_results: list[dict[str, Any]],
    save_path: Path,
) -> None:
    """Plot all 3 conditions on a single axes for comparison.

    Args:
        all_results: List of sweep result dicts, one per condition.
        save_path: Path to save the figure.
    """
    layers = np.arange(N_LAYERS, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Highlight Layer 5-8 region
    ax.axvspan(4.5, 8.5, color="#FFFACD", alpha=0.6, label="Layer 5–8 region")

    # Reference lines
    ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(
        N_LAYERS - 0.5,
        0.02,
        "no effect",
        fontsize=8,
        color="gray",
        ha="right",
    )
    ax.text(
        N_LAYERS - 0.5,
        1.02,
        "full recovery",
        fontsize=8,
        color="gray",
        ha="right",
    )

    for cond, result in zip(CONDITIONS, all_results):
        effects = result["patch_effects"]
        ax.plot(
            layers,
            effects,
            marker="o",
            linewidth=2.2,
            color=cond["color"],
            zorder=5,
            label=cond["label"],
        )

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean Patch Effect", fontsize=11)
    ax.set_title(
        "RD-1 Combined Layer Sweep: All Conditions (z = a·b + c)",
        fontweight="bold",
        fontsize=12,
    )
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results_dir = ROOT / "results" / "rd1" / "layer_sweep"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RD-1 M5-T3: All-Layer Patching Sweep (z = a·b + c)")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}")
    print(f"Clean parameters: a={A_CLEAN}, b={B_CLEAN}, c={C_CLEAN}")
    print(f"Conditions: {len(CONDITIONS)}")

    all_results: list[dict[str, Any]] = []

    for i, cond in enumerate(CONDITIONS):
        step = f"[{i + 1}/{len(CONDITIONS)}]"
        print(f"\n{step} Condition: {cond['name']}")
        print(f"  {cond['label']}")

        # Generate data for this condition
        data = _generate_condition_data(cond)
        print(f"  X_train: {data['X_train'].shape}, X_test: {data['X_test'].shape}")
        print(
            f"  y_clean mean: {np.mean(data['y_train_clean']):.4f}, "
            f"y_corrupt mean: {np.mean(data['y_train_corrupt']):.4f}"
        )

        # Run patching sweep
        result = _run_patching_sweep(data)
        all_results.append(result)

        # Print per-layer effects
        for layer_idx, effect in enumerate(result["patch_effects"]):
            print(f"    Layer {layer_idx:2d}: patch effect = {effect:+.4f}")
        print(f"  Peak layer: {result['peak_layer']} ({result['peak_effect']:+.4f})")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")

    cond_plot_path = results_dir / "condition_sweep.png"
    _plot_condition_sweep(all_results, cond_plot_path)
    print(f"  Saved: {cond_plot_path}")

    combined_plot_path = results_dir / "combined_sweep.png"
    _plot_combined_sweep(all_results, combined_plot_path)
    print(f"  Saved: {combined_plot_path}")

    # ── Save results JSON ──────────────────────────────────────────────────
    payload: dict[str, Any] = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "n_layers": N_LAYERS,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "clean_params": {"a": A_CLEAN, "b": B_CLEAN, "c": C_CLEAN},
        "conditions": {},
    }

    for cond, result in zip(CONDITIONS, all_results):
        payload["conditions"][cond["name"]] = {
            "label": cond["label"],
            "corrupt_val": cond["corrupt_val"],
            "patch_effect_per_layer": result["patch_effects"],
            "peak_layer": result["peak_layer"],
            "peak_effect": result["peak_effect"],
            "preds_clean_mean": result["preds_clean_mean"],
            "preds_corrupt_mean": result["preds_corrupt_mean"],
        }

    json_path = results_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\nDone. All outputs in: {results_dir}")


if __name__ == "__main__":
    main()
