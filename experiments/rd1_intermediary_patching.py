# pyright: reportMissingImports=false
"""RD-1 M5-T4: Activation Patching for Intermediary Value (a·b).

Tests the causal role of each layer in encoding the intermediary product
a·b in z = a·b + c by patching activations from a clean run into a
corrupted run. The corruption scales a and b (a*=3, b*=2) so that the
intermediary a·b changes dramatically while c stays the same.

If patching layer L restores the clean prediction, then layer L is
causally important for intermediary (a·b) encoding.

Reference:
    - Gupta et al. "TabPFN Through The Looking Glass" (2026)
      arXiv:2601.08181 — Intermediary a·b peaks at Layer 5-8
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

# Corruption factors for intermediary a·b
SCALE_A = 3.0  # multiply feature 'a' by this in corrupted set
SCALE_B = 2.0  # multiply feature 'b' by this in corrupted set


def _build_model() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _generate_shared_data() -> dict[str, Any]:
    """Generate clean and corrupted datasets for z = a·b + c.

    Both datasets share identical raw X features (3 columns: a, b, c).
    Clean uses X as-is; corrupted scales a by 3x and b by 2x so that
    the intermediary product a·b changes dramatically while c is fixed.

    Returns:
        Dict with X_train, X_test, y_train_clean, y_train_corrupt,
        X_train_clean, X_train_corrupt.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_total = N_TRAIN + N_TEST
    X = rng.uniform(0.5, 3.0, (n_total, 3))  # 3 features: a, b, c
    X_train, X_test = X[:N_TRAIN], X[N_TRAIN:]

    # Clean: standard a·b + c
    X_train_clean = X_train.copy()
    y_train_clean = X_train_clean[:, 0] * X_train_clean[:, 1] + X_train_clean[:, 2]

    # Corrupt: scale a by 3.0, b by 2.0 → a·b becomes much larger
    X_train_corrupt = X_train.copy()
    X_train_corrupt[:, 0] *= SCALE_A
    X_train_corrupt[:, 1] *= SCALE_B
    y_train_corrupt = (
        X_train_corrupt[:, 0] * X_train_corrupt[:, 1] + X_train_corrupt[:, 2]
    )

    return {
        "X_train_clean": X_train_clean,
        "X_train_corrupt": X_train_corrupt,
        "X_test": X_test,
        "y_train_clean": y_train_clean,
        "y_train_corrupt": y_train_corrupt,
    }


def _plot_patch_effect_curve(
    patch_effects: list[float],
    save_path: Path,
) -> None:
    """Plot patch effect across layers with Layer 5-8 highlight.

    Args:
        patch_effects: List of 12 mean patch effects (one per layer).
        save_path: Path to save the figure.
    """
    layers = np.arange(N_LAYERS, dtype=np.int32)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Highlight Layer 5-8 region (key intermediary computation zone)
    ax.axvspan(4.5, 8.5, color="#FFFACD", alpha=0.6, label="Layer 5\u20138 region")

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

    # Patch effect curve
    ax.plot(
        layers,
        patch_effects,
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        zorder=5,
        label="Mean patch effect",
    )

    # Mark peak layer
    peak_layer = int(np.argmax(np.abs(patch_effects)))
    ax.annotate(
        f"Peak: L{peak_layer} ({patch_effects[peak_layer]:.3f})",
        xy=(peak_layer, patch_effects[peak_layer]),
        xytext=(peak_layer + 1.2, patch_effects[peak_layer] + 0.08),
        fontsize=9,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#1f77b4"},
        color="#1f77b4",
    )

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean Patch Effect", fontsize=11)
    ax.set_title(
        "Activation Patching: Causal Role of Layers in Intermediary (a\u00b7b) Encoding\n"
        f"Clean (a\u00d71, b\u00d71) \u2192 Corrupted (a\u00d7{SCALE_A:.0f}, b\u00d7{SCALE_B:.0f})",
        fontweight="bold",
        fontsize=12,
    )
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter_best_layer(
    preds_clean: np.ndarray,
    preds_patched: np.ndarray,
    best_layer: int,
    save_path: Path,
) -> None:
    """Scatter plot: preds_clean (x) vs preds_patched (y) for best layer.

    Shows how well patching restores clean predictions. Perfect
    restoration = points on the diagonal.

    Args:
        preds_clean: Clean predictions, shape [N_test].
        preds_patched: Patched predictions at best layer, shape [N_test].
        best_layer: Layer index used for patching.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        preds_clean,
        preds_patched,
        c="#1f77b4",
        edgecolors="black",
        linewidths=0.5,
        s=60,
        alpha=0.8,
        zorder=5,
    )

    # Diagonal reference
    all_vals = np.concatenate([preds_clean, preds_patched])
    lo, hi = float(np.min(all_vals)) - 0.5, float(np.max(all_vals)) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, alpha=0.4, label="y = x")

    # Correlation
    corr = float(np.corrcoef(preds_clean, preds_patched)[0, 1])
    ax.text(
        0.05,
        0.92,
        f"r = {corr:.4f}",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )

    ax.set_xlabel("preds_clean", fontsize=11)
    ax.set_ylabel("preds_patched", fontsize=11)
    ax.set_title(
        f"Intermediary Patching: Clean vs Patched (Layer {best_layer})",
        fontweight="bold",
        fontsize=12,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results_dir = ROOT / "results" / "rd1" / "intermediary_patching"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RD-1 M5-T4: Activation Patching \u2014 Intermediary (a\u00b7b) Encoding")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}")
    print(f"Clean: a\u00d71, b\u00d71  (standard a\u00b7b + c)")
    print(
        f"Corrupted: a\u00d7{SCALE_A:.0f}, b\u00d7{SCALE_B:.0f}  (inflated a\u00b7b, same c)"
    )

    # ── Step 1: Generate shared-X datasets ──────────────────────────────
    print("\n[1/5] Generating shared-X datasets ...")
    data = _generate_shared_data()
    X_train_clean = data["X_train_clean"]
    X_train_corrupt = data["X_train_corrupt"]
    X_test = data["X_test"]
    y_train_clean = data["y_train_clean"]
    y_train_corrupt = data["y_train_corrupt"]
    print(f"  X_train: {X_train_clean.shape}, X_test: {X_test.shape}")
    print(f"  y_train_clean  mean={np.mean(y_train_clean):.3f}")
    print(f"  y_train_corrupt mean={np.mean(y_train_corrupt):.3f}")

    # ── Step 2: Fit clean and corrupted models ──────────────────────────
    print("\n[2/5] Fitting clean and corrupted models ...")
    model_clean = _build_model()
    model_clean.fit(X_train_clean, y_train_clean)
    print("  Clean model fitted.")

    model_corrupt = _build_model()
    model_corrupt.fit(X_train_corrupt, y_train_corrupt)
    print("  Corrupted model fitted.")

    # ── Step 3: Cache clean activations & get baseline predictions ──────
    print("\n[3/5] Caching clean activations & running corrupted baseline ...")
    patcher_clean = TabPFNActivationPatcher(model_clean)
    preds_clean, clean_cache = patcher_clean.run_with_cache(X_test)
    print(f"  preds_clean: shape={preds_clean.shape}, mean={np.mean(preds_clean):.4f}")

    patcher_corrupt = TabPFNActivationPatcher(model_corrupt)
    preds_corrupt, _ = patcher_corrupt.run_with_cache(X_test)
    print(
        f"  preds_corrupt: shape={preds_corrupt.shape}, "
        f"mean={np.mean(preds_corrupt):.4f}"
    )

    # ── Step 4: Sweep activation patching across all layers ─────────────
    print("\n[4/5] Sweeping activation patching across 12 layers ...")
    patch_effects: list[float] = []
    per_layer_preds: dict[int, np.ndarray] = {}

    for layer_idx in range(N_LAYERS):
        preds_patched = patcher_corrupt.patched_run(
            X_test, clean_cache, patch_layer=layer_idx
        )
        effect = compute_patch_effect(preds_clean, preds_corrupt, preds_patched)
        patch_effects.append(effect["mean"])
        per_layer_preds[layer_idx] = preds_patched
        print(f"  Layer {layer_idx:2d}: patch effect = {effect['mean']:+.4f}")

    # ── Analysis ────────────────────────────────────────────────────────
    peak_layer = int(np.argmax(np.abs(patch_effects)))
    peak_effect = patch_effects[peak_layer]
    print(f"\n  Peak layer: {peak_layer} (effect = {peak_effect:+.4f})")

    # Best layer within 5-8 range (expected key region for intermediary)
    effects_5_8 = {i: abs(patch_effects[i]) for i in range(5, 9)}
    best_layer_5_8 = max(effects_5_8, key=effects_5_8.get)  # type: ignore[arg-type]
    print(
        f"  Best layer in 5\u20138: {best_layer_5_8} (|effect| = {effects_5_8[best_layer_5_8]:.4f})"
    )

    # ── Step 5: Plots ──────────────────────────────────────────────────
    print("\n[5/5] Saving plots ...")

    # Plot 1: patch effect curve
    plot1_path = results_dir / "patch_effect_curve.png"
    _plot_patch_effect_curve(patch_effects, plot1_path)
    print(f"  Saved: {plot1_path}")

    # Plot 2: scatter of preds_clean vs preds_patched for best layer in 5-8
    plot2_path = results_dir / "scatter_best_layer.png"
    _plot_scatter_best_layer(
        preds_clean,
        per_layer_preds[best_layer_5_8],
        best_layer_5_8,
        plot2_path,
    )
    print(f"  Saved: {plot2_path}")

    # ── Save results JSON ───────────────────────────────────────────────
    payload: dict[str, Any] = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "n_layers": N_LAYERS,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "scale_a": SCALE_A,
        "scale_b": SCALE_B,
        "patch_effect_per_layer": patch_effects,
        "peak_layer": peak_layer,
        "peak_effect": peak_effect,
        "best_layer_5_8": best_layer_5_8,
        "effect_layer_5": patch_effects[5],
        "effect_layer_6": patch_effects[6],
        "effect_layer_7": patch_effects[7],
        "effect_layer_8": patch_effects[8],
    }

    json_path = results_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved results: {json_path}")

    print(f"\nDone. All outputs in: {results_dir}")


if __name__ == "__main__":
    main()
