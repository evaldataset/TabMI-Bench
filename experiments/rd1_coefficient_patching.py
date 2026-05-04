# pyright: reportMissingImports=false
"""RD-1 M5-T2: Activation Patching for Coefficient Encoding.

Tests the causal role of each layer in encoding linear coefficients
(z = αx + βy) by patching activations from a clean run (α=2, β=3) into
a corrupted run (α=5, β=1). If patching layer L restores the clean
prediction, then layer L is causally important for coefficient encoding.

Reference:
    - Gupta et al. "TabPFN Through The Looking Glass" (2026)
      arXiv:2601.08181 — Layer 6 identified as key for coefficient encoding
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

# Coefficient configurations
ALPHA_CLEAN = 2.0
BETA_CLEAN = 3.0
ALPHA_CORRUPT = 5.0
BETA_CORRUPT = 1.0

# Dataset sizes
N_TRAIN = 50
N_TEST = 10


def _build_model() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _generate_shared_data() -> dict[str, Any]:
    """Generate clean and corrupted datasets sharing the SAME X features.

    Both datasets use identical X_train and X_test (same rng seed for features)
    but compute y_train with different coefficients:
        clean:     y = α_clean * x + β_clean * y_col
        corrupted: y = α_corrupt * x + β_corrupt * y_col

    Returns:
        Dict with X_train, X_test, y_train_clean, y_train_corrupt.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_total = N_TRAIN + N_TEST
    X = rng.standard_normal((n_total, 2))
    X_train = X[:N_TRAIN]
    X_test = X[n_total - N_TEST :]

    y_train_clean = ALPHA_CLEAN * X_train[:, 0] + BETA_CLEAN * X_train[:, 1]
    y_train_corrupt = ALPHA_CORRUPT * X_train[:, 0] + BETA_CORRUPT * X_train[:, 1]

    return {
        "X_train": X_train,
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

    # Highlight Layer 5-8 region (key computation zone from base paper)
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

    # Patch effect curve
    ax.plot(
        layers,
        patch_effects,
        marker="o",
        linewidth=2.2,
        color="#d62728",
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
        arrowprops={"arrowstyle": "->", "color": "#d62728"},
        color="#d62728",
    )

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean Patch Effect", fontsize=11)
    ax.set_title(
        "Activation Patching: Causal Role of Layers in Coefficient Encoding\n"
        f"Clean (α={ALPHA_CLEAN}, β={BETA_CLEAN}) → "
        f"Corrupted (α={ALPHA_CORRUPT}, β={BETA_CORRUPT})",
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
    results_dir = ROOT / "results" / "rd1" / "coefficient_patching"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RD-1 M5-T2: Activation Patching — Coefficient Encoding")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}")
    print(f"Clean coefficients:     α={ALPHA_CLEAN}, β={BETA_CLEAN}")
    print(f"Corrupted coefficients: α={ALPHA_CORRUPT}, β={BETA_CORRUPT}")

    # ── Step 1: Generate shared-X datasets ──────────────────────────────
    print("\n[1/4] Generating shared-X datasets ...")
    data = _generate_shared_data()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train_clean = data["y_train_clean"]
    y_train_corrupt = data["y_train_corrupt"]
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # ── Step 2: Fit clean and corrupted models ──────────────────────────
    print("\n[2/4] Fitting clean and corrupted models ...")
    model_clean = _build_model()
    model_clean.fit(X_train, y_train_clean)
    print("  Clean model fitted.")

    model_corrupt = _build_model()
    model_corrupt.fit(X_train, y_train_corrupt)
    print("  Corrupted model fitted.")

    # ── Step 3: Cache clean activations & get baseline predictions ──────
    print("\n[3/4] Caching clean activations & running corrupted baseline ...")
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
    print("\n[4/4] Sweeping activation patching across 12 layers ...")
    patch_effects: list[float] = []

    for layer_idx in range(N_LAYERS):
        preds_patched = patcher_corrupt.patched_run(
            X_test, clean_cache, patch_layer=layer_idx
        )
        effect = compute_patch_effect(preds_clean, preds_corrupt, preds_patched)
        patch_effects.append(effect["mean"])
        print(f"  Layer {layer_idx:2d}: patch effect = {effect['mean']:+.4f}")

    # ── Analysis ────────────────────────────────────────────────────────
    peak_layer = int(np.argmax(np.abs(patch_effects)))
    peak_effect = patch_effects[peak_layer]
    print(f"\n  Peak layer: {peak_layer} (effect = {peak_effect:+.4f})")

    # ── Plot ────────────────────────────────────────────────────────────
    plot_path = results_dir / "patch_effect_curve.png"
    _plot_patch_effect_curve(patch_effects, plot_path)
    print(f"\n  Saved plot: {plot_path}")

    # ── Save results JSON ───────────────────────────────────────────────
    payload: dict[str, Any] = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "n_layers": N_LAYERS,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "alpha_clean": ALPHA_CLEAN,
        "beta_clean": BETA_CLEAN,
        "alpha_corrupt": ALPHA_CORRUPT,
        "beta_corrupt": BETA_CORRUPT,
        "patch_effect_per_layer": patch_effects,
        "peak_layer": peak_layer,
        "peak_effect": peak_effect,
    }

    json_path = results_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved results: {json_path}")

    print(f"\nDone. All outputs in: {results_dir}")


if __name__ == "__main__":
    main()
