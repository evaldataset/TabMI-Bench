# pyright: reportMissingImports=false
"""RD-3 M6-T1: Vector Ablation — Probing Direction Ablation.

Extracts probing direction vectors (via linear Ridge probes) from each
layer's hidden state and ablates them to measure how much of the probed
information is concentrated along that single direction.

A large R² drop after ablation confirms that the target information
(output y for z = αx + βy) is encoded along a single linear direction
in the residual stream.

Methodology:
    1. Fit TabPFN on z = αx + βy data, cache all 12 layer activations.
    2. Per layer: train a linear Ridge probe on train-sample label-token
       activations → extract the weight vector as the probing direction.
    3. Ablate (project out) that direction from the activations.
    4. Re-probe the ablated activations to measure R² drop.

Reference:
    - Gupta et al. "TabPFN Through The Looking Glass" (2026)
      arXiv:2601.08181 — Linear encoding of coefficients in residual stream
    - Elazar et al. "Amnesic Probing: Behavioral Inoculation" (2021)
      arXiv:2006.00995 — Concept erasure via linear projection
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

from src.hooks.tabpfn_hooker import TabPFNHookedModel  # noqa: E402
from src.probing.linear_probe import LinearProbe, probe_layer  # noqa: E402

# ── Global constants ───────────────────────────────────────────────────
QUICK_RUN = True
RANDOM_SEED = 42
N_LAYERS = 12

# Coefficient configuration (z = αx + βy)
ALPHA = 2.0
BETA = 3.0

# Dataset sizes
N_TRAIN = 50
N_TEST = 10

# Probing parameters
PROBE_TEST_SIZE = 0.2


# ── Helper functions ───────────────────────────────────────────────────


def _generate_data() -> dict[str, Any]:
    """Generate z = αx + βy data for probing experiments.

    Returns:
        Dict with X_train, X_test, y_train, y_test arrays.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_total = N_TRAIN + N_TEST
    X_all = rng.standard_normal((n_total, 2))
    X_train = X_all[:N_TRAIN]
    X_test = X_all[N_TRAIN:]

    y_train = ALPHA * X_train[:, 0] + BETA * X_train[:, 1]
    y_test = ALPHA * X_test[:, 0] + BETA * X_test[:, 1]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def _extract_probing_direction(
    activation: np.ndarray, y_targets: np.ndarray
) -> tuple[np.ndarray, LinearProbe]:
    """Extract normalized probing direction from a linear Ridge probe.

    Trains a LinearProbe (complexity=0, Ridge regression) on the full
    activation set, extracts the weight vector, and converts it from
    StandardScaler-transformed space back to original activation space.

    Args:
        activation: [N_train, emsize] label-token activations.
        y_targets: [N_train] target values (y_train).

    Returns:
        Tuple of (v_hat, probe):
            v_hat: [emsize] normalized probing direction in original space.
            probe: Fitted LinearProbe instance.
    """
    probe = LinearProbe(complexity=0, random_seed=RANDOM_SEED)
    probe.fit(activation, y_targets)

    # Ridge coef in scaled space → transform to original activation space
    # StandardScaler: X_scaled = (X - mean) / scale
    # Ridge learns: y ≈ w_scaled @ X_scaled → in original space: w = w_scaled / scale
    v_scaled = probe.model.coef_.flatten()  # [emsize]
    v_original = v_scaled / probe.scaler.scale_  # [emsize]

    norm = np.linalg.norm(v_original)
    v_hat = v_original / norm if norm > 0 else v_original

    return v_hat, probe


def _ablate_direction(activation: np.ndarray, v_hat: np.ndarray) -> np.ndarray:
    """Project out probing direction from activations.

    Removes the component along v_hat from each activation vector:
        ablated = act - (act · v̂) v̂

    Args:
        activation: [N, emsize] activation matrix.
        v_hat: [emsize] unit direction vector to ablate.

    Returns:
        Ablated activations [N, emsize].
    """
    # (act @ v_hat[:, None]) → [N, 1]; * v_hat[None, :] → [N, emsize]
    return activation - (activation @ v_hat[:, None]) * v_hat[None, :]


def _measure_r2(activation: np.ndarray, y_targets: np.ndarray) -> float:
    """Measure linear probe R² on given activations (complexity=0 only).

    Uses probe_layer with an internal 80/20 train/test split.

    Args:
        activation: [N, emsize] activation matrix.
        y_targets: [N] target values.

    Returns:
        R² score (float).
    """
    results = probe_layer(
        activation,
        y_targets,
        complexities=[0],
        test_size=PROBE_TEST_SIZE,
        random_seed=RANDOM_SEED,
    )
    return float(results[0]["r2"])


def _plot_ablation_results(
    original_r2: list[float],
    ablated_r2: list[float],
    r2_drop: list[float],
    save_path: Path,
) -> None:
    """Plot ablation results: R² comparison and R² drop per layer.

    Two subplots:
        (a) R² before and after ablation across layers.
        (b) R² drop per layer (bar chart with max highlighted).

    Args:
        original_r2: List of 12 original R² values.
        ablated_r2: List of 12 ablated R² values.
        r2_drop: List of 12 R² drop values.
        save_path: Path to save the figure.
    """
    layers = np.arange(N_LAYERS, dtype=np.int32)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── (a) R² before and after ablation ──────────────────────────────
    ax1.axvspan(4.5, 8.5, color="#FFFACD", alpha=0.6, label="Layer 5\u20138 region")
    ax1.plot(
        layers,
        original_r2,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="Original R\u00b2",
        zorder=5,
    )
    ax1.plot(
        layers,
        ablated_r2,
        marker="s",
        linewidth=2,
        color="#d62728",
        linestyle="--",
        label="Ablated R\u00b2",
        zorder=5,
    )
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("R\u00b2", fontsize=11)
    ax1.set_title(
        "Linear Probe R\u00b2: Before vs After Ablation",
        fontweight="bold",
    )
    ax1.set_xticks(layers)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)

    # ── (b) R² drop per layer ─────────────────────────────────────────
    ax2.axvspan(4.5, 8.5, color="#FFFACD", alpha=0.6, label="Layer 5\u20138 region")
    bars = ax2.bar(layers, r2_drop, color="#2ca02c", alpha=0.8, zorder=5)

    # Highlight max drop layer
    max_drop_idx = int(np.argmax(r2_drop))
    bars[max_drop_idx].set_color("#d62728")
    bars[max_drop_idx].set_alpha(1.0)
    ax2.annotate(
        f"Max drop: L{max_drop_idx}\n({r2_drop[max_drop_idx]:.3f})",
        xy=(max_drop_idx, r2_drop[max_drop_idx]),
        xytext=(max_drop_idx + 1.5, r2_drop[max_drop_idx] + 0.05),
        fontsize=9,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#d62728"},
        color="#d62728",
    )
    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("R\u00b2 Drop (Original \u2212 Ablated)", fontsize=11)
    ax2.set_title("R\u00b2 Drop After Direction Ablation", fontweight="bold")
    ax2.set_xticks(layers)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Vector Ablation: Probing Direction Removal (z = {ALPHA}x + {BETA}y)",
        fontweight="bold",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    results_dir = ROOT / "results" / "rd3"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RD-3 M6-T1: Vector Ablation \u2014 Probing Direction Ablation")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}")
    print(f"Coefficients: \u03b1={ALPHA}, \u03b2={BETA}")
    print(f"N_train={N_TRAIN}, N_test={N_TEST}")

    # ── Step 1: Generate data and fit model ────────────────────────────
    print("\n[1/4] Generating data and fitting TabPFN ...")
    data = _generate_data()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    model = TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")
    model.fit(X_train, y_train)
    print("  Model fitted.")

    # ── Step 2: Cache activations ──────────────────────────────────────
    print("\n[2/4] Caching layer activations ...")
    hooker = TabPFNHookedModel(model)
    preds, cache = hooker.forward_with_cache(X_test)
    single_eval_pos = int(cache["single_eval_pos"])
    print(f"  Predictions: {preds.shape}, single_eval_pos={single_eval_pos}")

    # ── Step 3: Extract directions and ablate per layer ────────────────
    print("\n[3/4] Extracting probing directions and measuring ablation ...")
    original_r2_list: list[float] = []
    ablated_r2_list: list[float] = []
    r2_drop_list: list[float] = []

    for layer_idx in range(N_LAYERS):
        # Extract train-sample label-token activations
        act_tensor = cache["layers"][layer_idx]
        act = act_tensor[0, :single_eval_pos, -1, :].detach().numpy()
        # act shape: [N_train, 192]

        # Step 3a: Measure original R²
        original_r2 = _measure_r2(act, y_train)

        # Step 3b: Extract probing direction (fit on full training set)
        v_hat, _ = _extract_probing_direction(act, y_train)

        # Step 3c: Ablate direction from activations
        act_ablated = _ablate_direction(act, v_hat)

        # Step 3d: Measure ablated R²
        ablated_r2 = _measure_r2(act_ablated, y_train)

        # R² drop
        r2_drop = original_r2 - ablated_r2

        original_r2_list.append(original_r2)
        ablated_r2_list.append(ablated_r2)
        r2_drop_list.append(r2_drop)

        print(
            f"  Layer {layer_idx:2d}: "
            f"R\u00b2={original_r2:+.4f} \u2192 {ablated_r2:+.4f}  "
            f"(drop={r2_drop:+.4f})"
        )

    # ── Step 4: Plot and save results ──────────────────────────────────
    print("\n[4/4] Saving results ...")

    max_drop_layer = int(np.argmax(r2_drop_list))
    max_drop_value = r2_drop_list[max_drop_layer]
    print(f"  Max R\u00b2 drop at Layer {max_drop_layer}: {max_drop_value:.4f}")

    # Plot
    plot_path = results_dir / "ablation_results.png"
    _plot_ablation_results(
        original_r2_list,
        ablated_r2_list,
        r2_drop_list,
        plot_path,
    )
    print(f"  Saved plot: {plot_path}")

    # Results JSON
    payload: dict[str, Any] = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "alpha": ALPHA,
        "beta": BETA,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_layers": N_LAYERS,
        "original_r2_per_layer": original_r2_list,
        "ablated_r2_per_layer": ablated_r2_list,
        "r2_drop_per_layer": r2_drop_list,
        "max_drop_layer": max_drop_layer,
        "max_drop_value": max_drop_value,
    }

    json_path = results_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved results: {json_path}")

    print(f"\nDone. All outputs in: {results_dir}")


if __name__ == "__main__":
    main()
