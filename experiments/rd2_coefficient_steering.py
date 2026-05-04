# pyright: reportMissingImports=false
"""RD-2 M9-T2: Coefficient Steering via Contrastive Direction Vectors.

Extracts a steering direction for the α coefficient from a contrastive
dataset pair (α=2 vs α=5), then applies steering at Layer 6 across a
range of λ values to verify that model predictions shift proportionally.

Methodology:
    1. Generate two datasets sharing X but with different α values.
    2. Extract direction: v̂ = normalize(mean(act_high) - mean(act_low))
    3. Steer: add λ·v̂ to Layer 6 label-token activations.
    4. Measure correlation between λ and mean prediction shift.

Reference:
    - Turner et al. "Activation Addition" (2023)
    - Todd et al. "Function Vectors in Large Language Models" (ICLR 2024)
    - Gupta et al. "TabPFN Through The Looking Glass" arXiv:2601.08181
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

from src.hooks.steering_vector import (  # noqa: E402
    TabPFNSteeringVector,
    compute_steering_effect,
)

QUICK_RUN = True
RANDOM_SEED = 42
N_LAYERS = 12

ALPHA_HIGH = 5.0
ALPHA_LOW = 2.0
BETA_SHARED = 3.0

ALPHA_BASELINE = 2.0
BETA_BASELINE = 3.0

N_TRAIN = 50
N_TEST = 10

STEERING_LAYER = 6
LAMBDA_VALUES = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]


def _build_model() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _generate_contrastive_data() -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    n_total = N_TRAIN + N_TEST
    X_all = rng.standard_normal((n_total, 2))

    X_train = X_all[:N_TRAIN]
    X_test = X_all[N_TRAIN:]

    y_train_high = ALPHA_HIGH * X_train[:, 0] + BETA_SHARED * X_train[:, 1]
    y_train_low = ALPHA_LOW * X_train[:, 0] + BETA_SHARED * X_train[:, 1]
    y_train_baseline = ALPHA_BASELINE * X_train[:, 0] + BETA_BASELINE * X_train[:, 1]

    y_test_baseline = ALPHA_BASELINE * X_test[:, 0] + BETA_BASELINE * X_test[:, 1]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_high": y_train_high,
        "y_train_low": y_train_low,
        "y_train_baseline": y_train_baseline,
        "y_test_baseline": y_test_baseline,
    }


def _plot_steering_effect(
    sweep_results: dict[str, Any],
    effect: dict[str, float],
    save_path: Path,
) -> None:
    lambdas = np.asarray(sweep_results["lambdas"], dtype=np.float64)
    mean_preds = np.asarray(
        [sweep_results["mean_preds"][float(lam)] for lam in lambdas],
        dtype=np.float64,
    )

    slope = float(effect["slope"])
    intercept = float(np.mean(mean_preds) - slope * np.mean(lambdas))
    reg_line = slope * lambdas + intercept

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.axhline(
        y=float(np.mean(mean_preds)), color="gray", linestyle="--", linewidth=1.0
    )
    ax1.plot(
        lambdas,
        mean_preds,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="Mean prediction",
        zorder=5,
    )
    ax1.plot(
        lambdas,
        reg_line,
        linestyle="--",
        linewidth=2,
        color="#d62728",
        label="Linear fit",
        zorder=4,
    )
    ax1.set_xlabel("Steering strength λ", fontsize=11)
    ax1.set_ylabel("Mean prediction", fontsize=11)
    ax1.set_title("Steering Effect Curve", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.text(
        0.03,
        0.97,
        f"r = {effect['pearson_r']:.4f}\n"
        f"p = {effect['pearson_p']:.3e}\n"
        f"slope = {effect['slope']:.4f}\n"
        f"range = {effect['prediction_range']:.4f}",
        transform=ax1.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    selected_lambdas = [-2.0, 0.0, 2.0]
    preds_by_lambda = sweep_results["predictions"]
    n_points = len(np.asarray(preds_by_lambda[selected_lambdas[0]], dtype=np.float64))
    x = np.arange(n_points)
    width = 0.25
    colors = ["#d62728", "#1f77b4", "#2ca02c"]

    for idx, lambda_val in enumerate(selected_lambdas):
        preds = np.asarray(preds_by_lambda[lambda_val], dtype=np.float64)
        ax2.bar(
            x + (idx - 1) * width,
            preds,
            width=width,
            color=colors[idx],
            alpha=0.85,
            label=f"λ={lambda_val:+.1f}",
            zorder=5,
        )

    ax2.axhline(y=0.0, color="gray", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Test sample index", fontsize=11)
    ax2.set_ylabel("Prediction", fontsize=11)
    ax2.set_title("Prediction Distribution at λ = -2, 0, +2", fontweight="bold")
    ax2.set_xticks(x)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(fontsize=9)

    fig.suptitle(
        "RD-2 Coefficient Steering: Contrastive Direction at Layer 6",
        fontweight="bold",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results_dir = ROOT / "results" / "rd2" / "coefficient_steering"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RD-2 M9-T2: Coefficient Steering via Contrastive Direction Vectors")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}")
    print(
        "Contrastive pair: "
        f"high(α={ALPHA_HIGH}, β={BETA_SHARED}) vs low(α={ALPHA_LOW}, β={BETA_SHARED})"
    )
    print(f"Baseline: α={ALPHA_BASELINE}, β={BETA_BASELINE}")
    print(f"Steering layer: {STEERING_LAYER}, lambdas={LAMBDA_VALUES}")

    print("\n[1/6] Generating contrastive and baseline datasets ...")
    data = _generate_contrastive_data()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train_high = data["y_train_high"]
    y_train_low = data["y_train_low"]
    y_train_baseline = data["y_train_baseline"]
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    print("\n[2/6] Fitting baseline model and extracting steering direction ...")
    model = _build_model()
    model.fit(X_train, y_train_baseline)
    print("  Baseline model fitted (precondition for steerer).")

    steerer = TabPFNSteeringVector(model)
    direction = steerer.extract_direction(
        X_train,
        y_train_high,
        X_train,
        y_train_low,
        X_test,
        layer=STEERING_LAYER,
        X_val=X_train,  # Use train features (no test-set leakage).
    )
    print(
        f"  Direction extracted: shape={direction.shape}, norm={np.linalg.norm(direction):.6f}"
    )

    print("\n[3/6] Re-fitting baseline and sweeping steering λ values ...")
    model.fit(X_train, y_train_baseline)
    sweep_results = steerer.sweep_lambda(
        X_test,
        layer=STEERING_LAYER,
        direction=direction,
        lambdas=LAMBDA_VALUES,
    )

    for lambda_val in sweep_results["lambdas"]:
        mean_pred = sweep_results["mean_preds"][float(lambda_val)]
        print(f"  λ={lambda_val:+.1f}: mean prediction={mean_pred:+.4f}")

    print("\n[4/6] Computing steering effect metrics ...")
    effect = compute_steering_effect(
        sweep_results["lambdas"], sweep_results["mean_preds"]
    )
    print(f"  Pearson r: {effect['pearson_r']:+.4f} (p={effect['pearson_p']:.3e})")
    print(f"  Slope: {effect['slope']:+.4f}")
    print(f"  Prediction range: {effect['prediction_range']:.4f}")

    print("\n[5/6] Plotting steering effect figure ...")
    plot_path = results_dir / "steering_effect.png"
    _plot_steering_effect(sweep_results, effect, plot_path)
    print(f"  Saved plot: {plot_path}")

    print("\n[6/6] Saving results JSON ...")
    mean_preds_map = {
        str(float(lam)): float(sweep_results["mean_preds"][float(lam)])
        for lam in sweep_results["lambdas"]
    }
    payload: dict[str, Any] = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "n_layers": N_LAYERS,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "steering_layer": STEERING_LAYER,
        "alpha_high": ALPHA_HIGH,
        "alpha_low": ALPHA_LOW,
        "beta_shared": BETA_SHARED,
        "alpha_baseline": ALPHA_BASELINE,
        "beta_baseline": BETA_BASELINE,
        "lambda_values": [float(v) for v in sweep_results["lambdas"]],
        "mean_preds": mean_preds_map,
        "pearson_r": float(effect["pearson_r"]),
        "pearson_p": float(effect["pearson_p"]),
        "slope": float(effect["slope"]),
        "prediction_range": float(effect["prediction_range"]),
    }

    json_path = results_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved results: {json_path}")

    print(f"\nDone. All outputs in: {results_dir}")


if __name__ == "__main__":
    main()
