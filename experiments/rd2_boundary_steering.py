# pyright: reportMissingImports=false
"""RD-2 M9-T4: Classification Boundary Steering.

Tests whether steering vectors can shift a classification decision boundary.
Uses TabPFNRegressor with binary float targets (0.0/1.0) since no classifier
checkpoint is available.

Methodology:
    1. Generate two classification datasets with different decision boundaries:
       - Boundary A: y = sign(2*x1 + 3*x2)
       - Boundary B: y = sign(4*x1 + 1*x2)
    2. Extract steering direction from contrastive pair (boundary A vs B).
    3. Apply steering at Layer 6 and measure accuracy/decision score shifts.

Reference:
    - Gupta et al. "TabPFN Through The Looking Glass" arXiv:2601.08181
    - Zhao et al. "Probing ICL Decision Boundaries" arXiv:2406.11233
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

# Boundary A coefficients
ALPHA_A = 2.0
BETA_A = 3.0

# Boundary B coefficients
ALPHA_B = 4.0
BETA_B = 1.0

# Dataset sizes
N_TRAIN = 50
N_TEST = 20

# Steering
STEERING_LAYER = 6
LAMBDA_VALUES = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]


def _generate_classification_data(
    alpha: float,
    beta: float,
    n_train: int,
    n_test: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate binary classification data using regressor-compatible targets.

    y_continuous = alpha * x1 + beta * x2
    y_binary = (y_continuous > 0).astype(float)  # 0.0 or 1.0

    TabPFNRegressor is used with float targets since no classifier ckpt is available.
    """
    n_total = n_train + n_test
    X = rng.standard_normal((n_total, 2))
    y_cont = alpha * X[:, 0] + beta * X[:, 1]
    y_binary = (y_cont > 0).astype(float)
    return {
        "X_train": X[:n_train],
        "y_train": y_binary[:n_train],
        "X_test": X[n_train:],
        "y_test": y_binary[n_train:],
        "y_continuous": y_cont[n_train:],
    }


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true.astype(np.int64) == y_pred.astype(np.int64)))


def _analyze_sweep(
    sweep_results: dict[str, Any],
    y_test_a: np.ndarray,
    y_test_b: np.ndarray,
) -> dict[str, Any]:
    lambdas = [float(v) for v in sweep_results["lambdas"]]
    rows: list[dict[str, float]] = []
    acc_a: list[float] = []
    acc_b: list[float] = []
    mean_scores: list[float] = []

    for lambda_val in lambdas:
        preds = np.asarray(sweep_results["predictions"][lambda_val], dtype=np.float64)
        classes = (preds > 0.5).astype(np.int64)
        a = _accuracy(y_test_a, classes)
        b = _accuracy(y_test_b, classes)
        m = float(np.mean(preds))
        rows.append(
            {
                "lambda": lambda_val,
                "accuracy_vs_A": a,
                "accuracy_vs_B": b,
                "mean_decision_score": m,
            }
        )
        acc_a.append(a)
        acc_b.append(b)
        mean_scores.append(m)

    return {
        "rows": rows,
        "acc_a": np.asarray(acc_a, dtype=np.float64),
        "acc_b": np.asarray(acc_b, dtype=np.float64),
        "mean_scores": np.asarray(mean_scores, dtype=np.float64),
    }


def _plot_results(
    sweep_results: dict[str, Any],
    analysis: dict[str, Any],
    effect: dict[str, float],
    save_path: Path,
) -> None:
    lambdas = np.asarray(sweep_results["lambdas"], dtype=np.float64)
    acc_a = np.asarray(analysis["acc_a"], dtype=np.float64)
    acc_b = np.asarray(analysis["acc_b"], dtype=np.float64)
    mean_scores = np.asarray(analysis["mean_scores"], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    ax.plot(lambdas, acc_a, "o-", linewidth=2, color="#1f77b4", label="Acc vs A")
    ax.plot(lambdas, acc_b, "s-", linewidth=2, color="#ff7f0e", label="Acc vs B")
    ax.set_xlabel("Steering strength λ")
    ax.set_ylabel("Accuracy")
    ax.set_title("(a) λ vs Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(lambdas, mean_scores, "o", color="#2ca02c", label="Mean score")
    slope, intercept = np.polyfit(lambdas, mean_scores, 1)
    ax.plot(
        lambdas,
        slope * lambdas + intercept,
        "--",
        color="#d62728",
        label=f"Linear fit (slope={effect['slope']:.3f})",
    )
    ax.set_xlabel("Steering strength λ")
    ax.set_ylabel("Mean prediction")
    ax.set_title(
        f"(b) λ vs Mean Score\nr={effect['pearson_r']:.3f}, p={effect['pearson_p']:.2e}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    for lambda_val, color in [(-2.0, "#9467bd"), (0.0, "#7f7f7f"), (2.0, "#17becf")]:
        preds = np.asarray(sweep_results["predictions"][lambda_val], dtype=np.float64)
        ax.hist(
            preds,
            bins=12,
            alpha=0.45,
            density=True,
            color=color,
            label=f"λ={lambda_val:+.1f}",
        )
    ax.axvline(0.5, linestyle="--", color="black", linewidth=1, alpha=0.8)
    ax.set_xlabel("Raw prediction")
    ax.set_ylabel("Density")
    ax.set_title("(c) Prediction Distribution")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(
        "RD-2 Boundary Steering\n"
        f"A: {ALPHA_A}*x1+{BETA_A}*x2, B: {ALPHA_B}*x1+{BETA_B}*x2, Layer={STEERING_LAYER}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results_dir = ROOT / "results" / "rd2" / "boundary_steering"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RD-2 M9-T4: Classification Boundary Steering")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}, RANDOM_SEED={RANDOM_SEED}")

    print("\n[1/7] Generate boundary A/B datasets with shared X")
    data_a = _generate_classification_data(
        alpha=ALPHA_A,
        beta=BETA_A,
        n_train=N_TRAIN,
        n_test=N_TEST,
        rng=np.random.default_rng(RANDOM_SEED),
    )
    data_b = _generate_classification_data(
        alpha=ALPHA_B,
        beta=BETA_B,
        n_train=N_TRAIN,
        n_test=N_TEST,
        rng=np.random.default_rng(RANDOM_SEED),
    )

    if not np.allclose(data_a["X_train"], data_b["X_train"]):
        raise RuntimeError("X_train mismatch between boundary A/B datasets.")
    if not np.allclose(data_a["X_test"], data_b["X_test"]):
        raise RuntimeError("X_test mismatch between boundary A/B datasets.")

    print(f"  X_train={data_a['X_train'].shape}, X_test={data_a['X_test'].shape}")
    print(
        "  test class balance A/B:",
        np.bincount(data_a["y_test"].astype(np.int64), minlength=2).tolist(),
        np.bincount(data_b["y_test"].astype(np.int64), minlength=2).tolist(),
    )

    print("\n[2/7] Extract steering direction from A (high) vs B (low)")
    model = TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")
    steerer = TabPFNSteeringVector(model)
    direction = steerer.extract_direction(
        data_a["X_train"],
        data_a["y_train"],
        data_b["X_train"],
        data_b["y_train"],
        data_a["X_test"],
        layer=STEERING_LAYER,
        X_val=data_a["X_train"],  # Use train features (no test-set leakage).
    )
    print(f"  direction_norm={np.linalg.norm(direction):.6f}")

    print("\n[3/7] Refit baseline on boundary A, then lambda sweep")
    model.fit(data_a["X_train"], data_a["y_train"])
    sweep_results = steerer.sweep_lambda(
        data_a["X_test"],
        layer=STEERING_LAYER,
        direction=direction,
        lambdas=LAMBDA_VALUES,
    )

    print("\n[4/7] Analyze per-lambda classification/score")
    analysis = _analyze_sweep(
        sweep_results=sweep_results,
        y_test_a=data_a["y_test"],
        y_test_b=data_b["y_test"],
    )

    print("\n[5/7] Compute steering effect on mean raw predictions")
    effect = compute_steering_effect(
        sweep_results["lambdas"], sweep_results["mean_preds"]
    )

    print("\n[6/7] Save 3-panel plot")
    plot_path = results_dir / "boundary_steering.png"
    _plot_results(
        sweep_results=sweep_results,
        analysis=analysis,
        effect=effect,
        save_path=plot_path,
    )
    print(f"  plot={plot_path}")

    print("\n[7/7] Save JSON summary")
    acc_shift_a = float(np.max(analysis["acc_a"]) - np.min(analysis["acc_a"]))
    acc_shift_b = float(np.max(analysis["acc_b"]) - np.min(analysis["acc_b"]))
    score_shift = float(
        np.max(analysis["mean_scores"]) - np.min(analysis["mean_scores"])
    )

    payload: dict[str, Any] = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "boundary_a": {"alpha": ALPHA_A, "beta": BETA_A},
        "boundary_b": {"alpha": ALPHA_B, "beta": BETA_B},
        "steering_layer": STEERING_LAYER,
        "lambdas": [float(v) for v in sweep_results["lambdas"]],
        "per_lambda": analysis["rows"],
        "steering_effect": effect,
        "accuracy_shift_vs_a": acc_shift_a,
        "accuracy_shift_vs_b": acc_shift_b,
        "decision_score_shift": score_shift,
    }

    results_path = results_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  json={results_path}")

    print("\nSuccess criteria")
    print(
        "  Pearson r(λ, mean_score): "
        f"{effect['pearson_r']:.4f} (p={effect['pearson_p']:.2e})"
    )
    print(
        "  Accuracy shift max(max-min): "
        f"{max(acc_shift_a, acc_shift_b):.4f} "
        f"(A={acc_shift_a:.4f}, B={acc_shift_b:.4f})"
    )
    print(
        f"  >10% criterion: {'PASSED' if max(acc_shift_a, acc_shift_b) > 0.10 else 'FAILED'}"
    )
    print(f"\nDone. Outputs in: {results_dir}")


if __name__ == "__main__":
    main()
