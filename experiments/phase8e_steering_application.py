# pyright: reportMissingImports=false
"""Phase 8E: Practical application of computation strategy knowledge.

Demonstrates that knowing the computation strategy enables targeted
prediction improvement: steering at the identified causal layer
corrects systematic prediction errors on real-world data.

Experiment: For each real-world dataset, identify test samples where
the model's prediction has the largest error, then apply targeted
steering at the known causal layer to reduce that error.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false
import json
import os
from typing import Any

import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tabpfn import TabPFNRegressor
from tabicl import TabICLRegressor

from src.hooks.steering_vector import TabPFNSteeringVector, compute_steering_effect
from src.hooks.tabicl_steering import (
    TabICLSteeringVector,
    compute_steering_effect as tabicl_compute_effect,
)
from rd5_config import cfg

RESULTS_DIR = ROOT / "results" / "phase8e" / "steering_application"
LAMBDA_GRID = [round(x, 1) for x in np.linspace(-3, 3, 13).tolist()]


def _load_dataset(name: str, seed: int) -> dict[str, np.ndarray] | None:
    rng = np.random.default_rng(seed)
    n_train, n_test = 200, 100

    if name == "california_housing":
        data = fetch_california_housing()
    elif name == "diabetes":
        data = load_diabetes()
    else:
        return None

    X, y = np.asarray(data.data, dtype=np.float64), np.asarray(data.target, dtype=np.float64)  # type: ignore[union-attr]
    idx = rng.permutation(len(y))[: n_train + n_test]
    X_sub, y_sub = X[idx], y[idx]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_sub[:n_train]).astype(np.float32)
    X_test = scaler.transform(X_sub[n_train:]).astype(np.float32)
    y_train = y_sub[:n_train].astype(np.float32)
    y_test = y_sub[n_train:].astype(np.float32)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def _targeted_steering_correction(
    model_name: str,
    data: dict[str, np.ndarray],
    target_layer: int,
) -> dict[str, Any]:
    """Apply steering to correct predictions toward true values.

    Strategy: Split training data into fit/val subsets, then by median
    target into high/low. Extract steering direction using X_val (not
    X_test). Select optimal lambda on the validation split, then report
    final MSE on X_test.
    """
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Split training data: 80% for fitting, 20% for validation
    rng = np.random.default_rng(cfg.SEED)
    n = len(y_train)
    n_val = max(int(n * 0.2), 2)
    perm = rng.permutation(n)
    val_idx, fit_idx = perm[:n_val], perm[n_val:]

    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]

    median_y = float(np.median(y_fit))
    high_mask = y_fit >= median_y
    low_mask = y_fit < median_y

    if model_name == "tabpfn":
        model = TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")
        steerer = TabPFNSteeringVector(model)
    else:
        model = TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)
        steerer = TabICLSteeringVector(model)

    # Extract steering direction using X_val (not X_test)
    direction = steerer.extract_direction(
        X_fit[high_mask], y_fit[high_mask],
        X_fit[low_mask], y_fit[low_mask],
        X_val,  # backward-compat positional arg
        layer=target_layer,
        X_val=X_val,
    )

    # Baseline predictions (no steering)
    baseline_preds = steerer.sweep_lambda(
        X_test, layer=target_layer, direction=direction, lambdas=[0.0],
    )["predictions"][0.0]
    baseline_mse = float(mean_squared_error(y_test, baseline_preds))

    # Select best lambda on VALIDATION set (not test set)
    val_sweep = steerer.sweep_lambda(
        X_val, layer=target_layer, direction=direction, lambdas=LAMBDA_GRID,
    )

    best_lambda = 0.0
    best_val_mse = float("inf")
    for lam in LAMBDA_GRID:
        if lam in val_sweep.get("predictions", {}):
            val_preds = val_sweep["predictions"][lam]
            val_mse = float(mean_squared_error(y_val, val_preds))
        else:
            val_mse = float("inf")
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_lambda = lam

    # Evaluate best lambda on TEST set
    test_sweep = steerer.sweep_lambda(
        X_test, layer=target_layer, direction=direction, lambdas=[best_lambda],
    )
    best_test_preds = test_sweep["predictions"][best_lambda]
    best_mse = float(mean_squared_error(y_test, best_test_preds))

    # Also collect all lambda MSEs on test for reporting
    full_test_sweep = steerer.sweep_lambda(
        X_test, layer=target_layer, direction=direction, lambdas=LAMBDA_GRID,
    )
    lambda_mse: dict[float, float] = {0.0: baseline_mse}
    for lam in LAMBDA_GRID:
        if lam in full_test_sweep.get("predictions", {}):
            steered_preds = full_test_sweep["predictions"][lam]
            lambda_mse[lam] = float(mean_squared_error(y_test, steered_preds))

    improvement_pct = (baseline_mse - best_mse) / baseline_mse * 100 if baseline_mse > 0 else 0

    return {
        "model": model_name,
        "layer": target_layer,
        "baseline_mse": baseline_mse,
        "best_lambda": best_lambda,
        "best_lambda_selected_on": "validation",
        "best_mse": best_mse,
        "improvement_pct": improvement_pct,
        "lambda_mse": {str(k): v for k, v in lambda_mse.items()},
    }


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("Phase 8E: Steering Application — Targeted Prediction Correction")
    print("=" * 72)

    datasets = ["california_housing", "diabetes"]
    models = [
        ("tabpfn", 6),   # Known causal layer
        ("tabicl", 10),  # Known best steering layer
    ]

    all_results: dict[str, dict[str, Any]] = {}

    for ds_name in datasets:
        print(f"\n--- Dataset: {ds_name} ---")
        data = _load_dataset(ds_name, cfg.SEED)
        if data is None:
            continue

        all_results[ds_name] = {}
        for model_name, layer in models:
            print(f"  {model_name} (L{layer})...")
            try:
                result = _targeted_steering_correction(model_name, data, layer)
                all_results[ds_name][model_name] = result
                print(f"    Baseline MSE: {result['baseline_mse']:.4f}")
                print(f"    Best λ={result['best_lambda']:.1f}, MSE: {result['best_mse']:.4f}")
                print(f"    Improvement: {result['improvement_pct']:.1f}%")
            except Exception as e:
                print(f"    FAILED: {e}")

    json_path = RESULTS_DIR / f"results_seed{cfg.SEED}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY: Steering-Based Prediction Correction")
    print("=" * 72)
    print(f"{'Dataset':<25} {'Model':<10} {'Baseline MSE':<15} {'Best MSE':<15} {'Improvement':<12}")
    for ds_name, ds_results in all_results.items():
        for model_name, result in ds_results.items():
            print(f"{ds_name:<25} {model_name:<10} {result['baseline_mse']:<15.4f} {result['best_mse']:<15.4f} {result['improvement_pct']:<12.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
