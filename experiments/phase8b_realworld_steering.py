# pyright: reportMissingImports=false
"""Phase 8B: Real-world steering validation.

Tests whether steering vectors produce monotonic prediction shifts on
real-world datasets, extending the synthetic-only steering evidence.
Uses median-split contrastive direction: "high target" vs "low target".
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
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

from src.hooks.steering_vector import TabPFNSteeringVector, compute_steering_effect
from src.hooks.tabicl_steering import (
    TabICLSteeringVector,
    compute_steering_effect as tabicl_compute_effect,
)
from rd5_config import cfg

LAMBDA_VALUES = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
TARGET_LAYER_TABPFN = 6
TARGET_LAYER_TABICL = 10

RESULTS_DIR = ROOT / "results" / "phase8b" / "realworld_steering"

# Dataset loaders: name -> (loader_func, n_train, n_test)
DATASETS: dict[str, dict[str, Any]] = {
    "california_housing": {"n_train": 200, "n_test": 50},
    "diabetes": {"n_train": 200, "n_test": 50},
}
if not cfg.QUICK_RUN:
    DATASETS.update({
        "wine_quality": {"n_train": 200, "n_test": 50, "openml_id": 287},
        "abalone": {"n_train": 200, "n_test": 50, "openml_id": 183},
        "bike_sharing": {"n_train": 200, "n_test": 50, "openml_id": 42712, "numeric_only": True},
    })


def _load_dataset(name: str, spec: dict[str, Any], seed: int) -> dict[str, np.ndarray] | None:
    """Load and prepare a real-world dataset."""
    rng = np.random.default_rng(seed)
    n_train = spec["n_train"]
    n_test = spec["n_test"]
    n_total = n_train + n_test

    try:
        if name == "california_housing":
            data = fetch_california_housing()
            X, y = data.data, data.target  # type: ignore[union-attr]
        elif name == "diabetes":
            data = load_diabetes()
            X, y = data.data, data.target  # type: ignore[union-attr]
        elif "openml_id" in spec:
            from sklearn.datasets import fetch_openml
            use_dataframe = spec.get("numeric_only", False)
            data = fetch_openml(data_id=spec["openml_id"], as_frame=use_dataframe, parser="auto")
            if use_dataframe:
                X = data.data.select_dtypes(include=[np.number]).values  # type: ignore[union-attr]
                y = np.asarray(data.target, dtype=np.float64)  # type: ignore[union-attr]
            else:
                X, y = data.data, data.target  # type: ignore[union-attr]
        else:
            print(f"  [SKIP] Unknown dataset: {name}")
            return None

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Remove NaN
        valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X, y = X[valid], y[valid]

        if len(y) < n_total:
            print(f"  [SKIP] {name}: only {len(y)} samples < {n_total}")
            return None

        idx = rng.permutation(len(y))[:n_total]
        X_sub, y_sub = X[idx], y[idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_sub[:n_train]).astype(np.float32)
        X_test = scaler.transform(X_sub[n_train:]).astype(np.float32)
        y_train = y_sub[:n_train].astype(np.float32)
        y_test = y_sub[n_train:].astype(np.float32)

        return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

    except Exception as e:
        print(f"  [SKIP] {name}: {e}")
        return None


def _median_split(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    *, val_fraction: float = 0.2, random_seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split training data by median target into high/low subsets.

    A held-out validation subset (X_val) is carved from X_train for
    steering direction extraction, avoiding test-set leakage.
    """
    rng = np.random.default_rng(random_seed)
    n = len(y_train)
    n_val = max(int(n * val_fraction), 2)
    perm = rng.permutation(n)
    val_idx, fit_idx = perm[:n_val], perm[n_val:]

    X_val = X_train[val_idx]
    X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]

    median = float(np.median(y_fit))
    high_mask = y_fit >= median
    low_mask = y_fit < median

    ds_high = {
        "X_train": X_fit[high_mask],
        "y_train": y_fit[high_mask],
        "X_test": X_test,
        "X_val": X_val,
    }
    ds_low = {
        "X_train": X_fit[low_mask],
        "y_train": y_fit[low_mask],
        "X_test": X_test,
        "X_val": X_val,
    }
    return ds_high, ds_low


def _run_tabpfn_steering(
    ds_high: dict[str, np.ndarray],
    ds_low: dict[str, np.ndarray],
) -> dict[str, Any]:
    model = TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")
    steerer = TabPFNSteeringVector(model)

    direction = steerer.extract_direction(
        ds_high["X_train"], ds_high["y_train"],
        ds_low["X_train"], ds_low["y_train"],
        ds_high["X_test"],
        layer=TARGET_LAYER_TABPFN,
        X_val=ds_high.get("X_val"),
    )
    sweep = steerer.sweep_lambda(
        ds_high["X_test"],
        layer=TARGET_LAYER_TABPFN,
        direction=direction,
        lambdas=LAMBDA_VALUES,
    )
    effect = compute_steering_effect(sweep["lambdas"], sweep["mean_preds"])

    return {
        "layer": TARGET_LAYER_TABPFN,
        "pearson_r": float(effect["pearson_r"]),
        "slope": float(effect["slope"]),
        "prediction_range": float(effect["prediction_range"]),
        "mean_preds": {str(float(k)): float(v) for k, v in sweep["mean_preds"].items()},
    }


def _run_tabicl_steering(
    ds_high: dict[str, np.ndarray],
    ds_low: dict[str, np.ndarray],
) -> dict[str, Any]:
    model = TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)
    steerer = TabICLSteeringVector(model)

    direction = steerer.extract_direction(
        ds_high["X_train"], ds_high["y_train"],
        ds_low["X_train"], ds_low["y_train"],
        ds_high["X_test"],
        layer=TARGET_LAYER_TABICL,
        X_val=ds_high.get("X_val"),
    )
    sweep = steerer.sweep_lambda(
        ds_high["X_test"],
        layer=TARGET_LAYER_TABICL,
        direction=direction,
        lambdas=LAMBDA_VALUES,
    )
    effect = tabicl_compute_effect(sweep["lambdas"], sweep["mean_preds"])

    return {
        "layer": TARGET_LAYER_TABICL,
        "pearson_r": float(effect["pearson_r"]),
        "slope": float(effect["slope"]),
        "prediction_range": float(effect["prediction_range"]),
        "mean_preds": {str(float(k)): float(v) for k, v in sweep["mean_preds"].items()},
    }


def _plot_results(all_results: dict[str, dict[str, Any]], save_dir: Path) -> None:
    datasets_with_results = [d for d in all_results if all_results[d].get("tabpfn")]
    if not datasets_with_results:
        return

    n_ds = len(datasets_with_results)
    fig, axes = plt.subplots(2, n_ds, figsize=(5 * n_ds, 8), squeeze=False)

    for col, ds_name in enumerate(datasets_with_results):
        for row, model_name in enumerate(["tabpfn", "tabicl"]):
            ax = axes[row, col]
            result = all_results[ds_name].get(model_name)
            if not result:
                ax.set_visible(False)
                continue

            lambdas = sorted([float(k) for k in result["mean_preds"]])
            preds = [result["mean_preds"][str(l)] for l in lambdas]

            color = "blue" if model_name == "tabpfn" else "orange"
            ax.plot(lambdas, preds, marker="o", linewidth=2, color=color)
            ax.set_title(f"{ds_name}\n{model_name.upper()} |r|={abs(result['pearson_r']):.3f}", fontsize=10)
            ax.set_xlabel("Lambda")
            ax.set_ylabel("Mean Prediction")
            ax.grid(alpha=0.25)

    fig.suptitle("Phase 8B: Real-World Steering Validation", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / "realworld_steering.png", dpi=180, bbox_inches="tight")
    fig.savefig(save_dir / "realworld_steering.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("Phase 8B: Real-World Steering Validation")
    print("=" * 72)
    print(f"SEED={cfg.SEED}, datasets={list(DATASETS.keys())}")

    all_results: dict[str, dict[str, Any]] = {}

    for ds_name, spec in DATASETS.items():
        print(f"\n--- Dataset: {ds_name} ---")
        data = _load_dataset(ds_name, spec, cfg.SEED)
        if data is None:
            continue

        ds_high, ds_low = _median_split(data["X_train"], data["y_train"], data["X_test"])
        all_results[ds_name] = {}

        print(f"  TabPFN steering (L{TARGET_LAYER_TABPFN})...")
        try:
            result_pfn = _run_tabpfn_steering(ds_high, ds_low)
            all_results[ds_name]["tabpfn"] = result_pfn
            print(f"    |r|={abs(result_pfn['pearson_r']):.4f}, slope={result_pfn['slope']:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")

        print(f"  TabICL steering (L{TARGET_LAYER_TABICL})...")
        try:
            result_icl = _run_tabicl_steering(ds_high, ds_low)
            all_results[ds_name]["tabicl"] = result_icl
            print(f"    |r|={abs(result_icl['pearson_r']):.4f}, slope={result_icl['slope']:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")

    json_path = RESULTS_DIR / f"results_seed{cfg.SEED}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    _plot_results(all_results, RESULTS_DIR)

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY: Real-World Steering |r| by dataset × model")
    print("=" * 72)
    print(f"{'Dataset':<25} {'TabPFN |r|':<15} {'TabICL |r|':<15}")
    for ds_name in all_results:
        pfn_r = abs(all_results[ds_name].get("tabpfn", {}).get("pearson_r", 0))
        icl_r = abs(all_results[ds_name].get("tabicl", {}).get("pearson_r", 0))
        print(f"{ds_name:<25} {pfn_r:<15.4f} {icl_r:<15.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
