#!/usr/bin/env python3
"""Phase 8B multi-seed: Real-world steering across 5 seeds.

Direct implementation — does not import from phase8b_realworld_steering.py
to avoid import/reload issues.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr
from sklearn.datasets import fetch_california_housing, load_diabetes, fetch_openml
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

SEEDS = [42, 123, 456, 789, 1024]
DEVICE = os.getenv("DEVICE", "cuda:0")


def _set_global_seed(seed: int) -> None:
    """Seed Python random, NumPy, and PyTorch globally for determinism."""
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
RESULTS_DIR = ROOT / "results" / "phase8b" / "realworld_steering_multiseed"
LAMBDA_VALUES = [float(x) for x in np.linspace(-3, 3, 13)]

DATASETS: dict[str, dict[str, Any]] = {
    "california_housing": {"loader": "california"},
    "diabetes": {"loader": "diabetes"},
    "wine_quality": {"openml_id": 287, "numeric_only": True},
    "bike_sharing": {"openml_id": 44063, "numeric_only": True},
}


def _load_dataset(name: str, seed: int, n_train: int = 200, n_test: int = 100) -> dict | None:
    try:
        rng = np.random.default_rng(seed)
        spec = DATASETS[name]

        if spec.get("loader") == "california":
            data = fetch_california_housing()
            X, y = data.data, data.target
        elif spec.get("loader") == "diabetes":
            data = load_diabetes()
            X, y = data.data, data.target
        elif "openml_id" in spec:
            data = fetch_openml(data_id=spec["openml_id"], as_frame=spec.get("numeric_only", False), parser="auto")
            if spec.get("numeric_only"):
                X = data.data.select_dtypes(include=[np.number]).values
                y = np.asarray(data.target, dtype=np.float64)
            else:
                X, y = data.data, data.target
        else:
            return None

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X, y = X[valid], y[valid]

        n_total = n_train + n_test
        if len(y) < n_total:
            return None

        idx = rng.permutation(len(y))[:n_total]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[idx[:n_train]]).astype(np.float32)
        X_test = scaler.transform(X[idx[n_train:n_total]]).astype(np.float32)
        y_train = y[idx[:n_train]].astype(np.float32)

        return {"X_train": X_train, "X_test": X_test, "y_train": y_train}
    except Exception as e:
        print(f"    [SKIP] {name}: {e}")
        return None


def _run_steering(data: dict, model_name: str, layer: int, seed: int) -> dict:
    X_train, X_test, y_train = data["X_train"], data["X_test"], data["y_train"]

    # Split train into fit (80%) + val (20%) for direction extraction
    rng = np.random.default_rng(seed)
    n = len(y_train)
    n_val = max(int(n * 0.2), 2)
    perm = rng.permutation(n)
    X_val = X_train[perm[:n_val]]
    X_fit, y_fit = X_train[perm[n_val:]], y_train[perm[n_val:]]

    median_y = float(np.median(y_fit))
    high_mask = y_fit >= median_y
    low_mask = y_fit < median_y

    if model_name == "tabpfn":
        from tabpfn import TabPFNRegressor
        from src.hooks.steering_vector import TabPFNSteeringVector, compute_steering_effect
        model = TabPFNRegressor(device=DEVICE, model_path="tabpfn-v2-regressor.ckpt")
        steerer = TabPFNSteeringVector(model)
    else:
        from tabicl import TabICLRegressor
        from src.hooks.tabicl_steering import TabICLSteeringVector, compute_steering_effect
        model = TabICLRegressor(device=DEVICE, random_state=seed)
        steerer = TabICLSteeringVector(model)

    direction = steerer.extract_direction(
        X_fit[high_mask], y_fit[high_mask],
        X_fit[low_mask], y_fit[low_mask],
        X_test,
        layer=layer,
        X_val=X_val,
    )

    sweep = steerer.sweep_lambda(X_test, layer=layer, direction=direction, lambdas=LAMBDA_VALUES)
    effect = compute_steering_effect(sweep["lambdas"], sweep["mean_preds"])

    return {
        "pearson_r": float(effect["pearson_r"]),
        "abs_r": float(abs(effect["pearson_r"])),
        "slope": float(effect["slope"]),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("Phase 8B: Real-World Steering — 5-Seed")
    print(f"Seeds={SEEDS}, device={DEVICE}")
    print("=" * 72)

    models_layers = [("tabpfn", 6), ("tabicl", 10)]
    ds_names = list(DATASETS.keys())

    all_raw: dict[str, dict] = {}
    agg: dict[str, dict] = {}

    for ds_name in ds_names:
        all_raw[ds_name] = {}
        agg[ds_name] = {}
        for model_name, layer in models_layers:
            rs = []
            for seed in SEEDS:
                _set_global_seed(seed)
                print(f"  [{ds_name}] [{model_name}] seed={seed}...", end=" ", flush=True)
                data = _load_dataset(ds_name, seed)
                if data is None:
                    print("SKIP")
                    continue
                try:
                    result = _run_steering(data, model_name, layer, seed)
                    rs.append(result["abs_r"])
                    all_raw.setdefault(ds_name, {}).setdefault(model_name, {})[str(seed)] = result
                    print(f"|r|={result['abs_r']:.3f}")
                except Exception as e:
                    print(f"FAIL: {e}")

            if rs:
                agg[ds_name][model_name] = {
                    "mean_abs_r": float(np.mean(rs)),
                    "std": float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0,
                    "n": len(rs),
                }

    # Save
    with (RESULTS_DIR / "raw_results.json").open("w") as f:
        json.dump(all_raw, f, indent=2, default=str)
    with (RESULTS_DIR / "aggregated.json").open("w") as f:
        json.dump(agg, f, indent=2)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"{'Dataset':<25} {'Model':<10} {'|r| (mean±std)':<20} {'n'}")
    for ds_name in ds_names:
        for model_name, _ in models_layers:
            a = agg.get(ds_name, {}).get(model_name, {})
            if a:
                print(f"  {ds_name:<23} {model_name:<10} {a['mean_abs_r']:.3f}±{a['std']:.3f}           {a['n']}")
    print(f"\nSaved: {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
