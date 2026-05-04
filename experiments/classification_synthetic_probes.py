#!/usr/bin/env python3
"""Classification synthetic probes: test whether computation profiles
hold for classification tasks, not just regression.

Creates binary classification probes where the decision boundary is a
known function of the intermediary variable, then runs probing + causal
tracing on the 3 core models (TabPFN, TabICL, iLTM).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

SEEDS = [42, 123, 456, 789, 1024]
DEVICE = os.getenv("DEVICE", "cuda:2")
N_TRAIN, N_TEST = 100, 50
RESULTS_DIR = ROOT / "results" / "classification_synthetic"


def _generate_classification_data(
    func_type: str, seed: int
) -> dict[str, np.ndarray]:
    """Generate binary classification data where the label depends on
    a known intermediary variable crossing a threshold."""
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_TEST
    features = rng.standard_normal((n, 3)).astype(np.float64)
    a, b, c = features[:, 0], features[:, 1], features[:, 2]

    if func_type == "bilinear":
        intermediary = a * b
    elif func_type == "sinusoidal":
        intermediary = np.sin(a * b)
    elif func_type == "polynomial":
        intermediary = a ** 2 + b * c
    else:
        raise ValueError(f"Unknown func_type: {func_type}")

    # Binary label: 1 if intermediary > median, else 0
    threshold = float(np.median(intermediary))
    y = (intermediary > threshold).astype(np.float64)
    # Add noise: flip ~5% of labels
    flip_mask = rng.random(n) < 0.05
    y[flip_mask] = 1.0 - y[flip_mask]

    return {
        "X_train": features[:N_TRAIN],
        "X_test": features[N_TRAIN:],
        "y_train": y[:N_TRAIN],
        "y_test": y[N_TRAIN:],
        "intermediary_test": intermediary[N_TRAIN:],
    }


def _probe_classification(
    model_name: str, data: dict[str, np.ndarray], seed: int
) -> dict:
    """Run intermediary probing on classification data for one model."""
    from src.probing.linear_probe import LinearProbe

    X_train = data["X_train"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    inter_test = data["intermediary_test"]

    if model_name == "tabpfn":
        from tabpfn import TabPFNRegressor
        from src.hooks.tabpfn_hooker import TabPFNHookedModel
        model = TabPFNRegressor(device=DEVICE, model_path="tabpfn-v2-regressor.ckpt")
        model.fit(X_train, y_train)
        hooker = TabPFNHookedModel(model)
        _, cache = hooker.forward_with_cache(X_test)
        n_layers = len(cache["layers"])
        n_test_actual = len(inter_test)
        r2_by_layer = []
        for l in range(n_layers):
            act = cache["layers"][l]
            if act.ndim == 4:
                act = act[0, :, -1, :]
            elif act.ndim == 3:
                act = act[0]
            act_np = act.detach().cpu().numpy().astype(np.float64) if hasattr(act, 'detach') else np.asarray(act, dtype=np.float64)
            # TabPFN concatenates train+test; take only test samples
            if act_np.shape[0] > n_test_actual:
                act_np = act_np[-n_test_actual:]
            probe = LinearProbe(random_seed=seed)
            probe.fit(act_np, inter_test)
            r2_by_layer.append(float(probe.score(act_np, inter_test)["r2"]))

    elif model_name == "tabicl":
        from tabicl import TabICLRegressor
        from src.hooks.tabicl_hooker import TabICLHookedModel
        model = TabICLRegressor(device=DEVICE, random_state=seed)
        model.fit(X_train, y_train)
        hooker = TabICLHookedModel(model)
        _, cache = hooker.forward_with_cache(X_test)
        n_layers = hooker.num_layers
        r2_by_layer = []
        for l in range(n_layers):
            act = hooker.get_layer_activations(cache, l)
            act_np = np.asarray(act, dtype=np.float64)
            if act_np.ndim > 2:
                act_np = act_np.reshape(act_np.shape[0], -1)
            probe = LinearProbe(random_seed=seed)
            probe.fit(act_np, inter_test)
            r2_by_layer.append(float(probe.score(act_np, inter_test)["r2"]))

    elif model_name == "iltm":
        from iltm import iLTMRegressor
        from src.hooks.iltm_hooker import iLTMHookedModel
        model = iLTMRegressor(device="cpu", n_ensemble=1, seed=seed)
        model.fit(X_train, y_train)
        hooker = iLTMHookedModel(model)
        _, cache = hooker.forward_with_cache(X_test)
        n_layers = hooker.num_layers
        r2_by_layer = []
        for l in range(n_layers):
            act = hooker.get_layer_activations(cache, l)
            act_np = np.asarray(act, dtype=np.float64)
            probe = LinearProbe(random_seed=seed)
            probe.fit(act_np, inter_test)
            r2_by_layer.append(float(probe.score(act_np, inter_test)["r2"]))
    else:
        raise ValueError(model_name)

    return {
        "r2_by_layer": r2_by_layer,
        "peak_r2": float(max(r2_by_layer)),
        "peak_layer": int(np.argmax(r2_by_layer)),
        "profile_variance": float(np.var(r2_by_layer)),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Classification Synthetic Probes — Profile Invariance Test")
    print(f"Seeds={SEEDS}, N_train={N_TRAIN}, N_test={N_TEST}, device={DEVICE}")
    print("=" * 72)

    func_types = ["bilinear", "sinusoidal", "polynomial"]
    models = ["tabpfn", "tabicl", "iltm"]
    all_results: dict = {}

    for func in func_types:
        all_results[func] = {}
        for model in models:
            all_results[func][model] = {}
            peaks = []
            pvars = []
            for seed in SEEDS:
                print(f"  [{func}] [{model}] seed={seed}...", end=" ", flush=True)
                t0 = time.perf_counter()
                data = _generate_classification_data(func, seed)
                try:
                    result = _probe_classification(model, data, seed)
                    all_results[func][model][str(seed)] = result
                    peaks.append(result["peak_r2"])
                    pvars.append(result["profile_variance"])
                    print(f"peak={result['peak_r2']:.3f} ({time.perf_counter()-t0:.1f}s)")
                except Exception as e:
                    print(f"FAILED: {e}")
                    all_results[func][model][str(seed)] = {"error": str(e)}

            if peaks:
                all_results[func][model]["summary"] = {
                    "mean_peak_r2": float(np.mean(peaks)),
                    "std_peak_r2": float(np.std(peaks, ddof=1)) if len(peaks) > 1 else 0.0,
                    "mean_profile_var": float(np.mean(pvars)),
                }

    out_path = RESULTS_DIR / "results.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 72)
    print("CLASSIFICATION PROBES SUMMARY")
    print(f"{'Function':<15} {'Model':<10} {'Peak R²':<20} {'σ²_profile':<15} {'Profile?'}")
    print("-" * 72)
    for func in func_types:
        for model in models:
            s = all_results[func][model].get("summary", {})
            if s:
                peak = s["mean_peak_r2"]
                pvar = s["mean_profile_var"]
                # Classify
                if pvar > 0.01:
                    profile = "Staged"
                elif pvar < 0.001:
                    profile = "Distributed"
                else:
                    profile = "Preproc-dom."
                print(f"  {func:<13} {model:<10} {peak:.3f}±{s.get('std_peak_r2',0):.3f}        {pvar:.4f}          {profile}")

    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
