#!/usr/bin/env python3
"""N=10K 3-seed scale validation.

Runs core intermediary probing and noising-based causal tracing at
N_train=10000, N_test=1000 across 3 seeds to strengthen scale evidence.
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

from src.data.synthetic_generator import generate_linear_data
from src.probing.linear_probe import LinearProbe

SEEDS = [42, 123, 456]
DEVICE = os.getenv("DEVICE", "cuda:1")
N_TRAIN = 10000
N_TEST = 1000
RESULTS_DIR = ROOT / "results" / "scale_10k_multiseed"


def _run_probing_seed(seed: int) -> dict:
    """Run intermediary probing at N=10K for one seed."""
    from tabpfn import TabPFNRegressor
    from tabicl import TabICLRegressor
    from src.hooks.tabpfn_hooker import TabPFNHookedModel
    from src.hooks.tabicl_hooker import TabICLHookedModel

    rng = np.random.default_rng(seed)
    alpha, beta = rng.uniform(1, 5, size=2)
    n = N_TRAIN + N_TEST
    features = rng.standard_normal((n, 3)).astype(np.float64)
    a, b, c = features[:, 0], features[:, 1], features[:, 2]
    intermediary = a * b
    y = (intermediary + c + 0.1 * rng.standard_normal(n)).astype(np.float64)

    X_train, X_test = features[:N_TRAIN], features[N_TRAIN:]
    y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]
    inter_test = intermediary[N_TRAIN:]

    results = {}

    # TabPFN
    print(f"    [tabpfn] seed={seed}")
    tabpfn = TabPFNRegressor(device=DEVICE, model_path="tabpfn-v2-regressor.ckpt")
    tabpfn.fit(X_train.astype(np.float32), y_train.astype(np.float32))
    hooker = TabPFNHookedModel(tabpfn)
    _, cache = hooker.forward_with_cache(X_test.astype(np.float32))
    n_layers = len(cache["layers"])
    r2_by_layer = []
    n_test_actual = len(inter_test)
    for l in range(n_layers):
        act = cache["layers"][l]
        if act.ndim == 4:
            # TabPFN: [1, n_samples, n_features, hidden] — take label token
            act = act[0, :, -1, :]
        elif act.ndim == 3:
            act = act[0]
        act_np = act.detach().cpu().numpy().astype(np.float64) if hasattr(act, 'detach') else np.asarray(act, dtype=np.float64)
        # TabPFN concatenates train+test; take only test samples
        if act_np.shape[0] > n_test_actual:
            act_np = act_np[-n_test_actual:]
        probe = LinearProbe(random_seed=seed)
        probe.fit(act_np, inter_test)
        scores = probe.score(act_np, inter_test)
        r2_by_layer.append(float(scores["r2"]))
    results["tabpfn"] = {
        "r2_by_layer": r2_by_layer,
        "peak_r2": float(max(r2_by_layer)),
        "profile_variance": float(np.var(r2_by_layer)),
    }

    # TabICL
    print(f"    [tabicl] seed={seed}")
    tabicl = TabICLRegressor(device=DEVICE, random_state=seed)
    tabicl.fit(X_train.astype(np.float32), y_train.astype(np.float32))
    hooker_icl = TabICLHookedModel(tabicl)
    _, cache_icl = hooker_icl.forward_with_cache(X_test.astype(np.float32))
    n_layers_icl = hooker_icl.num_layers
    r2_by_layer_icl = []
    for l in range(n_layers_icl):
        act = hooker_icl.get_layer_activations(cache_icl, l)
        act_np = np.asarray(act, dtype=np.float64)
        if act_np.ndim > 2:
            act_np = act_np.reshape(act_np.shape[0], -1)
        probe = LinearProbe(random_seed=seed)
        probe.fit(act_np, inter_test)
        scores = probe.score(act_np, inter_test)
        r2_by_layer_icl.append(float(scores["r2"]))
    results["tabicl"] = {
        "r2_by_layer": r2_by_layer_icl,
        "peak_r2": float(max(r2_by_layer_icl)),
        "profile_variance": float(np.var(r2_by_layer_icl)),
    }

    return results


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 72)
    print(f"N=10K Multi-Seed Scale Validation (seeds={SEEDS})")
    print(f"N_train={N_TRAIN}, N_test={N_TEST}, device={DEVICE}")
    print("=" * 72)

    all_results = {}
    for seed in SEEDS:
        t0 = time.perf_counter()
        print(f"\n--- Seed {seed} ---")
        all_results[seed] = _run_probing_seed(seed)
        print(f"    Done ({time.perf_counter() - t0:.1f}s)")

    # Save
    out_path = RESULTS_DIR / "results.json"
    with out_path.open("w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    for model in ["tabpfn", "tabicl"]:
        peaks = [all_results[s][model]["peak_r2"] for s in SEEDS]
        pvars = [all_results[s][model]["profile_variance"] for s in SEEDS]
        print(f"  {model}: peak R² = {np.mean(peaks):.4f}±{np.std(peaks, ddof=1):.4f}, "
              f"σ²_profile = {np.mean(pvars):.4f}±{np.std(pvars, ddof=1):.4f}")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
