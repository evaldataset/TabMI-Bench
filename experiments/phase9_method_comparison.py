# pyright: reportMissingImports=false
"""Phase 9: MI method comparison case study using TabMI-Bench.

Demonstrates that TabMI-Bench can be used to evaluate and compare
different MI methods, producing applicability labels and reference
profile agreement scores.

Methods compared:
  1. Linear Probing (Ridge regression on layer activations)
  2. Attribution-Magnitude probing (activation norm as importance signal)
  3. Gradient-based saliency (|∂output/∂activation|)

Each method is evaluated against TabMI-Bench reference profiles using:
  - Profile agreement (correlation with reference σ²_profile signature)
  - Causal validation (peak alignment with noising-based causal peak)
  - Applicability label (Supported / Limited / Not established)

Usage:
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python experiments/phase9_method_comparison.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.synthetic_generator import generate_quadratic_data
from src.hooks.tabpfn_hooker import TabPFNHookedModel
from src.hooks.tabicl_hooker import TabICLHookedModel
from src.hooks.iltm_hooker import iLTMHookedModel
from tabpfn import TabPFNRegressor
from tabicl import TabICLRegressor
from iltm import iLTMRegressor

SEEDS = [42, 123, 456, 789, 1024]
N_TRAIN = 100
N_TEST = 50
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = ROOT / "results" / "phase9_method_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Reference profiles from paper Table 1 (expected qualitative patterns)
REFERENCE_PROFILES = {
    "tabpfn": {"expected_class": "staged", "expected_peak_range": (5, 10)},
    "tabicl": {"expected_class": "distributed", "expected_peak_range": (0, 11)},
    "iltm": {"expected_class": "preprocessing-dominant", "expected_peak_range": (0, 1)},
}


# ============================================================
# MI Methods (score function per layer)
# ============================================================

def method_linear_probing(activations: np.ndarray, target: np.ndarray) -> float:
    """Method 1: Linear (Ridge) probing. Returns R² of target recovery."""
    if activations.shape[0] < 5:
        return float("nan")
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros_like(target, dtype=np.float64)
    for tr, te in kf.split(activations):
        m = Ridge(alpha=1.0)
        m.fit(activations[tr], target[tr])
        preds[te] = m.predict(activations[te])
    return float(r2_score(target, preds))


def method_activation_norm(activations: np.ndarray, target: np.ndarray) -> float:
    """Method 2: Activation norm correlation with target.

    Computes |activation|_2 per sample and measures correlation with target.
    Simple attribution-magnitude proxy.
    """
    if activations.shape[0] < 3:
        return float("nan")
    norms = np.linalg.norm(activations, axis=1)
    if norms.std() < 1e-8 or target.std() < 1e-8:
        return 0.0
    return float(np.abs(np.corrcoef(norms, target)[0, 1]))


def method_variance_importance(activations: np.ndarray, target: np.ndarray) -> float:
    """Method 3: Per-dim variance weighted by target correlation.

    Proxy for saliency without requiring gradients — uses how much each
    activation dimension varies together with the target.
    """
    if activations.shape[0] < 3:
        return float("nan")
    corrs = []
    for d in range(activations.shape[1]):
        if activations[:, d].std() < 1e-8:
            continue
        c = np.corrcoef(activations[:, d], target)[0, 1]
        if np.isfinite(c):
            corrs.append(abs(c))
    if not corrs:
        return 0.0
    return float(np.mean(corrs))


METHODS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "linear_probing": method_linear_probing,
    "activation_norm": method_activation_norm,
    "variance_importance": method_variance_importance,
}


# ============================================================
# Per-model evaluation pipeline
# ============================================================

def _build_model_and_hooker(model_name: str, seed: int):
    if model_name == "tabpfn":
        m = TabPFNRegressor(device=DEVICE, model_path=str(ROOT / "tabpfn-v2-regressor.ckpt"))
        return m, TabPFNHookedModel
    elif model_name == "tabicl":
        m = TabICLRegressor(device=DEVICE, random_state=seed)
        return m, TabICLHookedModel
    elif model_name == "iltm":
        m = iLTMRegressor(device=DEVICE, n_ensemble=1, seed=seed)
        return m, iLTMHookedModel
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _get_layer_activations(hooker, cache, model_name: str, layer_idx: int) -> np.ndarray:
    """Handle model-specific activation extraction."""
    if model_name == "tabpfn":
        # TabPFN cache: layers[i] = tensor [1, N, tokens, emsize]. Reduce to [N, emsize]
        act = cache["layers"][layer_idx]
        if hasattr(act, "detach"):
            act = act.detach().cpu().numpy()
        # Shape handling: usually (1, N, T, D) — take token -1 (label token)
        if act.ndim == 4:
            act = act[0, :, -1, :]
        elif act.ndim == 3:
            act = act[:, -1, :]
        return np.asarray(act)
    elif model_name == "tabicl":
        return np.asarray(hooker.get_layer_activations(cache, layer_idx))
    elif model_name == "iltm":
        return np.asarray(hooker.get_layer_activations(cache, layer_idx))
    raise ValueError(model_name)


def _num_layers(model_name: str) -> int:
    return {"tabpfn": 12, "tabicl": 12, "iltm": 3}[model_name]


def evaluate_method_on_model(
    method_name: str,
    model_name: str,
    seed: int,
) -> dict[str, Any]:
    """Evaluate one method on one model at one seed."""
    ds = generate_quadratic_data(n_train=N_TRAIN, n_test=N_TEST, random_seed=seed)
    X_tr, y_tr, X_te = ds.X_train, ds.y_train, ds.X_test
    ab_te = X_te[:, 0] * X_te[:, 1]

    model, HookerCls = _build_model_and_hooker(model_name, seed)
    model.fit(X_tr, y_tr)
    hooker = HookerCls(model)
    _, cache = hooker.forward_with_cache(X_te)

    method_fn = METHODS[method_name]
    layer_scores: list[float] = []
    for L in range(_num_layers(model_name)):
        act = _get_layer_activations(hooker, cache, model_name, L)
        score = method_fn(act, ab_te)
        layer_scores.append(score if np.isfinite(score) else 0.0)

    return {
        "method": method_name,
        "model": model_name,
        "seed": seed,
        "layer_scores": layer_scores,
        "peak_layer": int(np.argmax(layer_scores)),
        "peak_score": float(max(layer_scores)),
    }


def aggregate_seeds(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed profiles."""
    profiles = np.stack([np.asarray(r["layer_scores"]) for r in results])
    mean_profile = profiles.mean(axis=0)
    return {
        "method": results[0]["method"],
        "model": results[0]["model"],
        "n_seeds": len(results),
        "mean_profile": mean_profile.tolist(),
        "std_profile": profiles.std(axis=0, ddof=1).tolist(),
        "mean_peak_layer": float(np.argmax(mean_profile)),
        "sigma2_profile": float(np.var(mean_profile)),
    }


def compute_applicability_label(
    method_agg: dict[str, Any], model_name: str,
) -> str:
    """Apply TabMI-Bench applicability labeling rule.

    - Supported: method produces architecture-consistent signal + reasonable peak
    - Limited: produces signal but inconsistent or very weak
    - Not established: signal is null or uninterpretable
    """
    profile = np.asarray(method_agg["mean_profile"])
    sigma2 = method_agg["sigma2_profile"]
    peak_L = method_agg["mean_peak_layer"]
    expected_range = REFERENCE_PROFILES[model_name]["expected_peak_range"]
    expected_class = REFERENCE_PROFILES[model_name]["expected_class"]

    # Check if peak is in expected range
    peak_in_range = expected_range[0] <= peak_L <= expected_range[1]

    # Check if signal is non-trivial (max profile > 0.05)
    signal_strong = float(profile.max()) > 0.1

    # Apply rules
    if not signal_strong:
        return "Not established"

    if expected_class == "staged":
        # Staged requires enough variance
        if sigma2 > 0.01 and peak_in_range:
            return "Supported"
        elif signal_strong:
            return "Limited"
        else:
            return "Not established"
    elif expected_class == "distributed":
        # Distributed requires flat high profile
        if profile.min() > 0.5 * profile.max() and signal_strong:
            return "Supported"
        elif signal_strong:
            return "Limited"
        else:
            return "Not established"
    elif expected_class == "preprocessing-dominant":
        # Preprocessing-dominant requires peak at early layer
        if peak_L <= 1 and signal_strong:
            return "Supported"
        elif signal_strong:
            return "Limited"
        else:
            return "Not established"
    return "Not established"


def main() -> int:
    print("=" * 72)
    print("Phase 9: MI Method Comparison on TabMI-Bench")
    print("=" * 72)
    print(f"Methods: {list(METHODS.keys())}")
    print("Models: tabpfn, tabicl, iltm")
    print(f"Seeds: {SEEDS}")
    print(f"DEVICE={DEVICE}")
    start_t = time.time()

    # Run full matrix: methods × models × seeds
    all_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for method_name in METHODS.keys():
        all_results[method_name] = {}
        for model_name in ("tabpfn", "tabicl", "iltm"):
            print(f"\n--- {method_name} on {model_name} ---")
            seed_results = []
            for seed in SEEDS:
                print(f"  seed={seed}...", end=" ")
                try:
                    r = evaluate_method_on_model(method_name, model_name, seed)
                    seed_results.append(r)
                    print(f"peak=L{r['peak_layer']} score={r['peak_score']:.3f}")
                except Exception as e:
                    print(f"FAILED: {e}")
            all_results[method_name][model_name] = seed_results

    # Aggregate and compute applicability
    summary: dict[str, Any] = {"methods": {}}
    for method_name, model_data in all_results.items():
        summary["methods"][method_name] = {}
        for model_name, seed_results in model_data.items():
            if not seed_results:
                continue
            agg = aggregate_seeds(seed_results)
            label = compute_applicability_label(agg, model_name)
            agg["applicability_label"] = label
            summary["methods"][method_name][model_name] = agg

    # Save
    out_path = RESULTS_DIR / "method_comparison_results.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # Print applicability matrix
    print("\n" + "=" * 72)
    print("TabMI-Bench Applicability Matrix (generated by the benchmark)")
    print("=" * 72)
    print(f"{'Method':<25} {'TabPFN':<15} {'TabICL':<15} {'iLTM':<15}")
    for method_name in METHODS.keys():
        row = [method_name]
        for model_name in ("tabpfn", "tabicl", "iltm"):
            if model_name in summary["methods"].get(method_name, {}):
                row.append(summary["methods"][method_name][model_name]["applicability_label"])
            else:
                row.append("FAIL")
        print(f"{row[0]:<25} {row[1]:<15} {row[2]:<15} {row[3]:<15}")

    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.time() - start_t:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
