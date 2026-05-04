# pyright: reportMissingImports=false
"""Phase 9: Out-of-family holdout validation using NAM.

NAM (Neural Additive Model) is architecturally distinct from:
- Transformer-based TFMs (TabPFN, TabICL, TabDPT): NAM has NO attention
- Tree-based TFMs (iLTM): NAM has NO decision trees or PCA
- NAM uses feature-wise MLPs summed additively

This is a genuine out-of-family holdout. We apply the TabMI-Bench
operational decision rules (from §4.1) blind, and record:
  - σ²_profile
  - Mid/Early ratio
  - Classification: staged / distributed / preprocessing-dominant

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python experiments/phase9_nam_holdout.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.synthetic_generator import generate_quadratic_data
from src.hooks.nam_hooker import NAMHookedModel, NAMRegressor

SEEDS = [42, 123, 456, 789, 1024]
N_TRAIN = 100
N_TEST = 50
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = ROOT / "results" / "phase9_nam_holdout"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def probe_intermediary(activations: np.ndarray, target: np.ndarray) -> float:
    """Linear Ridge probe: predict target from activations. Returns R²."""
    if activations.shape[0] < 5 or activations.shape[1] < 1:
        return float("nan")
    from sklearn.model_selection import KFold

    # 5-fold cross-validated R²
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros_like(target, dtype=np.float64)
    for train_idx, test_idx in kf.split(activations):
        clf = Ridge(alpha=1.0)
        clf.fit(activations[train_idx], target[train_idx])
        preds[test_idx] = clf.predict(activations[test_idx])
    return float(r2_score(target, preds))


def run_seed(seed: int) -> dict[str, Any]:
    """Run NAM intermediary probing for one seed."""
    print(f"\n--- Seed {seed} ---")

    # Generate bilinear data: z = a*b + c (matching main experiments)
    ds = generate_quadratic_data(
        n_train=N_TRAIN,
        n_test=N_TEST,
        random_seed=seed,
    )
    X_train = ds.X_train
    y_train = ds.y_train
    X_test = ds.X_test
    # Intermediary is a*b (first two features)
    ab_test = X_test[:, 0] * X_test[:, 1]

    # Fit NAM
    model = NAMRegressor(device=DEVICE, n_epochs=300, random_state=seed)
    model.fit(X_train, y_train)

    # Extract layer activations
    hooker = NAMHookedModel(model)
    _, cache = hooker.forward_with_cache(X_test)

    # Probe intermediary at each layer
    layer_r2: list[float] = []
    for layer_idx in range(hooker.num_layers):
        acts = hooker.get_layer_activations(cache, layer_idx)
        r2 = probe_intermediary(acts, ab_test)
        layer_r2.append(r2)
        print(f"  L{layer_idx}: R²={r2:.3f}  shape={acts.shape}")

    return {
        "seed": seed,
        "layer_r2": layer_r2,
        "peak_layer": int(np.argmax(layer_r2)),
        "peak_r2": float(max(layer_r2)),
    }


def classify_profile(profile: list[float]) -> dict[str, Any]:
    """Apply TabMI-Bench operational decision rules (from §4.1).

    - staged: >0.15 absolute variation, non-monotone U-shape
    - distributed: R²>0.90 at every layer, <0.08 absolute variation
    - preprocessing-dominant: peaks at L0, strictly decreasing
    """
    arr = np.asarray(profile)
    abs_var = float(arr.max() - arr.min())
    min_r2 = float(arr.min())
    peak_at_L0 = bool(arr.argmax() == 0)
    strictly_decreasing = bool(all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1)))

    # Apply decision rules in order
    if min_r2 > 0.90 and abs_var < 0.08:
        label = "distributed"
    elif peak_at_L0 and strictly_decreasing:
        label = "preprocessing-dominant"
    elif abs_var > 0.15:
        label = "staged"
    else:
        label = "indeterminate"

    return {
        "sigma2_profile": float(np.var(arr)),
        "abs_variation": abs_var,
        "min_r2": min_r2,
        "max_r2": float(arr.max()),
        "mid_early_ratio": float(arr[len(arr) // 2] / arr[0]) if arr[0] > 0 else float("nan"),
        "peak_at_L0": peak_at_L0,
        "strictly_decreasing": strictly_decreasing,
        "classification": label,
    }


def main() -> int:
    print("=" * 72)
    print("Phase 9: NAM Out-of-Family Holdout Validation")
    print("=" * 72)
    print(f"DEVICE={DEVICE}, SEEDS={SEEDS}")
    start_t = time.time()

    per_seed = []
    for seed in SEEDS:
        per_seed.append(run_seed(seed))

    # Aggregate across seeds
    profiles_matrix = np.stack([np.asarray(r["layer_r2"]) for r in per_seed])
    mean_profile = profiles_matrix.mean(axis=0).tolist()
    std_profile = profiles_matrix.std(axis=0, ddof=1).tolist()

    classification = classify_profile(mean_profile)

    summary = {
        "model": "NAM",
        "architecture": "feature-wise MLPs, additive (no attention, no trees)",
        "n_layers": len(mean_profile),
        "seeds": SEEDS,
        "per_seed": per_seed,
        "mean_profile": mean_profile,
        "std_profile": std_profile,
        "classification": classification,
        "runtime_s": time.time() - start_t,
    }

    out_path = RESULTS_DIR / "nam_holdout_results.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("NAM Holdout — Classification Result")
    print("=" * 72)
    print(f"Mean profile: {[f'{v:.3f}' for v in mean_profile]}")
    print(f"σ²_profile = {classification['sigma2_profile']:.4f}")
    print(f"Abs variation = {classification['abs_variation']:.3f}")
    print(f"Mid/Early ratio = {classification['mid_early_ratio']:.3f}")
    print(f"Peak at L0 = {classification['peak_at_L0']}")
    print(f"Strictly decreasing = {classification['strictly_decreasing']}")
    print(f"\n*** NAM classified as: {classification['classification'].upper()} ***")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
