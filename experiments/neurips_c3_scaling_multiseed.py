#!/usr/bin/env python3
"""Multi-seed wrapper for N=10K scaling experiments (intermediary probing, patching, steering).

Runs rd5_intermediary_probing.py, rd5_patching_comparison.py, and rd5_steering_comparison.py
with N_TRAIN=10000, N_TEST=1000 across 3 seeds (42, 123, 456).

Aggregates results (mean ± std) and saves to results/neurips/c3_scaling_multiseed.json.

Usage:
    DEVICE=cuda:0 .venv/bin/python experiments/neurips_c3_scaling_multiseed.py [--dry-run]

Environment variables:
    DEVICE: PyTorch device (default: cpu)
    QUICK_RUN: Set to 0 for full-scale (default: 0 for this script)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = ROOT / "results"
SCALING_DIR = RESULTS_BASE / "scaling_10k"
NEURIPS_DIR = RESULTS_BASE / "neurips"

# Experiments to run
EXPERIMENTS = [
    (
        "rd5_intermediary_probing.py",
        "intermediary_probing.json",
        "results/rd5/intermediary_probing/results.json",
    ),
    (
        "rd5_patching_comparison.py",
        "patching.json",
        "results/rd5/patching/results.json",
    ),
    (
        "rd5_steering_comparison.py",
        "steering.json",
        "results/rd5/steering/results.json",
    ),
]

SEEDS = [42, 123, 456, 789, 1024]
N_TRAIN = 10000
N_TEST = 1000


def _run_experiment(
    script_name: str,
    seed: int,
    device: str,
    dry_run: bool = False,
) -> bool:
    """Run a single experiment with given seed.

    Args:
        script_name: Name of the experiment script (e.g., "rd5_intermediary_probing.py")
        seed: Random seed
        device: PyTorch device
        dry_run: If True, print command without executing

    Returns:
        True if successful, False otherwise
    """
    env = os.environ.copy()
    env["QUICK_RUN"] = "0"
    env["DEVICE"] = device
    env["N_TRAIN"] = str(N_TRAIN)
    env["N_TEST"] = str(N_TEST)
    env["SEED"] = str(seed)

    script_path = ROOT / "experiments" / script_name
    cmd = [".venv/bin/python", str(script_path)]

    print(f"\n{'=' * 70}")
    print(f"Running: {script_name} (seed={seed})")
    print(f"Command: {' '.join(cmd)}")
    print(
        f"Env: QUICK_RUN=0, SEED={seed}, N_TRAIN={N_TRAIN}, N_TEST={N_TEST}, DEVICE={device}"
    )
    print(f"{'=' * 70}")

    if dry_run:
        print("[DRY RUN] Would execute above command")
        return True

    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(ROOT),
            check=True,
            capture_output=False,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {script_name} failed with return code {e.returncode}")
        return False


def _copy_results(
    seed: int,
    dry_run: bool = False,
) -> bool:
    """Copy results from rd5 directories to scaling_10k/seed_{seed}/.

    Args:
        seed: Random seed
        dry_run: If True, print actions without executing

    Returns:
        True if all copies successful, False otherwise
    """
    seed_dir = SCALING_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for script_name, target_name, source_rel_path in EXPERIMENTS:
        source_path = ROOT / source_rel_path
        target_path = seed_dir / target_name

        if not source_path.exists():
            print(f"WARNING: Source not found: {source_path}")
            all_ok = False
            continue

        print(f"  Copying {source_path.name} -> {target_path.name}")

        if not dry_run:
            shutil.copy2(source_path, target_path)

    return all_ok


def _aggregate_results(dry_run: bool = False) -> dict[str, Any]:
    """Load per-seed results and aggregate (mean ± std).

    Args:
        dry_run: If True, don't save aggregated results

    Returns:
        Aggregated results dictionary
    """
    print(f"\n{'=' * 70}")
    print("Aggregating results across seeds...")
    print(f"{'=' * 70}")

    per_seed_results: dict[int, dict[str, Any]] = {}

    for seed in SEEDS:
        seed_dir = SCALING_DIR / f"seed_{seed}"
        seed_results: dict[str, Any] = {}

        for script_name, target_name, _ in EXPERIMENTS:
            json_path = seed_dir / target_name

            if not json_path.exists():
                print(f"WARNING: Missing {json_path}")
                continue

            with json_path.open("r", encoding="utf-8") as f:
                seed_results[target_name.replace(".json", "")] = json.load(f)

        per_seed_results[seed] = seed_results

    aggregated: dict[str, Any] = {
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "per_seed": per_seed_results,
        "aggregated": {},
    }

    seeds_with_data = [
        s for s in SEEDS if s in per_seed_results and per_seed_results[s]
    ]
    if not seeds_with_data:
        print("WARNING: No seed results found to aggregate")
        return aggregated

    first_seed = seeds_with_data[0]

    if "intermediary_probing" in per_seed_results[first_seed]:
        agg_intermediary = _aggregate_probing_results(
            per_seed_results,
            "intermediary_probing",
        )
        aggregated["aggregated"]["intermediary_probing"] = agg_intermediary

    if "patching" in per_seed_results[first_seed]:
        agg_patching = _aggregate_patching_results(
            per_seed_results,
            "patching",
        )
        aggregated["aggregated"]["patching"] = agg_patching

    if "steering" in per_seed_results[first_seed]:
        agg_steering = _aggregate_steering_results(
            per_seed_results,
            "steering",
        )
        aggregated["aggregated"]["steering"] = agg_steering

    # Save aggregated results
    NEURIPS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = NEURIPS_DIR / "c3_scaling_multiseed.json"

    if not dry_run:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nSaved aggregated results: {output_path}")
    else:
        print(f"[DRY RUN] Would save aggregated results to: {output_path}")

    return aggregated


def _aggregate_probing_results(
    per_seed_results: dict[int, dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    """Aggregate probing results (intermediary_r2_by_layer, final_r2_by_layer)."""
    models = ["tabpfn", "tabicl", "iltm"]
    aggregated: dict[str, Any] = {}

    for model in models:
        intermediary_arrays = []
        final_arrays = []

        for seed in SEEDS:
            if seed not in per_seed_results or key not in per_seed_results[seed]:
                continue
            if model not in per_seed_results[seed][key]:
                continue

            model_data = per_seed_results[seed][key][model]
            intermediary_arrays.append(
                np.array(model_data["intermediary_r2_by_layer"], dtype=np.float32)
            )
            final_arrays.append(
                np.array(model_data["final_r2_by_layer"], dtype=np.float32)
            )

        if intermediary_arrays:
            intermediary_stack = np.stack(intermediary_arrays, axis=0)
            final_stack = np.stack(final_arrays, axis=0)

            aggregated[model] = {
                "intermediary_r2_by_layer": {
                    "mean": intermediary_stack.mean(axis=0).tolist(),
                    "std": intermediary_stack.std(axis=0).tolist(),
                },
                "final_r2_by_layer": {
                    "mean": final_stack.mean(axis=0).tolist(),
                    "std": final_stack.std(axis=0).tolist(),
                },
                "peak_layer_intermediary": {
                    "mean": float(intermediary_stack.mean(axis=0).argmax()),
                    "std": float(intermediary_stack.argmax(axis=1).std()),
                },
                "peak_layer_final": {
                    "mean": float(final_stack.mean(axis=0).argmax()),
                    "std": float(final_stack.argmax(axis=1).std()),
                },
            }

    return aggregated


def _aggregate_patching_results(
    per_seed_results: dict[int, dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    """Aggregate patching results (summary_sensitivity by layer)."""
    models = ["tabpfn", "tabicl"]
    aggregated: dict[str, Any] = {}

    for model in models:
        sensitivity_arrays = []

        for seed in SEEDS:
            if seed not in per_seed_results or key not in per_seed_results[seed]:
                continue
            if model not in per_seed_results[seed][key]:
                continue

            model_data = per_seed_results[seed][key][model]
            sensitivity_arrays.append(
                np.array(model_data["summary_sensitivity"], dtype=np.float32)
            )

        if sensitivity_arrays:
            sensitivity_stack = np.stack(sensitivity_arrays, axis=0)

            aggregated[model] = {
                "summary_sensitivity": {
                    "mean": sensitivity_stack.mean(axis=0).tolist(),
                    "std": sensitivity_stack.std(axis=0).tolist(),
                },
                "most_sensitive_layer": {
                    "mean": float(sensitivity_stack.mean(axis=0).argmax()),
                    "std": float(sensitivity_stack.argmax(axis=1).std()),
                },
            }

    return aggregated


def _aggregate_steering_results(
    per_seed_results: dict[int, dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    """Aggregate steering results (pearson_r, slope)."""
    models = ["tabpfn", "tabicl"]
    aggregated: dict[str, Any] = {}

    for model in models:
        pearson_r_values = []
        slope_values = []

        for seed in SEEDS:
            if seed not in per_seed_results or key not in per_seed_results[seed]:
                continue
            if model not in per_seed_results[seed][key]:
                continue

            model_data = per_seed_results[seed][key][model]
            if "effect" in model_data:
                pearson_r_values.append(model_data["effect"]["pearson_r"])
                slope_values.append(model_data["effect"]["slope"])

        if pearson_r_values:
            pearson_r_array = np.array(pearson_r_values, dtype=np.float32)
            slope_array = np.array(slope_values, dtype=np.float32)

            aggregated[model] = {
                "pearson_r": {
                    "mean": float(pearson_r_array.mean()),
                    "std": float(pearson_r_array.std()),
                },
                "slope": {
                    "mean": float(slope_array.mean()),
                    "std": float(slope_array.std()),
                },
            }

    return aggregated


def main() -> int:
    """Main entry point."""
    dry_run = "--dry-run" in sys.argv
    device = os.environ.get("DEVICE", "cpu").strip()

    print(f"\n{'=' * 70}")
    print("NeurIPS C3: Multi-seed N=10K Scaling Experiments")
    print(f"{'=' * 70}")
    print(f"Seeds: {SEEDS}")
    print(f"N_TRAIN: {N_TRAIN}, N_TEST: {N_TEST}")
    print(f"Device: {device}")
    print(f"Dry run: {dry_run}")
    print(f"{'=' * 70}\n")

    # Run experiments for each seed
    for seed in SEEDS:
        print(f"\n{'#' * 70}")
        print(f"# SEED {seed}")
        print(f"{'#' * 70}")

        for script_name, _, _ in EXPERIMENTS:
            success = _run_experiment(script_name, seed, device, dry_run)
            if not success and not dry_run:
                print(f"ERROR: Failed to run {script_name} for seed {seed}")
                return 1

        # Copy results to scaling_10k/seed_{seed}/
        print(f"\nCopying results to {SCALING_DIR}/seed_{seed}/...")
        if not _copy_results(seed, dry_run):
            print(f"WARNING: Some results may not have been copied for seed {seed}")

    # Aggregate results
    aggregated = _aggregate_results(dry_run)

    print(f"\n{'=' * 70}")
    print("Summary:")
    print(f"  Seeds processed: {SEEDS}")
    print(f"  Output: {NEURIPS_DIR / 'c3_scaling_multiseed.json'}")
    print(f"{'=' * 70}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
