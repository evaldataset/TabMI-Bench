# pyright: reportMissingImports=false
"""Phase 8B multi-seed wrapper — runs real-world steering across 5 seeds.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python experiments/phase8b_multiseed_runner.py

This wrapper invokes phase8b_realworld_steering.py once per seed with
SEED env variable, then aggregates all results into a single summary.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [42, 123, 456, 789, 1024]
RESULTS_DIR = ROOT / "results" / "phase8b" / "realworld_steering"
AGG_PATH = RESULTS_DIR / "aggregated_5seed.json"


def run_single_seed(seed: int) -> bool:
    """Run phase8b_realworld_steering.py with given seed. Returns True on success."""
    env = {**os.environ, "SEED": str(seed), "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
    cmd = [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "experiments" / "phase8b_realworld_steering.py")]
    print(f"\n{'=' * 60}\nSeed {seed} — launching...\n{'=' * 60}")
    t0 = time.time()
    result = subprocess.run(cmd, env=env, cwd=str(ROOT))
    elapsed = time.time() - t0
    ok = result.returncode == 0
    print(f"[{'OK' if ok else 'FAIL'}] seed={seed} ({elapsed:.1f}s)")
    return ok


def aggregate_seeds() -> dict[str, Any]:
    """Aggregate per-seed results into mean ± std."""
    per_seed_data: list[dict[str, Any]] = []
    for seed in SEEDS:
        path = RESULTS_DIR / f"results_seed{seed}.json"
        if not path.exists():
            print(f"WARN: missing {path}")
            continue
        with path.open() as f:
            per_seed_data.append(json.load(f))

    if not per_seed_data:
        return {"error": "No seed results found"}

    # Structure: per_seed_data[i] = {dataset_name: {model_name: {pearson_r, slope, ...}}}
    datasets = list(per_seed_data[0].keys())
    summary: dict[str, Any] = {"seeds": SEEDS, "datasets": {}}

    for ds in datasets:
        summary["datasets"][ds] = {}
        for model in ("tabpfn", "tabicl"):
            r_values = []
            slope_values = []
            for seed_data in per_seed_data:
                if ds not in seed_data or model not in seed_data[ds]:
                    continue
                r_values.append(abs(float(seed_data[ds][model].get("pearson_r", 0.0))))
                slope_values.append(float(seed_data[ds][model].get("slope", 0.0)))
            if r_values:
                summary["datasets"][ds][model] = {
                    "n_seeds": len(r_values),
                    "abs_pearson_r_mean": float(np.mean(r_values)),
                    "abs_pearson_r_std": float(np.std(r_values, ddof=1) if len(r_values) > 1 else 0.0),
                    "slope_mean": float(np.mean(slope_values)),
                    "slope_std": float(np.std(slope_values, ddof=1) if len(slope_values) > 1 else 0.0),
                    "r_values": r_values,
                }
    return summary


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    start_t = time.time()
    print(f"Multi-seed runner: {len(SEEDS)} seeds = {SEEDS}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")

    n_ok = 0
    for seed in SEEDS:
        if run_single_seed(seed):
            n_ok += 1

    print(f"\n{'=' * 60}\nAggregating {n_ok}/{len(SEEDS)} successful seeds\n{'=' * 60}")
    summary = aggregate_seeds()
    with AGG_PATH.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {AGG_PATH}")

    # Print summary table
    print("\n=== 5-seed Real-world Steering Summary ===")
    print(f"{'Dataset':<20} {'Model':<10} {'|r| mean±std':<20} {'n_seeds'}")
    if "datasets" in summary:
        for ds, ds_data in summary["datasets"].items():
            for model, m_data in ds_data.items():
                r_mean = m_data["abs_pearson_r_mean"]
                r_std = m_data["abs_pearson_r_std"]
                n = m_data["n_seeds"]
                print(f"{ds:<20} {model:<10} {r_mean:.3f}±{r_std:.3f}{'':<8} {n}")

    print(f"\nTotal time: {time.time() - start_t:.1f}s")
    return 0 if n_ok == len(SEEDS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
