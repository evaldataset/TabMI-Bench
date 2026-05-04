#!/usr/bin/env python3
"""Multi-seed full-scale runner for Phase 6 experiments.

Runs all rd6_* experiment scripts across multiple random seeds,
saving results in seed-specific subdirectories.

Usage:
    # Full-scale, 5 seeds (default)
    .venv/bin/python experiments/run_fullscale_phase6.py

    # Quick smoke test (1 seed, quick mode)
    .venv/bin/python experiments/run_fullscale_phase6.py --quick

    # Specific seeds
    .venv/bin/python experiments/run_fullscale_phase6.py --seeds 42 123 456

    # Only run specific experiments
    .venv/bin/python experiments/run_fullscale_phase6.py --only robust_steering improved_sae
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_BASE = ROOT / "results" / "rd6"
FULLSCALE_DIR = ROOT / "results" / "rd6_fullscale"

# Phase 6 experiment scripts in execution order (fast → slow)
EXPERIMENT_SCRIPTS: dict[str, str] = {
    "robust_steering": "rd6_robust_steering.py",
    "improved_sae": "rd6_improved_sae.py",
    "realworld_expanded": "rd6_realworld_expanded.py",
    "classification_probing": "rd6_classification_probing.py",
    "attention_comparison": "rd6_attention_comparison.py",
}

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
PYTHON = str(ROOT / ".venv" / "bin" / "python")


def run_experiment(
    script_name: str,
    seed: int,
    quick_run: bool,
    n_train: int,
    n_test: int,
) -> dict[str, object]:
    """Run a single experiment script with given configuration."""
    script_path = EXPERIMENTS_DIR / script_name
    if not script_path.exists():
        return {
            "success": False,
            "duration": 0.0,
            "returncode": -1,
            "error": f"Script not found: {script_path}",
        }

    env = os.environ.copy()
    env["QUICK_RUN"] = "1" if quick_run else "0"
    env["SEED"] = str(seed)
    env["N_TRAIN"] = str(n_train)
    env["N_TEST"] = str(n_test)
    env["PYTHONUNBUFFERED"] = "1"

    log_dir = FULLSCALE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_stem = f"seed{seed}_{script_name.replace('.py', '')}"
    log_path = log_dir / f"{log_stem}.log"

    timeout_s = 7200  # 2 hours per experiment (SAE/steering can be slow)
    start = time.time()
    try:
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                [PYTHON, str(script_path)],
                cwd=str(ROOT),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            try:
                returncode = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                duration = time.time() - start
                return {
                    "success": False,
                    "duration": round(duration, 1),
                    "returncode": -2,
                    "error": f"Timeout ({timeout_s}s)",
                    "log": str(log_path),
                }

        duration = time.time() - start
        try:
            log_text = log_path.read_text(encoding="utf-8")
            stdout_tail = log_text[-500:] if log_text else ""
        except Exception:
            stdout_tail = ""

        return {
            "success": returncode == 0,
            "duration": round(duration, 1),
            "returncode": returncode,
            "stdout_tail": stdout_tail,
            "log": str(log_path),
        }
    except Exception as e:
        duration = time.time() - start
        return {
            "success": False,
            "duration": round(duration, 1),
            "returncode": -3,
            "error": str(e),
        }


def relocate_results(experiment_key: str, seed: int) -> None:
    """Copy results from default rd6 location to seed-specific subdirectory.

    Default: results/rd6/<experiment>/results.json
    Target:  results/rd6_fullscale/seed_<seed>/<experiment>/results.json
    """
    src_dir = RESULTS_BASE / experiment_key
    dst_dir = FULLSCALE_DIR / f"seed_{seed}" / experiment_key
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        return

    for f in src_dir.iterdir():
        if f.is_file():
            dst_file = dst_dir / f.name
            dst_file.write_bytes(f.read_bytes())


def run_seed(
    seed: int,
    experiments: dict[str, str],
    quick_run: bool,
    n_train: int,
    n_test: int,
) -> dict[str, dict[str, object]]:
    """Run all experiments for a single seed."""
    seed_results: dict[str, dict[str, object]] = {}

    for exp_key, script_name in experiments.items():
        print(f"\n{'─' * 60}")
        print(f"  Seed={seed} | Experiment: {exp_key} ({script_name})")
        print(f"{'─' * 60}")

        result = run_experiment(
            script_name=script_name,
            seed=seed,
            quick_run=quick_run,
            n_train=n_train,
            n_test=n_test,
        )
        seed_results[exp_key] = result

        status = "✅" if result["success"] else "❌"
        print(
            f"  {status} {exp_key}: {result['duration']}s (rc={result['returncode']})"
        )

        if result["success"]:
            relocate_results(exp_key, seed)

    return seed_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 Multi-seed Full-scale Runner")
    parser.add_argument(
        "--quick", action="store_true", help="Quick smoke test (QUICK_RUN=1)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=f"Seeds to run (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="Experiment keys to skip",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Only run these experiment keys",
    )
    parser.add_argument("--n-train", type=int, default=None, help="Override N_TRAIN")
    parser.add_argument("--n-test", type=int, default=None, help="Override N_TEST")
    args = parser.parse_args()

    seeds = args.seeds or DEFAULT_SEEDS
    quick_run = args.quick
    n_train = args.n_train or (50 if quick_run else 100)
    n_test = args.n_test or (10 if quick_run else 50)

    # Build experiment list
    experiments = dict(EXPERIMENT_SCRIPTS)
    if args.only:
        experiments = {k: v for k, v in experiments.items() if k in args.only}
    for skip_key in args.skip:
        experiments.pop(skip_key, None)

    mode = "QUICK" if quick_run else "FULL-SCALE"
    print("=" * 70)
    print(f"  Phase 6 Multi-Seed Runner ({mode})")
    print("=" * 70)
    print(f"  Seeds: {seeds}")
    print(f"  N_TRAIN={n_train}, N_TEST={n_test}")
    print(f"  Experiments ({len(experiments)}): {list(experiments.keys())}")
    print(
        f"  Total runs: {len(seeds)} seeds × {len(experiments)} experiments = {len(seeds) * len(experiments)}"
    )
    print("=" * 70)

    all_results: dict[str, dict[str, dict[str, object]]] = {}
    overall_start = time.time()

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n{'═' * 70}")
        print(f"  SEED {seed_idx}/{len(seeds)}: {seed}")
        print(f"{'═' * 70}")

        seed_results = run_seed(
            seed=seed,
            experiments=experiments,
            quick_run=quick_run,
            n_train=n_train,
            n_test=n_test,
        )
        all_results[str(seed)] = seed_results

    overall_duration = time.time() - overall_start

    # Save run metadata
    FULLSCALE_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = FULLSCALE_DIR / "run_metadata.json"

    meta = {
        "mode": mode,
        "seeds": seeds,
        "n_train": n_train,
        "n_test": n_test,
        "experiments": list(experiments.keys()),
        "total_duration_s": round(overall_duration, 1),
        "results": all_results,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY ({mode})")
    print(f"{'=' * 70}")
    print(f"  Total time: {overall_duration:.0f}s ({overall_duration / 60:.1f}m)")

    n_success = sum(
        1
        for seed_res in all_results.values()
        for exp_res in seed_res.values()
        if exp_res.get("success")
    )
    n_total = sum(len(seed_res) for seed_res in all_results.values())
    print(f"  Success: {n_success}/{n_total}")
    print(f"  Results: {FULLSCALE_DIR}")
    print(f"  Metadata: {meta_path}")

    return 0 if n_success == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
