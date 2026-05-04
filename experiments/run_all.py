#!/usr/bin/env python
"""Run the complete TFMI experiment pipeline: full-scale validation, aggregation,
figure generation, and statistical analysis.

Usage:
    # Full pipeline (Phase 5 + Phase 6 + aggregation + figures + stats)
    QUICK_RUN=0 PYTHONPATH=. .venv/bin/python experiments/run_all.py

    # Figures and stats only (no experiments — uses existing results)
    PYTHONPATH=. .venv/bin/python experiments/run_all.py --postprocess-only

    # Quick-run mode (smoke test with minimal data)
    QUICK_RUN=1 PYTHONPATH=. .venv/bin/python experiments/run_all.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")
EXPERIMENTS = ROOT / "experiments"


def _run(script: str, *, env_extra: dict[str, str] | None = None) -> bool:
    """Run a Python script and return True if it succeeds."""
    cmd = [PYTHON, str(EXPERIMENTS / script)]
    import os

    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
    if env_extra:
        env.update(env_extra)

    print(f"\n{'=' * 70}")
    print(f"  Running: {script}")
    print(f"{'=' * 70}")
    t0 = time.time()
    result = subprocess.run(cmd, env=env, cwd=str(ROOT))
    elapsed = time.time() - t0
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"  [{status}] {script} — {elapsed:.1f}s")
    return result.returncode == 0


def run_experiments() -> dict[str, bool]:
    """Run Phase 5 and Phase 6 full-scale experiments."""
    results: dict[str, bool] = {}

    # Phase 5: 5-seed × 9 experiments
    results["Phase 5 (5-seed)"] = _run(
        "run_fullscale.py",
        env_extra={"QUICK_RUN": "0"},
    )

    results["Phase 6 (default 5-seed)"] = _run(
        "run_fullscale_phase6.py",
        env_extra={"QUICK_RUN": "0"},
    )

    # v2.5 comparison (individual run — not in Phase 6 runner)
    results["TabPFN v2.5"] = _run(
        "rd6_tabpfn25_comparison.py",
        env_extra={"QUICK_RUN": "0"},
    )

    # SAE TopK comparison
    results["SAE TopK"] = _run(
        "rd6_sae_topk_comparison.py",
        env_extra={"QUICK_RUN": "0"},
    )

    # TabICL L5-L6 zoom
    results["TabICL L5-L6 Zoom"] = _run(
        "rd6_tabicl_l5l6_zoom.py",
        env_extra={"QUICK_RUN": "0"},
    )

    return results


def run_aggregation() -> dict[str, bool]:
    """Run aggregation scripts."""
    results: dict[str, bool] = {}
    results["Phase 5 Aggregation"] = _run("aggregate_results.py")
    results["Phase 6 Aggregation"] = _run("aggregate_phase6.py")
    results["SAE TopK Aggregation"] = _run("aggregate_sae_topk_fullscale.py")
    return results


def run_postprocessing() -> dict[str, bool]:
    """Run figure generation and statistical analysis."""
    results: dict[str, bool] = {}
    results["Paper Figures"] = _run("generate_paper_figures.py")
    results["Statistical Analysis"] = _run("statistical_analysis.py")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="TFMI full pipeline runner")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Skip experiments, run aggregation + figures + stats only",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Generate figures and stats only (no experiments, no aggregation)",
    )
    args = parser.parse_args()

    t_start = time.time()
    all_results: dict[str, bool] = {}

    if args.figures_only:
        all_results.update(run_postprocessing())
    elif args.postprocess_only:
        all_results.update(run_aggregation())
        all_results.update(run_postprocessing())
    else:
        all_results.update(run_experiments())
        all_results.update(run_aggregation())
        all_results.update(run_postprocessing())

    elapsed = time.time() - t_start

    # Summary
    print(f"\n{'=' * 70}")
    print("  PIPELINE SUMMARY")
    print(f"{'=' * 70}")
    passed = sum(1 for v in all_results.values() if v)
    total = len(all_results)
    for name, ok in all_results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")
    print(f"\n  {passed}/{total} passed — {elapsed:.1f}s total")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
