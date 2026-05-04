# pyright: reportMissingImports=false
"""Phase 8C: SAE comparison with 10 seeds.

Runs the phase7 SAE scaling experiment across 10 seeds to escape
the Wilcoxon n=5 p=0.0625 saturation issue.
With n=10, minimum Wilcoxon p = 0.00195 (survives Bonferroni).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
DEVICE = os.environ.get("DEVICE", "cuda:0")


def main() -> int:
    print("=" * 72)
    print("Phase 8C: SAE Multi-Seed (10 seeds)")
    print("=" * 72)
    print(f"Seeds: {SEEDS}")
    print(f"Device: {DEVICE}")

    for i, seed in enumerate(SEEDS):
        result_path = ROOT / "results" / "phase7" / f"sae_scaling_seed{seed}.json"
        if result_path.exists():
            print(f"\n[{i+1}/{len(SEEDS)}] seed={seed}: SKIP (already exists)")
            continue

        print(f"\n[{i+1}/{len(SEEDS)}] seed={seed}: running...")
        env = os.environ.copy()
        env["SEED"] = str(seed)
        env["DEVICE"] = DEVICE
        env["QUICK_RUN"] = "0"

        result = subprocess.run(
            [sys.executable, str(ROOT / "experiments" / "phase7_sae_scaling.py")],
            env=env,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})")
            print(f"  stderr: {result.stderr[-500:]}")
        else:
            print(f"  OK -> {result_path}")

    # Summary
    existing = list((ROOT / "results" / "phase7").glob("sae_scaling_seed*.json"))
    print(f"\n{'='*72}")
    print(f"Total SAE seed results: {len(existing)}")
    for p in sorted(existing):
        print(f"  {p.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
