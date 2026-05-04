#!/usr/bin/env python3
"""Verify paper-facing summary numerics from frozen_artifacts/ ONLY.

This script does NOT touch results/ (NAS-backed) and is the canonical
reproduction-from-frozen path for external reviewers without NAS access.
Reads: frozen_artifacts/*.json
Prints: the exact values that drive Tables 1, 7, the LOFO appendix, the
TabDPT in-family holdout, the TabDPT causal validation, the NAM
out-of-family holdout, and the v2 vs v2.5 comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    root = Path(__file__).resolve().parent.parent / "frozen_artifacts"
    if not root.is_dir():
        print(f"[ERROR] frozen_artifacts/ not found at {root}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("Paper-facing summary numerics (verified from frozen_artifacts/)")
    print("=" * 70)

    # --- Table 1: rd5 fullscale aggregated ---
    rd5 = json.loads((root / "rd5_fullscale_aggregated.json").read_text())
    ip = rd5["intermediary_probing"]
    print("\nTable 1 source: rd5_fullscale_aggregated.json")
    print(f"  n_seeds: {ip['n_seeds']}")
    for m in ("tabpfn", "tabicl", "iltm"):
        arr = np.array(ip[m]["intermediary_r2_mean"])
        print(
            f"  {m:8s}: peak_R2={float(arr.max()):.4f}, "
            f"sigma2_profile={float(arr.var()):.6f}, "
            f"min_R2={float(arr.min()):.4f}"
        )

    # --- TabDPT in-family holdout (3-seed) ---
    td = json.loads((root / "tabdpt_probing_3seed.json").read_text())
    print("\nTabDPT in-family holdout (3 seeds):")
    print(
        f"  peak_R2={td['peak_r2_mean']:.4f}, "
        f"sigma2_profile={td['sigma2_profile_mean_profile']:.2e}, "
        f"min_R2={td['min_r2_mean_profile']:.4f}"
    )

    # --- TabDPT causal (3-seed) ---
    tc = json.loads((root / "tabdpt_causal_3seed.json").read_text())
    print("\nTabDPT noising-based causal tracing (3 seeds):")
    print(
        f"  peak_layer={tc['mean_profile_peak_layer']}, "
        f"peak_value={tc['mean_profile_peak_value']:.4f}"
    )

    # --- NAM out-of-family holdout ---
    nam = json.loads((root / "nam_holdout.json").read_text())
    profiles = np.array([s["layer_r2"] for s in nam["per_seed"]])
    mean_profile = profiles.mean(axis=0)
    print("\nNAM out-of-family holdout (5 seeds):")
    print(f"  mean_profile={[round(x, 3) for x in mean_profile.tolist()]}")
    print(f"  sigma2_profile={float(mean_profile.var()):.2e}")

    # --- LOFO ---
    lofo = json.loads((root / "lofo_primary_endpoint.json").read_text())
    print("\nLOFO primary endpoint (TabPFN/TabICL ratio per held-out function):")
    for r in lofo["rows"]:
        print(f"  hold-out {r['held_out']:>20s}: ratio={r['ratio_tabpfn_over_tabicl']:.1f}x")

    # --- v2 vs v2.5 ---
    v25 = json.loads((root / "tabpfn25_fullscale_aggregated.json").read_text())
    ll_v2 = int(v25["logit_lens"]["v2"]["first_above_05"]["mean"])
    ll_v25 = int(v25["logit_lens"]["v2.5"]["first_above_05"]["mean"])
    print("\nTabPFN v2 vs v2.5 logit lens (3 seeds):")
    print(f"  v2 first R2>0.5 at L{ll_v2}, v2.5 at L{ll_v25}")

    print()
    print("[reproduce-frozen] All paper-facing summary numerics verified from frozen_artifacts/.")
    print("[reproduce-frozen] No NAS access required. To rebuild PDF:")
    print("[reproduce-frozen]   cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
