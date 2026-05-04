# pyright: reportMissingImports=false
"""Generate per-seed TabPFN causal peak distribution figure.

Visualizes the instability of TabPFN's causal peak across seeds,
complementing the average-profile report in the main text. Motivated by
REVISION §5.3.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEED_DIRS = sorted((ROOT / "results" / "rd5_fullscale").glob("seed_*"))
OUT_PATH = ROOT / "paper" / "figures" / "fig_seed_instability.pdf"


def load_sensitivity(seed_dir: Path) -> tuple[list[float], list[float]]:
    path = seed_dir / "patching" / "results.json"
    if not path.exists():
        return [], []
    with path.open() as f:
        d = json.load(f)
    return (
        list(map(float, d["tabpfn"]["summary_sensitivity"])),
        list(map(float, d["tabicl"]["summary_sensitivity"])),
    )


def main() -> int:
    seeds = []
    tabpfn_all = []
    tabicl_all = []
    for sd in SEED_DIRS:
        seed = int(sd.name.replace("seed_", ""))
        pfn, icl = load_sensitivity(sd)
        if not pfn or not icl:
            continue
        seeds.append(seed)
        tabpfn_all.append(pfn)
        tabicl_all.append(icl)

    if not seeds:
        print("No data found")
        return 1

    tabpfn_arr = np.asarray(tabpfn_all)  # [n_seeds, n_layers]
    tabicl_arr = np.asarray(tabicl_all)
    n_seeds, n_layers_pfn = tabpfn_arr.shape
    _, n_layers_icl = tabicl_arr.shape

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: TabPFN per-seed profiles + mean overlay
    ax = axes[0]
    for i, seed in enumerate(seeds):
        ax.plot(range(n_layers_pfn), tabpfn_arr[i], alpha=0.4,
                color="C0", linewidth=1.2, label=f"seed {seed}" if i < 3 else None)
    mean_pfn = tabpfn_arr.mean(axis=0)
    ax.plot(range(n_layers_pfn), mean_pfn, color="red", linewidth=2.5,
            label="Mean profile", linestyle="--")
    # Mark peak layers per seed
    peaks_pfn = [int(np.argmax(tabpfn_arr[i])) for i in range(n_seeds)]
    for i, pk in enumerate(peaks_pfn):
        ax.scatter([pk], [tabpfn_arr[i, pk]], color="C0", s=60, zorder=5,
                   edgecolors="black", linewidths=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized causal sensitivity")
    ax.set_title(f"TabPFN: per-seed causal profile (peaks: {peaks_pfn})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    # Right: TabICL per-seed profiles + mean overlay
    ax = axes[1]
    for i, seed in enumerate(seeds):
        ax.plot(range(n_layers_icl), tabicl_arr[i], alpha=0.4,
                color="C1", linewidth=1.2, label=f"seed {seed}" if i < 3 else None)
    mean_icl = tabicl_arr.mean(axis=0)
    ax.plot(range(n_layers_icl), mean_icl, color="red", linewidth=2.5,
            label="Mean profile", linestyle="--")
    peaks_icl = [int(np.argmax(tabicl_arr[i])) for i in range(n_seeds)]
    for i, pk in enumerate(peaks_icl):
        ax.scatter([pk], [tabicl_arr[i, pk]], color="C1", s=60, zorder=5,
                   edgecolors="black", linewidths=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized causal sensitivity")
    ax.set_title(f"TabICL: per-seed causal profile (peaks: {peaks_icl})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    print(f"TabPFN peak layers across {n_seeds} seeds: {peaks_pfn}")
    print(f"TabICL peak layers across {n_seeds} seeds: {peaks_icl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
