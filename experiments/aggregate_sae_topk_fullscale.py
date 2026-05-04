# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEEDS = [42, 123, 456, 789, 1024]
BASE_DIR = ROOT / "results" / "rd6" / "sae_topk_fullscale"
OUT_DIR = BASE_DIR / "aggregated"
MODELS = ["tabpfn", "tabicl", "iltm"]
VARIANTS = ["relu_16x", "jumprelu_16x", "topk_16x_6p25"]
METRICS = ["max_alpha_corr", "sparsity", "reconstruction_r2"]


def _load_seed(seed: int) -> dict[str, Any]:
    path = BASE_DIR / f"seed_{seed}" / "results.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _aggregate(seed_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {}
    for variant in VARIANTS:
        agg[variant] = {}
        for model in MODELS:
            agg[variant][model] = {}
            for metric in METRICS:
                vals = [
                    float(payload["results"][variant][model][metric])
                    for payload in seed_payloads
                ]
                arr = np.asarray(vals, dtype=np.float64)
                agg[variant][model][metric] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "values": vals,
                }
    return agg


def _plot_metric(
    agg: dict[str, Any], metric: str, ylabel: str, save_path: Path
) -> None:
    x = np.arange(len(MODELS))
    width = 0.8 / len(VARIANTS)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, variant in enumerate(VARIANTS):
        means = [agg[variant][m][metric]["mean"] for m in MODELS]
        stds = [agg[variant][m][metric]["std"] for m in MODELS]
        offsets = x - 0.4 + (i + 0.5) * width
        ax.bar(offsets, means, width=width, yerr=stds, capsize=4, label=variant)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel(ylabel)
    ax.set_title(f"SAE TopK 5-seed: {metric} (mean±std)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = [_load_seed(seed) for seed in SEEDS]
    agg = _aggregate(payloads)

    out_json = {
        "seeds": SEEDS,
        "variants": VARIANTS,
        "models": MODELS,
        "metrics": METRICS,
        "aggregated": agg,
    }
    json_path = OUT_DIR / "aggregated_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    _plot_metric(
        agg, "max_alpha_corr", "max |r_alpha|", OUT_DIR / "alpha_corr_errorbars.png"
    )
    _plot_metric(agg, "sparsity", "sparsity", OUT_DIR / "sparsity_errorbars.png")
    _plot_metric(
        agg,
        "reconstruction_r2",
        "reconstruction R2",
        OUT_DIR / "recon_r2_errorbars.png",
    )

    print(f"Saved: {json_path}")
    print(f"Saved plots in: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
