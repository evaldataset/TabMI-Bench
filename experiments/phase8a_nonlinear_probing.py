# pyright: reportMissingImports=false
"""Phase 8A: Non-linear function invariance — intermediary probing.

Tests whether the staged/distributed/preprocessing-dominant taxonomy holds
across non-linear synthetic functions (sinusoidal, polynomial, mixed),
not just the original z = a*b + c.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false
import json
import os
from typing import Any

import numpy as np
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor
from iltm import iLTMRegressor

from src.data.synthetic_generator import generate_nonlinear_data, generate_quadratic_data
from src.hooks.tabpfn_hooker import TabPFNHookedModel
from src.hooks.tabicl_hooker import TabICLHookedModel
from src.hooks.iltm_hooker import iLTMHookedModel
from src.probing.linear_probe import probe_layer
from rd5_config import cfg

FUNC_TYPES = ["sinusoidal", "polynomial", "mixed"]
BASELINE_FUNC = "quadratic"  # original z = a*b + c for comparison
PROBE_COMPLEXITIES = [0, 1]  # linear + 1-hidden-layer MLP

RESULTS_DIR = ROOT / "results" / "phase8a" / "nonlinear_probing"

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "tabpfn": {"layers": list(range(12)), "hidden_dim": 192},
    "tabicl": {"layers": list(range(12)), "hidden_dim": 512},
    "iltm": {"layers": [0, 1, 2], "hidden_dim": 512},
}


def _build_model(name: str) -> Any:
    if name == "tabpfn":
        return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")
    elif name == "tabicl":
        return TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)
    elif name == "iltm":
        m = iLTMRegressor(device="cpu", n_ensemble=1, seed=cfg.SEED)
        m.n_ensemble = 1
        return m
    raise ValueError(name)


def _build_hooker(name: str, model: Any) -> Any:
    if name == "tabpfn":
        return TabPFNHookedModel(model)
    elif name == "tabicl":
        return TabICLHookedModel(model)
    elif name == "iltm":
        return iLTMHookedModel(model)
    raise ValueError(name)


def _get_activations(name: str, hooker: Any, cache: Any, layer_idx: int) -> np.ndarray:
    if name == "tabpfn":
        return np.asarray(hooker.get_test_label_token(cache, layer_idx), dtype=np.float32)
    return np.asarray(hooker.get_layer_activations(cache, layer_idx), dtype=np.float32)


def _n_datasets(model_name: str) -> int:
    if cfg.QUICK_RUN:
        return 3 if model_name == "iltm" else 5
    return 5 if model_name == "iltm" else 10


def _run_probing_for_func(
    model_name: str,
    func_type: str,
) -> dict[str, Any]:
    """Run intermediary probing for one model × one function type.

    Note: Activations from multiple datasets are pooled before train/test
    splitting inside probe_layer(). The split is random and may mix samples
    from the same underlying dataset. This follows the same protocol as the
    Phase 5 rd5_intermediary_probing.py experiments for consistency.
    """
    layer_indices = MODEL_SPECS[model_name]["layers"]
    n_ds = _n_datasets(model_name)

    pooled_acts: dict[int, list[np.ndarray]] = {i: [] for i in layer_indices}
    pooled_intermediary: list[np.ndarray] = []
    pooled_final: list[np.ndarray] = []

    for ds_idx in range(n_ds):
        seed = cfg.SEED + ds_idx

        if func_type == BASELINE_FUNC:
            dataset = generate_quadratic_data(
                n_train=cfg.N_TRAIN, n_test=cfg.N_TEST, random_seed=seed,
            )
        else:
            dataset = generate_nonlinear_data(
                func_type=func_type,
                n_train=cfg.N_TRAIN, n_test=cfg.N_TEST, random_seed=seed,
            )

        model = _build_model(model_name)
        model.fit(dataset.X_train, dataset.y_train)
        hooker = _build_hooker(model_name, model)
        _, cache = hooker.forward_with_cache(dataset.X_test)

        for li in layer_indices:
            pooled_acts[li].append(_get_activations(model_name, hooker, cache, li))

        pooled_intermediary.append(
            np.asarray(dataset.intermediary_test, dtype=np.float32).reshape(-1)
        )
        pooled_final.append(
            np.asarray(dataset.y_test, dtype=np.float32).reshape(-1)
        )

    intermediary_targets = np.concatenate(pooled_intermediary)
    final_targets = np.concatenate(pooled_final)

    results_by_complexity: dict[int, dict[str, list[float]]] = {}

    for complexity in PROBE_COMPLEXITIES:
        intermediary_r2: list[float] = []
        final_r2: list[float] = []

        for li in layer_indices:
            acts = np.vstack(pooled_acts[li])
            iprobe = probe_layer(acts, intermediary_targets, complexities=[complexity], random_seed=cfg.SEED)
            fprobe = probe_layer(acts, final_targets, complexities=[complexity], random_seed=cfg.SEED)
            intermediary_r2.append(float(iprobe[complexity]["r2"]))
            final_r2.append(float(fprobe[complexity]["r2"]))

        results_by_complexity[complexity] = {
            "intermediary_r2_by_layer": intermediary_r2,
            "final_r2_by_layer": final_r2,
        }

    # Use linear probe (complexity=0) for peak layer
    linear_r2 = results_by_complexity[0]["intermediary_r2_by_layer"]
    peak_layer = int(np.argmax(np.asarray(linear_r2)))

    return {
        "func_type": func_type,
        "model": model_name,
        "n_datasets": n_ds,
        "results_by_complexity": {str(k): v for k, v in results_by_complexity.items()},
        "peak_layer_intermediary": peak_layer,
        "peak_r2_intermediary": linear_r2[peak_layer],
    }


def _plot_results(all_results: dict[str, dict[str, dict[str, Any]]], save_dir: Path) -> None:
    """Plot intermediary R² profiles per model, overlaying function types."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    model_order = ["tabpfn", "tabicl", "iltm"]
    colors = {"quadratic": "gray", "sinusoidal": "blue", "polynomial": "green", "mixed": "red"}

    for ax, model_name in zip(axes, model_order, strict=True):
        for func_type in [BASELINE_FUNC] + FUNC_TYPES:
            if func_type not in all_results[model_name]:
                continue
            r2 = all_results[model_name][func_type]["results_by_complexity"]["0"]["intermediary_r2_by_layer"]
            layers = np.arange(len(r2))
            style = "--" if func_type == BASELINE_FUNC else "-"
            ax.plot(layers, r2, marker="o", linewidth=2 if style == "-" else 1.5,
                    linestyle=style, color=colors.get(func_type, "black"),
                    label=func_type, alpha=0.85)
        ax.set_title(model_name.upper(), fontsize=12)
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.25)
        if model_name == "tabpfn":
            ax.set_ylabel("Intermediary R²")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Phase 8A: Non-Linear Function Invariance — Intermediary Probing (Linear)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / "nonlinear_probing_overlay.png", dpi=180, bbox_inches="tight")
    fig.savefig(save_dir / "nonlinear_probing_overlay.pdf", bbox_inches="tight")
    plt.close(fig)

    # MLP probe comparison
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, model_name in zip(axes2, model_order, strict=True):
        for func_type in FUNC_TYPES:
            if func_type not in all_results[model_name]:
                continue
            r2_linear = all_results[model_name][func_type]["results_by_complexity"]["0"]["intermediary_r2_by_layer"]
            r2_mlp = all_results[model_name][func_type]["results_by_complexity"]["1"]["intermediary_r2_by_layer"]
            layers = np.arange(len(r2_linear))
            ax.plot(layers, r2_linear, marker="o", linewidth=1.5, linestyle="--",
                    color=colors[func_type], alpha=0.6, label=f"{func_type} (linear)")
            ax.plot(layers, r2_mlp, marker="s", linewidth=2, linestyle="-",
                    color=colors[func_type], alpha=0.85, label=f"{func_type} (MLP)")
        ax.set_title(model_name.upper(), fontsize=12)
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.25)
        if model_name == "tabpfn":
            ax.set_ylabel("Intermediary R²")
        ax.legend(fontsize=7, loc="best")

    fig2.suptitle("Phase 8A: Linear vs MLP Probe Comparison", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(save_dir / "nonlinear_linear_vs_mlp.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("Phase 8A: Non-Linear Function Invariance — Intermediary Probing")
    print("=" * 72)
    print(f"QUICK_RUN={cfg.QUICK_RUN}, SEED={cfg.SEED}, N_TRAIN={cfg.N_TRAIN}, N_TEST={cfg.N_TEST}")
    print(f"Functions: {[BASELINE_FUNC] + FUNC_TYPES}")
    print(f"Probe complexities: {PROBE_COMPLEXITIES}")

    all_results: dict[str, dict[str, dict[str, Any]]] = {}

    for model_name in MODEL_SPECS:
        all_results[model_name] = {}
        for func_type in [BASELINE_FUNC] + FUNC_TYPES:
            print(f"\n--- {model_name} × {func_type} ---")
            result = _run_probing_for_func(model_name, func_type)
            all_results[model_name][func_type] = result

            peak = result["peak_layer_intermediary"]
            r2 = result["peak_r2_intermediary"]
            print(f"  Peak layer: L{peak}, R²={r2:.4f}")

    # Save results
    json_path = RESULTS_DIR / f"results_seed{cfg.SEED}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Plot
    _plot_results(all_results, RESULTS_DIR)
    print(f"Plots saved to: {RESULTS_DIR}")

    # Summary table
    print("\n" + "=" * 72)
    print("SUMMARY: Peak intermediary R² (linear probe) by model × function")
    print("=" * 72)
    header = f"{'Model':<10}" + "".join(f"{ft:<14}" for ft in [BASELINE_FUNC] + FUNC_TYPES)
    print(header)
    for model_name in MODEL_SPECS:
        row = f"{model_name:<10}"
        for ft in [BASELINE_FUNC] + FUNC_TYPES:
            if ft in all_results[model_name]:
                r = all_results[model_name][ft]
                row += f"L{r['peak_layer_intermediary']:>2} R²={r['peak_r2_intermediary']:.3f}  "
            else:
                row += f"{'N/A':<14}"
        print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
