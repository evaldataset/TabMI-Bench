# pyright: reportMissingImports=false
"""Phase 8A: Non-linear function invariance — noising-based causal tracing.

Verifies that causal layer sensitivity profiles are invariant to function
type: TabPFN should peak at L5-L8 and TabICL at L0 regardless of whether
the target function is sinusoidal, polynomial, or mixed.
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
import torch
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

from src.data.synthetic_generator import generate_nonlinear_data, generate_quadratic_data
from rd5_config import cfg

FUNC_TYPES = ["sinusoidal", "polynomial", "mixed"]
BASELINE_FUNC = "quadratic"
NOISE_SCALE = 0.5  # fraction of activation std

RESULTS_DIR = ROOT / "results" / "phase8a" / "nonlinear_causal"


def _generate_data(func_type: str, seed: int) -> dict[str, np.ndarray]:
    """Generate data for a given function type."""
    if func_type == BASELINE_FUNC:
        ds = generate_quadratic_data(n_train=cfg.N_TRAIN, n_test=cfg.N_TEST, random_seed=seed)
    else:
        ds = generate_nonlinear_data(func_type=func_type, n_train=cfg.N_TRAIN, n_test=cfg.N_TEST, random_seed=seed)
    return {
        "X_train": ds.X_train.astype(np.float32),
        "X_test": ds.X_test.astype(np.float32),
        "y_train": ds.y_train.astype(np.float32),
    }


def _noising_sweep(
    model: Any,
    layers: Any,
    X_test: np.ndarray,
    noise_scale: float,
    seed: int,
) -> dict[str, Any]:
    """Run noising-based causal tracing on a list of layer modules."""
    n_layers = len(layers)
    preds_clean = model.predict(X_test)
    mse_by_layer: list[float] = []

    for layer_idx in range(n_layers):
        # Get activation scale
        act_stds: list[float] = []

        def _stats_hook(module: Any, input: Any, output: torch.Tensor) -> None:  # noqa: A002
            act_stds.append(output.std().item())

        h = layers[layer_idx].register_forward_hook(_stats_hook)
        _ = model.predict(X_test)
        h.remove()

        if not act_stds:
            import warnings
            warnings.warn(f"Layer {layer_idx}: no activation stats collected, using default std=1.0")
        mean_std = float(np.mean(act_stds)) if act_stds else 1.0
        rng = np.random.default_rng(seed + layer_idx)

        def _make_noise_hook(scale: float, act_std: float, rng_: np.random.Generator) -> Any:
            def hook(module: Any, input: Any, output: torch.Tensor) -> torch.Tensor:  # noqa: A002
                noise = torch.from_numpy(
                    rng_.normal(0, scale * act_std, output.shape).astype(np.float32)
                ).to(output.device)
                return output + noise
            return hook

        h = layers[layer_idx].register_forward_hook(
            _make_noise_hook(noise_scale, mean_std, rng)
        )
        preds_noised = model.predict(X_test)
        h.remove()

        mse = float(np.mean((preds_noised - preds_clean) ** 2))
        mse_by_layer.append(mse)

    max_mse = max(mse_by_layer) if max(mse_by_layer) > 0 else 1.0
    normalized = [m / max_mse for m in mse_by_layer]
    peak = int(np.argmax(mse_by_layer))

    return {
        "mse_by_layer": mse_by_layer,
        "normalized_sensitivity": normalized,
        "most_sensitive_layer": peak,
    }


def _run_model_func(model_name: str, func_type: str) -> dict[str, Any]:
    """Run causal tracing for one model × one function."""
    data = _generate_data(func_type, cfg.SEED)

    if model_name == "tabpfn":
        model = TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")
        model.fit(data["X_train"], data["y_train"])
        layers = model.model_.transformer_encoder.layers
    elif model_name == "tabicl":
        model = TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)
        model.fit(data["X_train"], data["y_train"])
        layers = model.model_.icl_predictor.tf_icl.blocks
    else:
        raise ValueError(model_name)

    result = _noising_sweep(model, layers, data["X_test"], NOISE_SCALE, cfg.SEED)
    result["model"] = model_name
    result["func_type"] = func_type
    result["n_layers"] = len(layers)
    return result


def _plot_results(all_results: dict[str, dict[str, dict[str, Any]]], save_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"quadratic": "gray", "sinusoidal": "blue", "polynomial": "green", "mixed": "red"}

    for ax, model_name in zip(axes, ["tabpfn", "tabicl"], strict=True):
        for func_type in [BASELINE_FUNC] + FUNC_TYPES:
            r = all_results[model_name][func_type]
            sens = r["normalized_sensitivity"]
            layers = np.arange(len(sens))
            style = "--" if func_type == BASELINE_FUNC else "-"
            ax.plot(layers, sens, marker="o", linewidth=2 if style == "-" else 1.5,
                    linestyle=style, color=colors[func_type], label=func_type, alpha=0.85)
        ax.set_title(f"{model_name.upper()} — Causal Sensitivity", fontsize=12)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Normalized Sensitivity")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle("Phase 8A: Non-Linear Causal Tracing — Function Invariance", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / "nonlinear_causal_overlay.png", dpi=180, bbox_inches="tight")
    fig.savefig(save_dir / "nonlinear_causal_overlay.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("Phase 8A: Non-Linear Function Invariance — Causal Tracing")
    print("=" * 72)
    print(f"SEED={cfg.SEED}, noise_scale={NOISE_SCALE}")

    all_results: dict[str, dict[str, dict[str, Any]]] = {
        "tabpfn": {}, "tabicl": {},
    }

    for model_name in ["tabpfn", "tabicl"]:
        for func_type in [BASELINE_FUNC] + FUNC_TYPES:
            print(f"\n--- {model_name} × {func_type} ---")
            result = _run_model_func(model_name, func_type)
            all_results[model_name][func_type] = result
            print(f"  Most sensitive layer: L{result['most_sensitive_layer']}")

    json_path = RESULTS_DIR / f"results_seed{cfg.SEED}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    _plot_results(all_results, RESULTS_DIR)

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY: Most sensitive layer by model × function")
    print("=" * 72)
    for model_name in ["tabpfn", "tabicl"]:
        peaks = []
        for ft in [BASELINE_FUNC] + FUNC_TYPES:
            peak = all_results[model_name][ft]["most_sensitive_layer"]
            peaks.append(f"{ft}=L{peak}")
        print(f"  {model_name}: {', '.join(peaks)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
