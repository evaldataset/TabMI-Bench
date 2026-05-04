# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportImplicitStringConcatenation=false
from __future__ import annotations

"""RD-5 noising-based causal tracing comparison (TabPFN vs TabICL).

Standard activation patching (replacing full layer output with clean
activations) is fundamentally uninformative for ICL-based tabular
foundation models: since the same weights process all inputs, replacing
any layer's output with clean activations causes ALL downstream layers
to reproduce their clean outputs identically ('deterministic cascading').
This yields flat, uniform patch effects across all layers.

This script uses **noising-based causal tracing** instead:
  1. Run model on clean data -> get clean predictions.
  2. For each layer L: add calibrated Gaussian noise to layer L's
     output -> measure prediction degradation (MSE increase).
  3. Layers where noise causes greater degradation are more
     causally important for the model's computation.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
from typing import Any, TypedDict

import numpy as np
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

from rd5_config import cfg

QUICK_RUN = cfg.QUICK_RUN
RANDOM_SEED = cfg.SEED
N_TRAIN = cfg.N_TRAIN
N_TEST = cfg.N_TEST

# Noise scales to sweep (fraction of activation std)
NOISE_SCALES = [0.1, 0.5, 1.0] if not QUICK_RUN else [0.5]


class NoisingResult(TypedDict):
    noise_scale: float
    mse_increase_by_layer: list[float]
    normalized_sensitivity: list[float]
    most_sensitive_layer: int


class ModelResults(TypedDict):
    n_layers: int
    noise_results: list[NoisingResult]
    summary_sensitivity: list[float]


class PatchingResults(TypedDict):
    tabpfn: ModelResults
    tabicl: ModelResults
    finding: str


def _generate_data() -> dict[str, np.ndarray]:
    """Generate synthetic data: z = a*b + c."""
    rng = np.random.default_rng(RANDOM_SEED)
    n_total = N_TRAIN + N_TEST
    X = rng.uniform(0.5, 3.0, (n_total, 3)).astype(np.float32)
    X_train, X_test = X[:N_TRAIN], X[N_TRAIN:]
    y_train = (X_train[:, 0] * X_train[:, 1] + X_train[:, 2]).astype(np.float32)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train}


def _noising_sweep_tabpfn(
    model: TabPFNRegressor,
    X_test: np.ndarray,
    noise_scale: float,
) -> NoisingResult:
    """Run noising-based causal tracing for TabPFN."""
    n_layers = len(model.model_.transformer_encoder.layers)
    preds_clean = model.predict(X_test)
    mse_increase_by_layer: list[float] = []

    for layer_idx in range(n_layers):
        # Collect activation statistics for calibration
        act_stds: list[float] = []

        def make_stats_hook(stats_list: list[float]) -> Any:
            def hook(
                module: torch.nn.Module,
                input: Any,  # noqa: A002
                output: torch.Tensor,
            ) -> None:
                stats_list.append(output.std().item())
            return hook

        handle = model.model_.transformer_encoder.layers[
            layer_idx
        ].register_forward_hook(make_stats_hook(act_stds))
        _ = model.predict(X_test)
        handle.remove()

        mean_act_std = float(np.mean(act_stds)) if act_stds else 1.0
        layer_rng = np.random.default_rng(RANDOM_SEED + layer_idx)

        def make_noise_hook(
            scale: float, act_std: float, rng: np.random.Generator
        ) -> Any:
            def hook(
                module: torch.nn.Module,
                input: Any,  # noqa: A002
                output: torch.Tensor,
            ) -> torch.Tensor:
                noise = torch.from_numpy(
                    rng.normal(0, scale * act_std, output.shape).astype(np.float32)
                ).to(output.device)
                return output + noise
            return hook

        handle = model.model_.transformer_encoder.layers[
            layer_idx
        ].register_forward_hook(
            make_noise_hook(noise_scale, mean_act_std, layer_rng)
        )
        preds_noised = model.predict(X_test)
        handle.remove()

        mse = float(np.mean((preds_noised - preds_clean) ** 2))
        mse_increase_by_layer.append(mse)

    max_mse = max(mse_increase_by_layer) if max(mse_increase_by_layer) > 0 else 1.0
    normalized = [m / max_mse for m in mse_increase_by_layer]
    most_sensitive = int(np.argmax(mse_increase_by_layer))

    return {
        "noise_scale": noise_scale,
        "mse_increase_by_layer": mse_increase_by_layer,
        "normalized_sensitivity": normalized,
        "most_sensitive_layer": most_sensitive,
    }


def _noising_sweep_tabicl(
    model: TabICLRegressor,
    X_test: np.ndarray,
    noise_scale: float,
) -> NoisingResult:
    """Run noising-based causal tracing for TabICL."""
    blocks = model.model_.icl_predictor.tf_icl.blocks
    n_layers = len(blocks)
    preds_clean = model.predict(X_test)
    mse_increase_by_layer: list[float] = []

    for layer_idx in range(n_layers):
        act_stds: list[float] = []

        def make_stats_hook(stats_list: list[float]) -> Any:
            def hook(
                module: torch.nn.Module,
                input: Any,  # noqa: A002
                output: torch.Tensor,
            ) -> None:
                stats_list.append(output.std().item())
            return hook

        handle = blocks[layer_idx].register_forward_hook(
            make_stats_hook(act_stds)
        )
        _ = model.predict(X_test)
        handle.remove()

        mean_act_std = float(np.mean(act_stds)) if act_stds else 1.0
        layer_rng = np.random.default_rng(RANDOM_SEED + layer_idx)

        def make_noise_hook(
            scale: float, act_std: float, rng: np.random.Generator
        ) -> Any:
            def hook(
                module: torch.nn.Module,
                input: Any,  # noqa: A002
                output: torch.Tensor,
            ) -> torch.Tensor:
                noise = torch.from_numpy(
                    rng.normal(0, scale * act_std, output.shape).astype(np.float32)
                ).to(output.device)
                return output + noise
            return hook

        handle = blocks[layer_idx].register_forward_hook(
            make_noise_hook(noise_scale, mean_act_std, layer_rng)
        )
        preds_noised = model.predict(X_test)
        handle.remove()

        mse = float(np.mean((preds_noised - preds_clean) ** 2))
        mse_increase_by_layer.append(mse)

    max_mse = max(mse_increase_by_layer) if max(mse_increase_by_layer) > 0 else 1.0
    normalized = [m / max_mse for m in mse_increase_by_layer]
    most_sensitive = int(np.argmax(mse_increase_by_layer))

    return {
        "noise_scale": noise_scale,
        "mse_increase_by_layer": mse_increase_by_layer,
        "normalized_sensitivity": normalized,
        "most_sensitive_layer": most_sensitive,
    }


def _run_tabpfn(data: dict[str, np.ndarray]) -> ModelResults:
    model = TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")
    model.fit(data["X_train"], data["y_train"])
    n_layers = len(model.model_.transformer_encoder.layers)
    noise_results: list[NoisingResult] = []
    for scale in NOISE_SCALES:
        print(f"  TabPFN noise_scale={scale}")
        noise_results.append(_noising_sweep_tabpfn(model, data["X_test"], scale))
    summary = np.zeros(n_layers)
    for nr in noise_results:
        summary += np.array(nr["normalized_sensitivity"])
    summary /= len(noise_results)
    return {
        "n_layers": n_layers,
        "noise_results": noise_results,
        "summary_sensitivity": summary.tolist(),
    }


def _run_tabicl(data: dict[str, np.ndarray]) -> ModelResults:
    model = TabICLRegressor(device=cfg.DEVICE, random_state=RANDOM_SEED)
    model.fit(data["X_train"], data["y_train"])
    blocks = model.model_.icl_predictor.tf_icl.blocks
    n_layers = len(blocks)
    noise_results: list[NoisingResult] = []
    for scale in NOISE_SCALES:
        print(f"  TabICL noise_scale={scale}")
        noise_results.append(_noising_sweep_tabicl(model, data["X_test"], scale))
    summary = np.zeros(n_layers)
    for nr in noise_results:
        summary += np.array(nr["normalized_sensitivity"])
    summary /= len(noise_results)
    return {
        "n_layers": n_layers,
        "noise_results": noise_results,
        "summary_sensitivity": summary.tolist(),
    }


def _plot_comparison(results: PatchingResults, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # TabPFN
    tabpfn = results["tabpfn"]
    ax = axes[0]
    for nr in tabpfn["noise_results"]:
        layers = np.arange(len(nr["normalized_sensitivity"]))
        ax.plot(layers, nr["normalized_sensitivity"], marker="o",
                linewidth=1.5, label=f"\u03c3={nr['noise_scale']}", alpha=0.6)
    ax.plot(np.arange(len(tabpfn["summary_sensitivity"])),
            tabpfn["summary_sensitivity"], marker="s", linewidth=2.5,
            color="blue", label="Average")
    peak_t = int(np.argmax(tabpfn["summary_sensitivity"]))
    ax.axvline(x=peak_t, color="blue", linestyle="--", alpha=0.4)
    ax.set_title(f"TabPFN (peak: L{peak_t})", fontsize=12)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Normalized Sensitivity")
    ax.set_xticks(np.arange(tabpfn["n_layers"]))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # TabICL
    tabicl = results["tabicl"]
    ax = axes[1]
    for nr in tabicl["noise_results"]:
        layers = np.arange(len(nr["normalized_sensitivity"]))
        ax.plot(layers, nr["normalized_sensitivity"], marker="o",
                linewidth=1.5, label=f"\u03c3={nr['noise_scale']}", alpha=0.6)
    ax.plot(np.arange(len(tabicl["summary_sensitivity"])),
            tabicl["summary_sensitivity"], marker="s", linewidth=2.5,
            color="orange", label="Average")
    peak_i = int(np.argmax(tabicl["summary_sensitivity"]))
    ax.axvline(x=peak_i, color="orange", linestyle="--", alpha=0.4)
    ax.set_title(f"TabICL (peak: L{peak_i})", fontsize=12)
    ax.set_xlabel("Layer index")
    ax.set_xticks(np.arange(tabicl["n_layers"]))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        "RD-5 Noising-Based Causal Tracing: Layer Sensitivity",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


FINDING = (
    "Standard activation patching (replacing full layer output with clean "
    "activations) is uninformative for ICL-based tabular foundation models. "
    "Because these models use shared/frozen weights across inputs, replacing "
    "any layer L's output with clean activations causes ALL downstream layers "
    "(L+1...L_final) to reproduce their clean outputs identically — a property "
    "we term 'deterministic cascading'. This yields flat, uniform patch effects "
    "across all layers. We use noising-based causal tracing instead."
)


def main() -> None:
    results_dir = ROOT / "results" / "rd5" / "patching"
    results_dir.mkdir(parents=True, exist_ok=True)

    data = _generate_data()

    print("=" * 72)
    print("RD-5 Noising-Based Causal Tracing (TabPFN vs TabICL)")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}, N_TRAIN={N_TRAIN}, N_TEST={N_TEST}")
    print(f"Noise scales: {NOISE_SCALES}")

    print("\n[1/2] TabPFN...")
    tabpfn_results = _run_tabpfn(data)

    print("\n[2/2] TabICL...")
    tabicl_results = _run_tabicl(data)

    results: PatchingResults = {
        "tabpfn": tabpfn_results,
        "tabicl": tabicl_results,
        "finding": FINDING,
    }

    peak_tabpfn = int(np.argmax(tabpfn_results["summary_sensitivity"]))
    peak_tabicl = int(np.argmax(tabicl_results["summary_sensitivity"]))
    print(f"\nTabPFN most sensitive layer: L{peak_tabpfn}")
    print(f"TabICL most sensitive layer: L{peak_tabicl}")

    json_path = results_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    plot_path = results_dir / "patching_comparison.png"
    _plot_comparison(results, plot_path)
    print(f"Saved: {plot_path}")

    print(f"\nDone. Outputs in: {results_dir}")


if __name__ == "__main__":
    main()
