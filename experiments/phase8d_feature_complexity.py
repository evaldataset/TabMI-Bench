# pyright: reportMissingImports=false
"""Phase 8D: Feature complexity sweep — peak layer vs feature count.

Shows that TabPFN's peak computation layer shifts to deeper layers as
feature count increases (non-trivial finding), while TabICL remains
uniformly distributed regardless of feature count.
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
from iltm import iLTMRegressor

from src.data.synthetic_generator import generate_multifeature_data
from src.hooks.tabpfn_hooker import TabPFNHookedModel
from src.hooks.tabicl_hooker import TabICLHookedModel
from src.hooks.iltm_hooker import iLTMHookedModel
from src.probing.linear_probe import probe_layer
from rd5_config import cfg

FEATURE_COUNTS = [2, 4, 8, 16]
NOISE_SCALE = 0.5

RESULTS_DIR = ROOT / "results" / "phase8d" / "feature_complexity"

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


def _run_probing(model_name: str, n_features: int) -> dict[str, Any]:
    """Run intermediary probing for one model × one feature count."""
    layer_indices = MODEL_SPECS[model_name]["layers"]
    n_ds = 3 if cfg.QUICK_RUN else 8

    pooled_acts: dict[int, list[np.ndarray]] = {i: [] for i in layer_indices}
    pooled_intermediary: list[np.ndarray] = []

    for ds_idx in range(n_ds):
        seed = cfg.SEED + ds_idx
        dataset = generate_multifeature_data(
            n_features=n_features,
            n_train=cfg.N_TRAIN, n_test=cfg.N_TEST,
            random_seed=seed,
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

    targets = np.concatenate(pooled_intermediary)
    r2_by_layer: list[float] = []
    for li in layer_indices:
        acts = np.vstack(pooled_acts[li])
        probe_result = probe_layer(acts, targets, complexities=[0], random_seed=cfg.SEED)
        r2_by_layer.append(float(probe_result[0]["r2"]))  # key=complexity=0

    peak_layer = int(np.argmax(np.asarray(r2_by_layer)))
    return {
        "n_features": n_features,
        "r2_by_layer": r2_by_layer,
        "peak_layer": peak_layer,
        "peak_r2": r2_by_layer[peak_layer],
    }


def _run_causal(model_name: str, n_features: int) -> dict[str, Any]:
    """Run noising-based causal tracing for one model × one feature count.

    Note: Shares the same noising-sweep logic as phase8a_nonlinear_causal.py.
    Both scripts inline the sweep to avoid an import dependency; a shared
    utility (e.g., src/hooks/noising_sweep.py) would reduce duplication.
    """
    if model_name not in ("tabpfn", "tabicl"):
        return {"n_features": n_features, "skipped": True}

    dataset = generate_multifeature_data(
        n_features=n_features,
        n_train=cfg.N_TRAIN, n_test=cfg.N_TEST,
        random_seed=cfg.SEED,
    )

    if model_name == "tabpfn":
        model = TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")
        model.fit(dataset.X_train, dataset.y_train)
        layers = model.model_.transformer_encoder.layers
    else:
        model = TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)
        model.fit(dataset.X_train, dataset.y_train)
        layers = model.model_.icl_predictor.tf_icl.blocks

    X_test = dataset.X_test.astype(np.float32)
    preds_clean = model.predict(X_test)
    mse_by_layer: list[float] = []

    for layer_idx in range(len(layers)):
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
        rng = np.random.default_rng(cfg.SEED + layer_idx)

        def _make_noise_hook(scale: float, act_std: float, rng_: np.random.Generator) -> Any:
            def hook(module: Any, input: Any, output: torch.Tensor) -> torch.Tensor:  # noqa: A002
                noise = torch.from_numpy(
                    rng_.normal(0, scale * act_std, output.shape).astype(np.float32)
                ).to(output.device)
                return output + noise
            return hook

        h = layers[layer_idx].register_forward_hook(
            _make_noise_hook(NOISE_SCALE, mean_std, rng)
        )
        preds_noised = model.predict(X_test)
        h.remove()
        mse_by_layer.append(float(np.mean((preds_noised - preds_clean) ** 2)))

    max_mse = max(mse_by_layer) if max(mse_by_layer) > 0 else 1.0
    normalized = [m / max_mse for m in mse_by_layer]
    peak = int(np.argmax(mse_by_layer))

    return {
        "n_features": n_features,
        "mse_by_layer": mse_by_layer,
        "normalized_sensitivity": normalized,
        "most_sensitive_layer": peak,
    }


def _plot_results(all_results: dict[str, dict[int, dict[str, Any]]], save_dir: Path) -> None:
    # Plot 1: Peak layer vs feature count
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = {"tabpfn": "blue", "tabicl": "orange", "iltm": "green"}
    markers = {"tabpfn": "o", "tabicl": "s", "iltm": "^"}

    for model_name in MODEL_SPECS:
        peaks = []
        for d in FEATURE_COUNTS:
            if d in all_results[model_name]:
                peaks.append(all_results[model_name][d]["probing"]["peak_layer"])
            else:
                peaks.append(None)
        valid_d = [d for d, p in zip(FEATURE_COUNTS, peaks) if p is not None]
        valid_p = [p for p in peaks if p is not None]
        ax.plot(valid_d, valid_p, marker=markers[model_name], linewidth=2,
                color=colors[model_name], label=model_name.upper(), markersize=8)

    ax.set_xlabel("Number of Features (d)", fontsize=12)
    ax.set_ylabel("Peak Intermediary Layer", fontsize=12)
    ax.set_title("Phase 8D: Computation Depth vs Feature Complexity", fontsize=13)
    ax.set_xticks(FEATURE_COUNTS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_dir / "peak_layer_vs_features.png", dpi=180, bbox_inches="tight")
    fig.savefig(save_dir / "peak_layer_vs_features.pdf", bbox_inches="tight")
    plt.close(fig)

    # Plot 2: R² profiles per feature count (TabPFN only — most interesting)
    fig2, axes2 = plt.subplots(1, len(FEATURE_COUNTS), figsize=(5 * len(FEATURE_COUNTS), 5), sharey=True)
    for ax, d in zip(axes2, FEATURE_COUNTS, strict=True):
        for model_name in ["tabpfn", "tabicl"]:
            if d in all_results[model_name]:
                r2 = all_results[model_name][d]["probing"]["r2_by_layer"]
                layers = np.arange(len(r2))
                ax.plot(layers, r2, marker="o", linewidth=2, color=colors[model_name],
                        label=model_name.upper(), alpha=0.85)
        ax.set_title(f"d = {d}", fontsize=12)
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.25)
        if d == FEATURE_COUNTS[0]:
            ax.set_ylabel("Intermediary R²")
        ax.legend(fontsize=9)

    fig2.suptitle("Phase 8D: Layer R² Profiles by Feature Count", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(save_dir / "r2_profiles_by_features.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("Phase 8D: Feature Complexity Sweep")
    print("=" * 72)
    print(f"SEED={cfg.SEED}, feature counts={FEATURE_COUNTS}")

    all_results: dict[str, dict[int, dict[str, Any]]] = {}

    for model_name in MODEL_SPECS:
        all_results[model_name] = {}
        for d in FEATURE_COUNTS:
            print(f"\n--- {model_name} × d={d} ---")
            probing_result = _run_probing(model_name, d)
            causal_result = _run_causal(model_name, d)
            all_results[model_name][d] = {
                "probing": probing_result,
                "causal": causal_result,
            }
            print(f"  Probing peak: L{probing_result['peak_layer']} (R²={probing_result['peak_r2']:.4f})")
            if not causal_result.get("skipped"):
                print(f"  Causal peak:  L{causal_result['most_sensitive_layer']}")

    # Serialize with string keys for JSON
    json_results = {}
    for model_name, by_d in all_results.items():
        json_results[model_name] = {str(d): v for d, v in by_d.items()}

    json_path = RESULTS_DIR / f"results_seed{cfg.SEED}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    _plot_results(all_results, RESULTS_DIR)

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY: Peak layer by model × feature count")
    print("=" * 72)
    header = f"{'Model':<10}" + "".join(f"d={d:<8}" for d in FEATURE_COUNTS)
    print(header)
    for model_name in MODEL_SPECS:
        row = f"{model_name:<10}"
        for d in FEATURE_COUNTS:
            p = all_results[model_name][d]["probing"]["peak_layer"]
            row += f"L{p:<9}"
        print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
