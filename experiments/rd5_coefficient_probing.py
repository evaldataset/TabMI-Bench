# pyright: reportMissingImports=false
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportImplicitStringConcatenation=false

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from iltm import iLTMRegressor
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

from src.hooks.iltm_hooker import iLTMHookedModel
from src.hooks.tabicl_hooker import TabICLHookedModel
from src.hooks.tabpfn_hooker import TabPFNHookedModel
from src.probing.linear_probe import probe_layer
from rd5_config import cfg


QUICK_RUN = cfg.QUICK_RUN

SEED = cfg.SEED

N_TRAIN = cfg.N_TRAIN

N_TEST = cfg.N_TEST


@dataclass
class DatasetSpec:
    alpha: float
    beta: float
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def generate_datasets(coef_pairs: list[tuple[int, int]]) -> list[DatasetSpec]:
    rng = np.random.default_rng(SEED)
    x1_train = rng.normal(size=N_TRAIN).astype(np.float32)
    x2_train = rng.normal(size=N_TRAIN).astype(np.float32)
    x1_test = rng.normal(size=N_TEST).astype(np.float32)
    x2_test = rng.normal(size=N_TEST).astype(np.float32)

    X_train = np.column_stack([x1_train, x2_train]).astype(np.float32)
    X_test = np.column_stack([x1_test, x2_test]).astype(np.float32)

    datasets: list[DatasetSpec] = []
    for alpha, beta in coef_pairs:
        y_train = (alpha * x1_train + beta * x2_train).astype(np.float32)
        y_test = (alpha * x1_test + beta * x2_test).astype(np.float32)
        datasets.append(
            DatasetSpec(
                alpha=float(alpha),
                beta=float(beta),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
        )
    return datasets


def _build_tabpfn() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _build_tabicl() -> TabICLRegressor:
    return TabICLRegressor(device=cfg.DEVICE, random_state=SEED)


def _build_iltm() -> iLTMRegressor:
    model = iLTMRegressor(device="cpu", n_ensemble=1, seed=SEED)
    model.n_ensemble = 1
    return model


def run_model_probing(
    model_name: str,
    datasets: list[DatasetSpec],
) -> dict[str, Any]:
    pooled_activations: dict[int, list[np.ndarray]] = {}
    pooled_alphas: list[np.ndarray] = []
    pooled_betas: list[np.ndarray] = []
    all_layer_indices: list[int] | None = None

    for idx, ds in enumerate(datasets, start=1):
        if model_name == "tabpfn":
            model = _build_tabpfn()
            model.fit(ds.X_train, ds.y_train)
            hooker = TabPFNHookedModel(model)
        elif model_name == "tabicl":
            model = _build_tabicl()
            model.fit(ds.X_train, ds.y_train)
            hooker = TabICLHookedModel(model)
        elif model_name == "iltm":
            model = _build_iltm()
            model.fit(ds.X_train, ds.y_train)
            hooker = iLTMHookedModel(model)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        _, cache = hooker.forward_with_cache(ds.X_test)

        num_layers = int(getattr(hooker, "num_layers", getattr(hooker, "_n_layers")))
        if model_name == "iltm":
            layer_indices = list(range(max(num_layers - 1, 0)))
        else:
            layer_indices = list(range(num_layers))

        if all_layer_indices is None:
            all_layer_indices = layer_indices
            pooled_activations = {layer_idx: [] for layer_idx in layer_indices}

        for layer_idx in layer_indices:
            if hasattr(hooker, "get_layer_activations"):
                act = np.asarray(
                    hooker.get_layer_activations(cache, layer_idx),
                    dtype=np.float32,
                )
            else:
                act = np.asarray(
                    hooker.get_test_label_token(cache, layer_idx), dtype=np.float32
                )
            pooled_activations[layer_idx].append(act)

        pooled_alphas.append(np.full(ds.X_test.shape[0], ds.alpha, dtype=np.float32))
        pooled_betas.append(np.full(ds.X_test.shape[0], ds.beta, dtype=np.float32))

        print(
            f"[{model_name}] dataset {idx}/{len(datasets)} done "
            f"(alpha={ds.alpha:.1f}, beta={ds.beta:.1f})"
        )

    if all_layer_indices is None:
        raise RuntimeError(f"No datasets were processed for model={model_name}")

    targets_alpha = np.concatenate(pooled_alphas)
    targets_beta = np.concatenate(pooled_betas)

    alpha_r2_by_layer: list[float] = []
    beta_r2_by_layer: list[float] = []
    for layer_idx in all_layer_indices:
        layer_acts = np.vstack(pooled_activations[layer_idx])
        alpha_probe = probe_layer(
            layer_acts, targets_alpha, complexities=[0], random_seed=SEED
        )
        beta_probe = probe_layer(
            layer_acts, targets_beta, complexities=[0], random_seed=SEED
        )
        alpha_r2_by_layer.append(float(alpha_probe[0]["r2"]))
        beta_r2_by_layer.append(float(beta_probe[0]["r2"]))

    peak_layer_alpha = int(np.argmax(np.asarray(alpha_r2_by_layer)))
    peak_layer_beta = int(np.argmax(np.asarray(beta_r2_by_layer)))

    return {
        "alpha_r2_by_layer": alpha_r2_by_layer,
        "beta_r2_by_layer": beta_r2_by_layer,
        "n_layers": len(all_layer_indices),
        "peak_layer_alpha": peak_layer_alpha,
        "peak_layer_beta": peak_layer_beta,
    }


def _normalized_layer_axis(n_layers: int) -> np.ndarray:
    if n_layers <= 1:
        return np.asarray([0.0], dtype=np.float32)
    return np.linspace(0.0, 1.0, num=n_layers, dtype=np.float32)


def plot_model_panels(results: dict[str, dict[str, Any]], save_path: Path) -> None:
    model_order = ["tabpfn", "tabicl", "iltm"]
    titles = {"tabpfn": "TabPFN", "tabicl": "TabICL", "iltm": "iLTM"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, model_key in zip(axes, model_order, strict=True):
        alpha = np.asarray(results[model_key]["alpha_r2_by_layer"])
        beta = np.asarray(results[model_key]["beta_r2_by_layer"])
        layers = np.arange(alpha.shape[0])

        ax.plot(layers, alpha, marker="o", linewidth=2.0, label="R2(alpha)")
        ax.plot(layers, beta, marker="s", linewidth=2.0, label="R2(beta)")
        ax.set_title(titles[model_key])
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.25)
        if model_key == "tabpfn":
            ax.set_ylabel("R2")
        ax.legend(loc="best")

    fig.suptitle("RD5 Coefficient Probing: Per-Model Layer-wise R2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_alpha(results: dict[str, dict[str, Any]], save_path: Path) -> None:
    model_styles = {
        "tabpfn": ("TabPFN", "o"),
        "tabicl": ("TabICL", "s"),
        "iltm": ("iLTM", "^"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_key in ["tabpfn", "tabicl", "iltm"]:
        alpha = np.asarray(results[model_key]["alpha_r2_by_layer"])
        x_norm = _normalized_layer_axis(len(alpha))
        label, marker = model_styles[model_key]
        ax.plot(x_norm, alpha, marker=marker, linewidth=2.0, label=label)

    ax.set_title("RD5 Coefficient Probing: Alpha R2 Overlay (Normalized Layer Axis)")
    ax.set_xlabel("Normalized Layer Position")
    ax.set_ylabel("R2(alpha)")
    ax.grid(alpha=0.25)
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_hypotheses(
    results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    tabicl_alpha = np.asarray(results["tabicl"]["alpha_r2_by_layer"])
    tabicl_beta = np.asarray(results["tabicl"]["beta_r2_by_layer"])
    tabpfn_peak_mean = 0.5 * (
        results["tabpfn"]["peak_layer_alpha"] + results["tabpfn"]["peak_layer_beta"]
    )
    tabicl_peak_mean = 0.5 * (
        results["tabicl"]["peak_layer_alpha"] + results["tabicl"]["peak_layer_beta"]
    )

    tabicl_combined = np.maximum(tabicl_alpha, tabicl_beta)
    tabicl_peak = float(np.max(tabicl_combined))
    high_band = np.where(tabicl_combined >= 0.9 * tabicl_peak)[0]
    h1_supported = bool(tabicl_peak > 0.10 and high_band.size >= 1)

    h2_supported = bool(tabicl_peak_mean < 5.0 and tabpfn_peak_mean >= 5.0)

    return {
        "H1": {
            "description": "TabICL has core computation zone",
            "supported": h1_supported,
            "evidence": (
                f"TabICL peak max R2={tabicl_peak:.3f}; high-R2 band (>=90% peak) "
                f"layers={high_band.tolist()}"
            ),
        },
        "H2": {
            "description": "TabICL zone is earlier than TabPFN L5-8",
            "supported": h2_supported,
            "evidence": (
                f"TabICL mean peak layer={tabicl_peak_mean:.2f}, "
                f"TabPFN mean peak layer={tabpfn_peak_mean:.2f}"
            ),
        },
    }


def main() -> int:
    output_dir = ROOT / "results" / "rd5" / "coefficient_probing"
    os.makedirs(output_dir, exist_ok=True)

    full_values = [1, 2, 3, 4, 5]
    quick_values = [1, 3, 5]
    if QUICK_RUN:
        base_values = quick_values
    else:
        base_values = full_values

    tabpfn_pairs = [(a, b) for a in base_values for b in base_values]
    tabicl_pairs = [(a, b) for a in base_values for b in base_values]
    if QUICK_RUN:
        iltm_pairs = [(a, 3) for a in [1, 3, 5]]
    else:
        iltm_pairs = [(a, b) for a in full_values for b in full_values]

    tabpfn_datasets = generate_datasets(tabpfn_pairs)
    tabicl_datasets = generate_datasets(tabicl_pairs)
    iltm_datasets = generate_datasets(iltm_pairs)

    print(f"QUICK_RUN={QUICK_RUN}, seed={SEED}")
    print(
        f"Datasets -> TabPFN: {len(tabpfn_datasets)}, "
        f"TabICL: {len(tabicl_datasets)}, iLTM: {len(iltm_datasets)}"
    )

    results: dict[str, dict[str, Any]] = {}
    results["tabpfn"] = run_model_probing("tabpfn", tabpfn_datasets)
    results["tabicl"] = run_model_probing("tabicl", tabicl_datasets)
    results["iltm"] = run_model_probing("iltm", iltm_datasets)
    hypotheses = evaluate_hypotheses(results)

    payload: dict[str, Any] = {
        "tabpfn": results["tabpfn"],
        "tabicl": results["tabicl"],
        "iltm": results["iltm"],
        "hypotheses": hypotheses,
    }

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_model_panels(results, output_dir / "r2_comparison.png")
    plot_overlay_alpha(results, output_dir / "r2_overlay.png")

    print(f"Saved: {output_dir / 'results.json'}")
    print(f"Saved: {output_dir / 'r2_comparison.png'}")
    print(f"Saved: {output_dir / 'r2_overlay.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
