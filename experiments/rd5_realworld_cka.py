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
from typing import Any

import numpy as np
from iltm import iLTMRegressor
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

from rd5_config import cfg
from src.data.real_world_datasets import (
    RealWorldDataset,
    load_california_housing,
    load_diabetes_sklearn,
    load_wine_quality,
)
from src.hooks.iltm_hooker import iLTMHookedModel
from src.hooks.tabicl_hooker import TabICLHookedModel
from src.hooks.tabpfn_hooker import TabPFNHookedModel

RESULTS_DIR = ROOT / "results" / "rd5" / "realworld_cka"


def _build_tabpfn() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _build_tabicl() -> TabICLRegressor:
    return TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)


def _build_iltm() -> iLTMRegressor:
    model = iLTMRegressor(device="cpu", n_ensemble=1, seed=cfg.SEED)
    model.n_ensemble = 1
    return model


def _load_realworld_datasets() -> list[RealWorldDataset]:
    datasets: list[RealWorldDataset] = []
    dataset_specs: list[tuple[str, Any, int, int]] = [
        (
            "california_housing",
            load_california_housing,
            cfg.N_TRAIN,
            cfg.N_TEST,
        ),
        (
            "diabetes",
            load_diabetes_sklearn,
            min(cfg.N_TRAIN, 300),
            min(cfg.N_TEST, 100),
        ),
        (
            "wine_quality",
            load_wine_quality,
            cfg.N_TRAIN,
            cfg.N_TEST,
        ),
    ]

    for display_name, loader_fn, n_train, n_test in dataset_specs:
        try:
            ds = loader_fn(n_train=n_train, n_test=n_test, random_seed=cfg.SEED)
            ds.name = display_name
            datasets.append(ds)
        except Exception as exc:
            print(f"[dataset:{display_name}] skip due to load error: {exc}")

    return datasets


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)

    numerator = np.linalg.norm(y.T @ x, ord="fro") ** 2
    x_norm = np.linalg.norm(x.T @ x, ord="fro")
    y_norm = np.linalg.norm(y.T @ y, ord="fro")
    denominator = x_norm * y_norm

    if denominator <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def compute_cka_matrix(layer_activations: list[np.ndarray]) -> np.ndarray:
    n_layers = len(layer_activations)
    matrix = np.zeros((n_layers, n_layers), dtype=np.float64)

    for i in range(n_layers):
        for j in range(n_layers):
            matrix[i, j] = compute_cka(layer_activations[i], layer_activations[j])
    return matrix


def _extract_activations(
    model_name: str,
    hooker: Any,
    cache: dict[str, Any],
    layer_idx: int,
) -> np.ndarray:
    if model_name == "tabpfn":
        return np.asarray(
            hooker.get_test_label_token(cache, layer_idx), dtype=np.float64
        )
    return np.asarray(hooker.get_layer_activations(cache, layer_idx), dtype=np.float64)


def _collect_layer_activations(
    dataset: RealWorldDataset,
    model_name: str,
) -> tuple[list[np.ndarray], list[int]]:
    if model_name == "tabpfn":
        model = _build_tabpfn()
        hooker_cls = TabPFNHookedModel
    elif model_name == "tabicl":
        model = _build_tabicl()
        hooker_cls = TabICLHookedModel
    elif model_name == "iltm":
        model = _build_iltm()
        hooker_cls = iLTMHookedModel
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(dataset.X_train, dataset.y_train)
    hooker = hooker_cls(model)
    _, cache = hooker.forward_with_cache(dataset.X_test)

    layer_indices = cfg.layer_indices(model_name)
    activations: list[np.ndarray] = []
    for layer_idx in layer_indices:
        activations.append(_extract_activations(model_name, hooker, cache, layer_idx))
    return activations, layer_indices


def _plot_cka_heatmaps(averaged: dict[str, Any], save_path: Path) -> None:
    model_order = ["tabpfn", "tabicl", "iltm"]
    model_titles = {"tabpfn": "TabPFN", "tabicl": "TabICL", "iltm": "iLTM"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, model_name in enumerate(model_order):
        ax = axes[idx]
        model_data = averaged.get(model_name)
        if model_data is None or "avg_cka_matrix" not in model_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=11)
            ax.set_title(f"{model_titles[model_name]} (failed)")
            ax.set_axis_off()
            continue

        cka_matrix = np.asarray(model_data["avg_cka_matrix"], dtype=np.float64)
        layer_indices = np.asarray(model_data["layer_indices"], dtype=np.int32)
        labels = [f"L{i}" for i in layer_indices.tolist()]

        im = ax.imshow(cka_matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(
            f"{model_titles[model_name]} ({cka_matrix.shape[0]}x{cka_matrix.shape[1]})"
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("RD5 Real-world CKA (Average across datasets)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = _load_realworld_datasets()
    model_order = ["tabpfn", "tabicl", "iltm"]

    print(
        f"QUICK_RUN={cfg.QUICK_RUN}, seed={cfg.SEED}, "
        f"n_train={cfg.N_TRAIN}, n_test={cfg.N_TEST}"
    )
    print(f"Datasets loaded: {[ds.name for ds in datasets]}")

    per_dataset_results: dict[str, dict[str, Any]] = {}
    matrices_by_model: dict[str, list[np.ndarray]] = {m: [] for m in model_order}
    layer_indices_by_model: dict[str, list[int]] = {}

    for dataset in datasets:
        per_dataset_results[dataset.name] = {}
        for model_name in model_order:
            try:
                print(f"[{dataset.name}] {model_name} CKA...")
                layer_acts, layer_indices = _collect_layer_activations(
                    dataset, model_name
                )
                cka_matrix = compute_cka_matrix(layer_acts)

                per_dataset_results[dataset.name][model_name] = {
                    "layer_indices": [int(i) for i in layer_indices],
                    "cka_matrix": cka_matrix.tolist(),
                }
                matrices_by_model[model_name].append(cka_matrix)
                layer_indices_by_model[model_name] = [int(i) for i in layer_indices]
            except Exception as exc:
                print(f"[{dataset.name}] {model_name} failed: {exc}")
                per_dataset_results[dataset.name][model_name] = {"error": str(exc)}

    averaged_results: dict[str, Any] = {}
    for model_name in model_order:
        mats = matrices_by_model[model_name]
        if not mats:
            averaged_results[model_name] = {"error": "No successful datasets"}
            continue

        avg_mat = np.mean(np.stack(mats, axis=0), axis=0)
        averaged_results[model_name] = {
            "layer_indices": layer_indices_by_model[model_name],
            "n_datasets_used": int(len(mats)),
            "avg_cka_matrix": avg_mat.tolist(),
        }

    payload: dict[str, Any] = {
        "quick_run": bool(cfg.QUICK_RUN),
        "seed": int(cfg.SEED),
        "n_train": int(cfg.N_TRAIN),
        "n_test": int(cfg.N_TEST),
        "datasets": per_dataset_results,
        "averaged_by_model": averaged_results,
    }

    json_path = RESULTS_DIR / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_path = RESULTS_DIR / "cka_heatmaps.png"
    _plot_cka_heatmaps(averaged_results, plot_path)

    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
