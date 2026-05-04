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
from src.probing.linear_probe import probe_layer

RESULTS_DIR = ROOT / "results" / "rd5" / "realworld_probing"


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


def _extract_activations(
    model_name: str,
    hooker: Any,
    cache: dict[str, Any],
    layer_idx: int,
) -> np.ndarray:
    if model_name == "tabpfn":
        return np.asarray(
            hooker.get_test_label_token(cache, layer_idx), dtype=np.float32
        )
    return np.asarray(hooker.get_layer_activations(cache, layer_idx), dtype=np.float32)


def _probe_dataset_with_model(
    dataset: RealWorldDataset,
    model_name: str,
) -> dict[str, Any]:
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
    # Extract activations from BOTH splits: train for probe fitting, test for evaluation
    _, cache_train = hooker.forward_with_cache(dataset.X_train)
    _, cache_test = hooker.forward_with_cache(dataset.X_test)

    layer_indices = cfg.layer_indices(model_name)
    y_train = np.asarray(dataset.y_train, dtype=np.float32).reshape(-1)
    y_test = np.asarray(dataset.y_test, dtype=np.float32).reshape(-1)
    r2_by_layer: list[float] = []

    for layer_idx in layer_indices:
        act_train = _extract_activations(model_name, hooker, cache_train, layer_idx)
        act_test = _extract_activations(model_name, hooker, cache_test, layer_idx)
        # Probe: fit on train activations, evaluate on test activations
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        ridge = Ridge(alpha=1.0)
        ridge.fit(act_train, y_train)
        pred = ridge.predict(act_test)
        r2 = float(r2_score(y_test, pred))
        r2_by_layer.append(r2)

    return {
        "layer_indices": [int(i) for i in layer_indices],
        "r2_by_layer": r2_by_layer,
        "best_layer": int(layer_indices[int(np.argmax(np.asarray(r2_by_layer)))]),
        "best_r2": float(np.max(np.asarray(r2_by_layer))),
    }


def _plot_results(results: dict[str, dict[str, Any]], save_path: Path) -> None:
    dataset_order = ["california_housing", "diabetes", "wine_quality"]
    model_order = ["tabpfn", "tabicl", "iltm"]
    model_title = {"tabpfn": "TabPFN", "tabicl": "TabICL", "iltm": "iLTM"}

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=True)

    for row_idx, dataset_name in enumerate(dataset_order):
        for col_idx, model_name in enumerate(model_order):
            ax = axes[row_idx, col_idx]
            model_result = results.get(dataset_name, {}).get(model_name)

            if model_result is None or "error" in model_result:
                ax.text(0.5, 0.5, "failed", ha="center", va="center", fontsize=11)
                ax.set_title(f"{dataset_name} x {model_title[model_name]}")
                ax.set_xlabel("Layer")
                if col_idx == 0:
                    ax.set_ylabel("R2")
                ax.grid(alpha=0.25)
                continue

            layers = np.asarray(model_result["layer_indices"], dtype=np.int32)
            r2 = np.asarray(model_result["r2_by_layer"], dtype=np.float64)

            ax.plot(layers, r2, marker="o", linewidth=2.0)
            ax.set_title(f"{dataset_name} x {model_title[model_name]}")
            ax.set_xlabel("Layer")
            if col_idx == 0:
                ax.set_ylabel("R2")
            ax.grid(alpha=0.25)

    fig.suptitle("RD5 Real-world Layer-wise Probing (Target y)")
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

    results: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        results[dataset.name] = {}
        for model_name in model_order:
            try:
                print(f"[{dataset.name}] {model_name} probing...")
                results[dataset.name][model_name] = _probe_dataset_with_model(
                    dataset=dataset,
                    model_name=model_name,
                )
            except Exception as exc:
                print(f"[{dataset.name}] {model_name} failed: {exc}")
                results[dataset.name][model_name] = {"error": str(exc)}

    payload: dict[str, Any] = {
        "quick_run": bool(cfg.QUICK_RUN),
        "seed": int(cfg.SEED),
        "n_train": int(cfg.N_TRAIN),
        "n_test": int(cfg.N_TEST),
        "datasets": results,
    }

    json_path = RESULTS_DIR / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_path = RESULTS_DIR / "r2_grid.png"
    _plot_results(results, plot_path)

    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
