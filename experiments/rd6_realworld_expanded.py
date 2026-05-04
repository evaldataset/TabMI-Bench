# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from iltm import iLTMRegressor
from sklearn.linear_model import Ridge
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from rd5_config import cfg
from src.data.real_world_datasets import (
    RealWorldDataset,
    load_abalone,
    load_adult_income,
    load_bank_marketing,
    load_bike_sharing,
    load_boston,
    load_breast_cancer,
    load_california_housing,
    load_concrete,
    load_credit_g,
    load_diabetes_sklearn,
    load_energy_efficiency,
    load_iris_binary,
    load_satellite,
    load_segment,
    load_vehicle,
    load_wine_quality,
)
from src.hooks.iltm_hooker import iLTMHookedModel
from src.hooks.tabicl_hooker import TabICLHookedModel
from src.hooks.tabpfn_hooker import TabPFNHookedModel

probe_layer: Any | None = None
try:
    from src.probing.linear_probe import probe_layer as _probe_layer

    probe_layer = _probe_layer
except Exception:
    probe_layer = None


RESULTS_DIR = ROOT / "results" / "rd6" / "realworld_expanded"
MODEL_ORDER = ["tabpfn", "tabicl", "iltm"]
MODEL_TITLE = {"tabpfn": "TabPFN", "tabicl": "TabICL", "iltm": "iLTM"}
QUICK_DATASETS = ["california_housing", "breast_cancer", "boston", "wine_quality"]


def _build_tabpfn() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _build_tabicl() -> TabICLRegressor:
    return TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)


def _build_iltm() -> iLTMRegressor:
    model = iLTMRegressor(device="cpu", n_ensemble=1, seed=cfg.SEED)
    model.n_ensemble = 1
    return model


def _extract_activations(
    model_name: str,
    hooker: Any,
    cache: dict[str, Any],
    layer_idx: int,
) -> np.ndarray:
    if model_name == "tabpfn":
        act = hooker.get_test_label_token(cache, layer_idx)
    else:
        act = hooker.get_layer_activations(cache, layer_idx)
    arr = np.asarray(act, dtype=np.float32)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _layer_r2(activations: np.ndarray, targets: np.ndarray) -> float:
    y = np.asarray(targets, dtype=np.float32).reshape(-1)
    x = np.asarray(activations, dtype=np.float32)

    if probe_layer is not None:
        result = probe_layer(
            activations=x,
            targets=y,
            complexities=[0],
            random_seed=cfg.SEED,
        )
        return float(result[0]["r2"])

    split = max(1, int(0.8 * x.shape[0]))
    rng = np.random.default_rng(cfg.SEED)
    indices = rng.permutation(x.shape[0])
    tr_idx = indices[:split]
    te_idx = indices[split:]
    if te_idx.size == 0:
        return float("nan")

    reg = Ridge(alpha=1.0)
    reg.fit(x[tr_idx], y[tr_idx])
    y_pred = reg.predict(x[te_idx])
    y_true = y[te_idx]
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _load_datasets() -> list[RealWorldDataset]:
    n_train = min(cfg.N_TRAIN, 200)
    n_test = min(cfg.N_TEST, 50)

    loader_specs: list[tuple[str, Callable[..., RealWorldDataset], int, int]] = [
        ("california_housing", load_california_housing, n_train, n_test),
        ("diabetes", load_diabetes_sklearn, min(n_train, 300), min(n_test, 100)),
        ("wine_quality", load_wine_quality, n_train, n_test),
        ("boston", load_boston, n_train, n_test),
        ("abalone", load_abalone, n_train, n_test),
        ("bike_sharing", load_bike_sharing, n_train, n_test),
        ("energy_efficiency", load_energy_efficiency, n_train, n_test),
        ("concrete", load_concrete, n_train, n_test),
        ("breast_cancer", load_breast_cancer, n_train, n_test),
        ("iris_binary", load_iris_binary, min(n_train, 80), min(n_test, 20)),
        ("adult_income", load_adult_income, n_train, n_test),
        ("satellite", load_satellite, n_train, n_test),
        ("bank_marketing", load_bank_marketing, n_train, n_test),
        ("credit_g", load_credit_g, n_train, n_test),
        ("segment", load_segment, n_train, n_test),
        ("vehicle", load_vehicle, n_train, n_test),
    ]

    if cfg.QUICK_RUN:
        selected_names = set(QUICK_DATASETS)
        loader_specs = [spec for spec in loader_specs if spec[0] in selected_names]

    loaded: list[RealWorldDataset] = []
    for name, loader_fn, ds_train, ds_test in loader_specs:
        try:
            ds = loader_fn(n_train=ds_train, n_test=ds_test, random_seed=cfg.SEED)
            ds.name = name
            loaded.append(ds)
        except Exception as exc:
            print(f"[dataset:{name}] warning: skipped due to load error: {exc}")

    return loaded


def _probe_dataset_with_model(
    dataset: RealWorldDataset, model_name: str
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
        raise ValueError(f"unknown model: {model_name}")

    model.fit(dataset.X_train, dataset.y_train)
    hooker = hooker_cls(model)
    _, cache = hooker.forward_with_cache(dataset.X_test)

    layers = cfg.layer_indices(model_name)
    y_test = np.asarray(dataset.y_test, dtype=np.float32).reshape(-1)

    r2_by_layer: list[float] = []
    for layer_idx in layers:
        act = _extract_activations(model_name, hooker, cache, layer_idx)
        r2_by_layer.append(_layer_r2(act, y_test))

    r2_arr = np.asarray(r2_by_layer, dtype=np.float64)
    best_idx = int(np.nanargmax(r2_arr)) if np.isfinite(r2_arr).any() else 0
    return {
        "layer_indices": [int(v) for v in layers],
        "r2_by_layer": [float(v) for v in r2_by_layer],
        "best_layer": int(layers[best_idx]),
        "best_r2": float(r2_arr[best_idx]),
    }


def _plot_results(results: dict[str, dict[str, Any]], save_path: Path) -> None:
    dataset_names = list(results.keys())
    n_rows = max(1, len(dataset_names))
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.0 * n_rows), squeeze=False)

    for row_idx, dataset_name in enumerate(dataset_names):
        ax = axes[row_idx, 0]
        for model_name in MODEL_ORDER:
            model_result = results.get(dataset_name, {}).get(model_name)
            if model_result is None or "error" in model_result:
                continue

            layers = np.asarray(model_result["layer_indices"], dtype=np.int32)
            r2 = np.asarray(model_result["r2_by_layer"], dtype=np.float64)
            ax.plot(
                layers,
                r2,
                marker="o",
                linewidth=1.8,
                label=MODEL_TITLE[model_name],
            )

        ax.set_title(dataset_name)
        ax.set_xlabel("Layer")
        ax.set_ylabel("R2")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle("RD6 Expanded Real-world Probing: R2 by Layer")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = _load_datasets()
    print(
        f"QUICK_RUN={cfg.QUICK_RUN}, seed={cfg.SEED}, "
        f"n_train={min(cfg.N_TRAIN, 200)}, n_test={min(cfg.N_TEST, 50)}"
    )
    print(f"Datasets loaded ({len(datasets)}): {[ds.name for ds in datasets]}")

    results: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        results[dataset.name] = {
            "task_type": dataset.task_type,
            "n_train": int(dataset.n_train),
            "n_test": int(dataset.n_test),
            "n_features": int(dataset.n_features),
            "models": {},
        }

        for model_name in MODEL_ORDER:
            try:
                print(f"[{dataset.name}] {model_name} probing...")
                results[dataset.name]["models"][model_name] = _probe_dataset_with_model(
                    dataset=dataset,
                    model_name=model_name,
                )
            except Exception as exc:
                print(f"[{dataset.name}] {model_name} failed: {exc}")
                results[dataset.name]["models"][model_name] = {"error": str(exc)}

    payload: dict[str, Any] = {
        "quick_run": bool(cfg.QUICK_RUN),
        "seed": int(cfg.SEED),
        "n_train": int(min(cfg.N_TRAIN, 200)),
        "n_test": int(min(cfg.N_TEST, 50)),
        "datasets": results,
    }

    json_path = RESULTS_DIR / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    flat_results = {
        ds_name: {
            model_name: ds_payload["models"].get(model_name, {})
            for model_name in MODEL_ORDER
        }
        for ds_name, ds_payload in results.items()
    }
    plot_path = RESULTS_DIR / "r2_by_dataset.png"
    _plot_results(flat_results, plot_path)

    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
