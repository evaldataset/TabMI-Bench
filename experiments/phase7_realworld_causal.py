# pyright: reportMissingImports=false
from __future__ import annotations

import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from iltm import iLTMRegressor
from sklearn.model_selection import train_test_split
# StandardScaler removed — loaders handle scaling internally
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import real_world_datasets as rw_datasets
from src.hooks.iltm_hooker import iLTMHookedModel

_maybe_load_diabetes = getattr(rw_datasets, "load_diabetes", None)
if callable(_maybe_load_diabetes):
    diabetes_sig = inspect.signature(_maybe_load_diabetes)
    if "random_seed" in diabetes_sig.parameters:
        load_diabetes = _maybe_load_diabetes
    else:
        load_diabetes = rw_datasets.load_diabetes_sklearn
else:
    load_diabetes = rw_datasets.load_diabetes_sklearn


SEED = int(os.getenv("SEED", "42"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
QUICK_RUN = os.getenv("QUICK_RUN", "0") == "1"
NOISE_SCALE = 0.5

RESULTS_DIR = ROOT / "results" / "phase7" / "realworld_causal"
JSON_PATH = RESULTS_DIR / f"realworld_causal_seed{SEED}.json"


def _extract_xy(
    loader: Callable[..., Any], random_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract train/test data, preserving the loader's existing split.

    Returns (X_train, X_test, y_train, y_test) without re-splitting,
    so that test data is never used for fitting or direction extraction.
    """
    raw = loader(random_seed=random_seed)
    if isinstance(raw, tuple) and len(raw) == 2:
        # Raw (X, y) tuple — do a proper split here
        X_all = np.asarray(raw[0], dtype=np.float64)
        y_all = np.asarray(raw[1], dtype=np.float64).reshape(-1)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=random_seed,
        )
        return X_train, X_test, y_train, y_test
    if hasattr(raw, "X_train") and hasattr(raw, "X_test"):
        ds_obj: Any = raw
        return (
            np.asarray(ds_obj.X_train, dtype=np.float64),
            np.asarray(ds_obj.X_test, dtype=np.float64),
            np.asarray(ds_obj.y_train, dtype=np.float64),
            np.asarray(ds_obj.y_test, dtype=np.float64),
        )
    raise TypeError(f"Unsupported dataset type: {type(raw)}")


def _prepare_dataset(
    loader: Callable[..., Any], name: str
) -> dict[str, np.ndarray | str]:
    try:
        X_train, X_test, y_train, y_test = _extract_xy(loader=loader, random_seed=SEED)
    except Exception as e:
        print(f"  [{name}] SKIPPED: {e}")
        return {}

    if QUICK_RUN and X_train.shape[0] > 100:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(X_train.shape[0], size=100, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        idx_te = rng.choice(X_test.shape[0], size=min(20, X_test.shape[0]), replace=False)
        X_test = X_test[idx_te]
        y_test = y_test[idx_te]

    # NOTE: loaders from real_world_datasets.py already apply StandardScaler
    # via _prepare_regression_dataset(). Do NOT re-scale here to avoid
    # double-scaling artifacts.
    return {
        "name": name,
        "X_train": np.asarray(X_train, dtype=np.float64),
        "X_test": np.asarray(X_test, dtype=np.float64),
        "y_train": np.asarray(y_train, dtype=np.float64),
        "y_test": np.asarray(y_test, dtype=np.float64),
    }


def _noising_sweep_tabpfn(
    model: TabPFNRegressor,
    X_test: np.ndarray,
    noise_scale: float,
) -> dict[str, Any]:
    model_impl: Any = model.model_
    layers: Any = model_impl.transformer_encoder.layers
    preds_clean = np.asarray(model.predict(X_test), dtype=np.float64)
    mse_increase: list[float] = []

    for layer_idx in range(len(layers)):
        act_stds: list[float] = []

        def stats_hook(
            module: torch.nn.Module, input: Any, output: torch.Tensor
        ) -> None:  # noqa: A002
            del module, input
            act_stds.append(float(output.std().item()))

        h_stats = layers[layer_idx].register_forward_hook(stats_hook)
        _ = model.predict(X_test)
        h_stats.remove()

        act_std = float(np.mean(act_stds)) if act_stds else 1.0
        layer_rng = np.random.default_rng(SEED + layer_idx)

        def noise_hook(
            module: torch.nn.Module,
            input: Any,  # noqa: A002
            output: torch.Tensor,
        ) -> torch.Tensor:
            del module, input
            noise = torch.from_numpy(
                layer_rng.normal(0.0, noise_scale * act_std, size=output.shape).astype(
                    np.float32
                )
            ).to(output.device)
            return output + noise

        h_noise = layers[layer_idx].register_forward_hook(noise_hook)
        preds_noised = np.asarray(model.predict(X_test), dtype=np.float64)
        h_noise.remove()

        mse_increase.append(float(np.mean((preds_noised - preds_clean) ** 2)))

    max_mse = max(mse_increase) if max(mse_increase) > 0 else 1.0
    sensitivity = [float(v / max_mse) for v in mse_increase]
    return {
        "noise_scale": float(noise_scale),
        "mse_increase_by_layer": [float(v) for v in mse_increase],
        "normalized_sensitivity": sensitivity,
        "peak_layer": int(np.argmax(sensitivity)),
    }


def _noising_sweep_tabicl(
    model: TabICLRegressor,
    X_test: np.ndarray,
    noise_scale: float,
) -> dict[str, Any]:
    blocks: Any = model.model_.icl_predictor.tf_icl.blocks
    preds_clean = np.asarray(model.predict(X_test), dtype=np.float64)
    mse_increase: list[float] = []

    for layer_idx in range(len(blocks)):
        act_stds: list[float] = []

        def stats_hook(
            module: torch.nn.Module, input: Any, output: torch.Tensor
        ) -> None:  # noqa: A002
            del module, input
            act_stds.append(float(output.std().item()))

        h_stats = blocks[layer_idx].register_forward_hook(stats_hook)
        _ = model.predict(X_test)
        h_stats.remove()

        act_std = float(np.mean(act_stds)) if act_stds else 1.0
        layer_rng = np.random.default_rng(SEED + layer_idx)

        def noise_hook(
            module: torch.nn.Module,
            input: Any,  # noqa: A002
            output: torch.Tensor,
        ) -> torch.Tensor:
            del module, input
            noise = torch.from_numpy(
                layer_rng.normal(0.0, noise_scale * act_std, size=output.shape).astype(
                    np.float32
                )
            ).to(output.device)
            return output + noise

        h_noise = blocks[layer_idx].register_forward_hook(noise_hook)
        preds_noised = np.asarray(model.predict(X_test), dtype=np.float64)
        h_noise.remove()

        mse_increase.append(float(np.mean((preds_noised - preds_clean) ** 2)))

    max_mse = max(mse_increase) if max(mse_increase) > 0 else 1.0
    sensitivity = [float(v / max_mse) for v in mse_increase]
    return {
        "noise_scale": float(noise_scale),
        "mse_increase_by_layer": [float(v) for v in mse_increase],
        "normalized_sensitivity": sensitivity,
        "peak_layer": int(np.argmax(sensitivity)),
    }


def _noising_sweep_iltm(
    model: iLTMRegressor,
    X_test: np.ndarray,
    noise_scale: float,
) -> dict[str, Any]:
    hooker = iLTMHookedModel(model)
    predictor = model.predictors_[0]
    main_network: Any = predictor["main_network"]
    n_layers = len(main_network)

    preds_clean_np, cache = hooker.forward_with_cache(X_test)
    preds_clean = np.asarray(preds_clean_np, dtype=np.float64).ravel()
    mse_increase: list[float] = []

    device = model.device
    model._move_predictor_to_device(predictor, device=device)

    with torch.no_grad():
        base_tensor = torch.from_numpy(cache["layers"][0]).to(device)
        act_stds_list: list[float] = []
        x_tmp = base_tensor.clone()
        residual = x_tmp
        for n, layer in enumerate(main_network):
            if n % 2 == 0:
                residual = x_tmp
            x_tmp = layer.to(device)(x_tmp)
            if n % 2 == 1 and n != n_layers - 1:
                x_tmp = x_tmp + residual
            if n != n_layers - 1:
                x_tmp = torch.relu(x_tmp)
            act_stds_list.append(float(x_tmp.std().item()))

    for layer_idx in range(n_layers):
        act_std = act_stds_list[layer_idx] if act_stds_list else 1.0
        layer_rng = np.random.default_rng(SEED + layer_idx)

        with torch.no_grad():
            x = torch.from_numpy(cache["layers"][0]).to(device)
            residual = x
            for n, layer in enumerate(main_network):
                if n % 2 == 0:
                    residual = x
                x = layer.to(device)(x)
                if n == layer_idx:
                    noise = torch.from_numpy(
                        layer_rng.normal(
                            0.0, noise_scale * act_std, size=x.shape
                        ).astype(np.float32)
                    ).to(device)
                    x = x + noise
                if n % 2 == 1 and n != n_layers - 1:
                    x = x + residual
                if n != n_layers - 1:
                    x = torch.relu(x)
            preds_noised = x.squeeze(-1).detach().cpu().numpy()

        preds_noised = np.asarray(preds_noised, dtype=np.float64).ravel()
        if len(preds_noised) > len(preds_clean):
            preds_noised = preds_noised[: len(preds_clean)]
        elif len(preds_noised) < len(preds_clean):
            preds_clean = preds_clean[: len(preds_noised)]
        mse_increase.append(float(np.mean((preds_noised - preds_clean) ** 2)))

    model._move_predictor_to_cpu(predictor)

    max_mse = max(mse_increase) if max(mse_increase) > 0 else 1.0
    sensitivity = [float(v / max_mse) for v in mse_increase]
    return {
        "noise_scale": float(noise_scale),
        "mse_increase_by_layer": [float(v) for v in mse_increase],
        "normalized_sensitivity": sensitivity,
        "peak_layer": int(np.argmax(sensitivity)),
    }


def _run_dataset(dataset: dict[str, np.ndarray | str]) -> dict[str, Any]:
    X_train = np.asarray(dataset["X_train"], dtype=np.float64)
    X_test = np.asarray(dataset["X_test"], dtype=np.float64)
    y_train = np.asarray(dataset["y_train"], dtype=np.float64)

    print("  [tabpfn] fit + noising")
    tabpfn = TabPFNRegressor(device=DEVICE, model_path="tabpfn-v2-regressor.ckpt")
    tabpfn.fit(X_train, y_train)
    tabpfn_patching = _noising_sweep_tabpfn(tabpfn, X_test, NOISE_SCALE)

    print("  [tabicl] fit + noising")
    tabicl = TabICLRegressor(device=DEVICE, random_state=SEED)
    tabicl.fit(X_train, y_train)
    tabicl_patching = _noising_sweep_tabicl(tabicl, X_test, NOISE_SCALE)

    print("  [iltm] fit + noising")
    iltm = iLTMRegressor(device=DEVICE, n_ensemble=1, seed=SEED)
    iltm.fit(X_train, y_train)
    iltm_patching = _noising_sweep_iltm(iltm, X_test, NOISE_SCALE)

    return {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "patching": {
            "tabpfn": tabpfn_patching,
            "tabicl": tabicl_patching,
            "iltm": iltm_patching,
            "expected_peak": {"tabpfn": 5, "tabicl": 0, "iltm": 2},
            "matches_synthetic": {
                "tabpfn_peak_is_l5": bool(tabpfn_patching["peak_layer"] == 5),
                "tabicl_peak_is_l0": bool(tabicl_patching["peak_layer"] == 0),
                "iltm_peak_is_l2": bool(iltm_patching["peak_layer"] == 2),
            },
        },
    }


def main() -> int:
    start_t = time.perf_counter()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 11 datasets matching paper Section 3 ("11 OpenML datasets")
    all_dataset_loaders: list[tuple[str, Callable[..., Any]]] = [
        ("california_housing", rw_datasets.load_california_housing),
        ("diabetes", load_diabetes),
        ("bike_sharing", rw_datasets.load_bike_sharing),
        ("wine_quality", rw_datasets.load_wine_quality),
        ("abalone", rw_datasets.load_abalone),
        ("boston", rw_datasets.load_boston),
        ("energy_efficiency", rw_datasets.load_energy_efficiency),
        ("breast_cancer", rw_datasets.load_breast_cancer),
        ("iris_binary", rw_datasets.load_iris_binary),
        ("adult_income", rw_datasets.load_adult_income),
        ("credit_g", rw_datasets.load_credit_g),
    ]

    dataset_loaders = all_dataset_loaders
    if QUICK_RUN:
        quick_names = {"california_housing", "diabetes", "wine_quality"}
        dataset_loaders = [it for it in all_dataset_loaders if it[0] in quick_names]

    print("=" * 72)
    print("Phase 7 real-world noising-based causal tracing")
    print("=" * 72)
    print(f"seed={SEED}, device={DEVICE}, quick_run={QUICK_RUN}")
    print(f"datasets={len(dataset_loaders)}")

    all_results: dict[str, Any] = {}
    for idx, (name, loader) in enumerate(dataset_loaders, start=1):
        ds_t0 = time.perf_counter()
        print(f"[{idx}/{len(dataset_loaders)}] [{name}] preparing dataset...")
        dataset = _prepare_dataset(loader=loader, name=name)
        if not dataset:
            continue
        print(f"[{idx}/{len(dataset_loaders)}] [{name}] running models...")
        all_results[name] = _run_dataset(dataset)
        print(
            f"[{idx}/{len(dataset_loaders)}] [{name}] done "
            f"({time.perf_counter() - ds_t0:.1f}s)"
        )

    payload = {
        "quick_run": bool(QUICK_RUN),
        "seed": int(SEED),
        "device": DEVICE,
        "noise_scale": float(NOISE_SCALE),
        "datasets": all_results,
    }

    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    total_sec = time.perf_counter() - start_t
    print(f"Saved: {JSON_PATH}")
    print(f"Total runtime: {total_sec:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
