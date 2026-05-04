# pyright: reportMissingImports=false
from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rd5_config import cfg
from src.data import real_world_datasets as rw_datasets
from src.data.real_world_datasets import (
    load_abalone,
    load_bike_sharing,
    load_california_housing,
    load_concrete,
    load_wine_quality,
)
from src.hooks.tabicl_hooker import TabICLHookedModel
from src.hooks.tabicl_steering import TabICLSteeringVector
from src.hooks.tabpfn_hooker import TabPFNHookedModel
from src.hooks.steering_vector import TabPFNSteeringVector

_maybe_load_diabetes = getattr(rw_datasets, "load_diabetes", None)
if callable(_maybe_load_diabetes):
    diabetes_sig = inspect.signature(_maybe_load_diabetes)
    if "random_seed" in diabetes_sig.parameters:
        load_diabetes = _maybe_load_diabetes
    else:
        load_diabetes = rw_datasets.load_diabetes_sklearn
else:
    load_diabetes = rw_datasets.load_diabetes_sklearn


RESULTS_DIR = ROOT / "results" / "neurips"
JSON_PATH = RESULTS_DIR / "c4_realworld_causal.json"
PLOT_PATH = RESULTS_DIR / "c4_realworld_causal.png"

NOISE_SCALE = 0.5
TABPFN_STEER_LAYER = 6
TABICL_STEER_LAYER = 10
LAMBDA_VALUES = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def _safe_abs_pearson(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape or x_arr.size < 2:
        return 0.0
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return 0.0
    corr = float(np.corrcoef(x_arr, y_arr)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(abs(corr))


def _normalize(v: np.ndarray) -> np.ndarray:
    vec = np.asarray(v, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if np.isclose(norm, 0.0, atol=1e-12):
        raise ValueError("Direction has near-zero norm.")
    return vec / norm


def _extract_xy(
    loader: Callable[..., Any], random_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    raw = loader(random_seed=random_seed)
    if isinstance(raw, tuple) and len(raw) == 2:
        return (
            np.asarray(raw[0], dtype=np.float64),
            np.asarray(raw[1], dtype=np.float64).reshape(-1),
        )
    if hasattr(raw, "X_train") and hasattr(raw, "X_test"):
        ds_obj: Any = raw
        x_all = np.concatenate(
            [np.asarray(ds_obj.X_train), np.asarray(ds_obj.X_test)], axis=0
        )
        y_all = np.concatenate(
            [np.asarray(ds_obj.y_train), np.asarray(ds_obj.y_test)], axis=0
        )
        return np.asarray(x_all, dtype=np.float64), np.asarray(y_all, dtype=np.float64)
    raise TypeError(f"Unsupported dataset type: {type(raw)}")


def _prepare_dataset(
    loader: Callable[..., Any], name: str
) -> dict[str, np.ndarray | str]:
    x_all, y_all = _extract_xy(loader=loader, random_seed=cfg.SEED)

    if cfg.QUICK_RUN and x_all.shape[0] > 600:
        rng = np.random.default_rng(cfg.SEED)
        idx = rng.choice(x_all.shape[0], size=600, replace=False)
        x_all = x_all[idx]
        y_all = y_all[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=0.2,
        random_state=cfg.SEED,
    )
    scaler = StandardScaler()
    X_train_scaled = np.asarray(scaler.fit_transform(X_train), dtype=np.float64)
    X_test_scaled = np.asarray(scaler.transform(X_test), dtype=np.float64)

    return {
        "name": name,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
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
            act_stds.append(float(output.std().item()))

        h_stats = layers[layer_idx].register_forward_hook(stats_hook)
        _ = model.predict(X_test)
        h_stats.remove()

        act_std = float(np.mean(act_stds)) if act_stds else 1.0
        layer_rng = np.random.default_rng(cfg.SEED + layer_idx)

        def noise_hook(
            module: torch.nn.Module,
            input: Any,  # noqa: A002
            output: torch.Tensor,
        ) -> torch.Tensor:
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
            act_stds.append(float(output.std().item()))

        h_stats = blocks[layer_idx].register_forward_hook(stats_hook)
        _ = model.predict(X_test)
        h_stats.remove()

        act_std = float(np.mean(act_stds)) if act_stds else 1.0
        layer_rng = np.random.default_rng(cfg.SEED + layer_idx)

        def noise_hook(
            module: torch.nn.Module,
            input: Any,  # noqa: A002
            output: torch.Tensor,
        ) -> torch.Tensor:
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


def _steering_metrics(
    model_name: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    layer_idx: int,
) -> dict[str, Any]:
    q25, q75 = np.quantile(y_test, [0.25, 0.75])
    high_mask = y_test >= q75
    low_mask = y_test <= q25
    if int(high_mask.sum()) < 2 or int(low_mask.sum()) < 2:
        raise ValueError("Insufficient quartile samples for steering direction.")

    if model_name == "tabpfn":
        hooker = TabPFNHookedModel(model)
        _, cache = hooker.forward_with_cache(X_test)
        acts = np.asarray(
            hooker.get_test_label_token(cache, layer_idx), dtype=np.float64
        )
        steerer = TabPFNSteeringVector(model)
    else:
        hooker = TabICLHookedModel(model)
        _, cache = hooker.forward_with_cache(X_test)
        acts = np.asarray(
            hooker.get_layer_activations(cache, layer_idx), dtype=np.float64
        )
        steerer = TabICLSteeringVector(model)

    direction = _normalize(acts[high_mask].mean(axis=0) - acts[low_mask].mean(axis=0))
    rng = np.random.default_rng(
        cfg.SEED + layer_idx + (0 if model_name == "tabpfn" else 1000)
    )
    random_direction = _normalize(rng.normal(0.0, 1.0, size=direction.shape[0]))

    preds_base = np.asarray(model.predict(X_test), dtype=np.float64)
    causal_shifts: list[float] = []
    random_shifts: list[float] = []

    for lambda_val in LAMBDA_VALUES:
        if model_name == "tabpfn":
            preds_causal = np.asarray(
                steerer.steer(
                    X_test, layer=layer_idx, direction=direction, lambda_val=lambda_val
                ),
                dtype=np.float64,
            )
            preds_random = np.asarray(
                steerer.steer(
                    X_test,
                    layer=layer_idx,
                    direction=random_direction,
                    lambda_val=lambda_val,
                ),
                dtype=np.float64,
            )
        else:
            preds_causal = np.asarray(
                steerer.steer(
                    X_test, layer=layer_idx, direction=direction, lambda_val=lambda_val
                ),
                dtype=np.float64,
            )
            preds_random = np.asarray(
                steerer.steer(
                    X_test,
                    layer=layer_idx,
                    direction=random_direction,
                    lambda_val=lambda_val,
                ),
                dtype=np.float64,
            )

        causal_shifts.append(float(np.mean(preds_causal - preds_base)))
        random_shifts.append(float(np.mean(preds_random - preds_base)))

    return {
        "layer": int(layer_idx),
        "quartiles": {"q25": float(q25), "q75": float(q75)},
        "n_high": int(high_mask.sum()),
        "n_low": int(low_mask.sum()),
        "lambdas": [float(v) for v in LAMBDA_VALUES],
        "causal_mean_shift": [float(v) for v in causal_shifts],
        "random_mean_shift": [float(v) for v in random_shifts],
        "causal_abs_pearson": _safe_abs_pearson(LAMBDA_VALUES, causal_shifts),
        "random_abs_pearson": _safe_abs_pearson(LAMBDA_VALUES, random_shifts),
    }


def _run_dataset(dataset: dict[str, np.ndarray | str]) -> dict[str, Any]:
    X_train = np.asarray(dataset["X_train"], dtype=np.float64)
    X_test = np.asarray(dataset["X_test"], dtype=np.float64)
    y_train = np.asarray(dataset["y_train"], dtype=np.float64)
    y_test = np.asarray(dataset["y_test"], dtype=np.float64)

    tabpfn = TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")
    tabpfn.fit(X_train, y_train)
    tabpfn_patching = _noising_sweep_tabpfn(tabpfn, X_test, NOISE_SCALE)
    tabpfn_steering = _steering_metrics(
        model_name="tabpfn",
        model=tabpfn,
        X_test=X_test,
        y_test=y_test,
        layer_idx=TABPFN_STEER_LAYER,
    )

    tabicl = TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)
    tabicl.fit(X_train, y_train)
    tabicl_patching = _noising_sweep_tabicl(tabicl, X_test, NOISE_SCALE)
    tabicl_steering = _steering_metrics(
        model_name="tabicl",
        model=tabicl,
        X_test=X_test,
        y_test=y_test,
        layer_idx=TABICL_STEER_LAYER,
    )

    return {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "patching": {
            "tabpfn": tabpfn_patching,
            "tabicl": tabicl_patching,
            "expected_peak": {"tabpfn": 5, "tabicl": 0},
            "matches_synthetic": {
                "tabpfn_peak_is_l5": bool(tabpfn_patching["peak_layer"] == 5),
                "tabicl_peak_is_l0": bool(tabicl_patching["peak_layer"] == 0),
            },
        },
        "steering": {
            "tabpfn": tabpfn_steering,
            "tabicl": tabicl_steering,
        },
    }


def _plot(all_results: dict[str, Any], save_path: Path) -> None:
    dataset_names = list(all_results.keys())
    n_cols = len(dataset_names)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 9), squeeze=False)

    for col_idx, name in enumerate(dataset_names):
        result = all_results[name]

        ax_patch = axes[0, col_idx]
        pfn_sens = np.asarray(
            result["patching"]["tabpfn"]["normalized_sensitivity"], dtype=np.float64
        )
        icl_sens = np.asarray(
            result["patching"]["tabicl"]["normalized_sensitivity"], dtype=np.float64
        )
        ax_patch.plot(np.arange(pfn_sens.size), pfn_sens, marker="o", label="TabPFN")
        ax_patch.plot(np.arange(icl_sens.size), icl_sens, marker="s", label="TabICL")
        ax_patch.axvline(5, linestyle="--", color="#1f77b4", alpha=0.25)
        ax_patch.axvline(0, linestyle="--", color="#ff7f0e", alpha=0.25)
        ax_patch.set_title(f"{name}: noising patching")
        ax_patch.set_xlabel("Layer")
        ax_patch.set_ylabel("Normalized sensitivity")
        ax_patch.grid(alpha=0.25)
        ax_patch.legend(fontsize=8)

        ax_steer = axes[1, col_idx]
        lambdas = np.asarray(result["steering"]["tabpfn"]["lambdas"], dtype=np.float64)
        pfn_causal = np.asarray(
            result["steering"]["tabpfn"]["causal_mean_shift"], dtype=np.float64
        )
        pfn_random = np.asarray(
            result["steering"]["tabpfn"]["random_mean_shift"], dtype=np.float64
        )
        icl_causal = np.asarray(
            result["steering"]["tabicl"]["causal_mean_shift"], dtype=np.float64
        )
        icl_random = np.asarray(
            result["steering"]["tabicl"]["random_mean_shift"], dtype=np.float64
        )

        ax_steer.plot(lambdas, pfn_causal, marker="o", label="TabPFN causal")
        ax_steer.plot(
            lambdas, pfn_random, linestyle="--", marker="o", label="TabPFN random"
        )
        ax_steer.plot(lambdas, icl_causal, marker="s", label="TabICL causal")
        ax_steer.plot(
            lambdas, icl_random, linestyle="--", marker="s", label="TabICL random"
        )
        ax_steer.set_title(f"{name}: steering")
        ax_steer.set_xlabel("lambda")
        ax_steer.set_ylabel("Mean prediction shift")
        ax_steer.grid(alpha=0.25)
        ax_steer.legend(fontsize=8)

    fig.suptitle("NeurIPS C4 Real-world Causal Validation")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_loaders: list[tuple[str, Callable[..., Any]]] = [
        ("california_housing", load_california_housing),
        ("diabetes", load_diabetes),
        ("bike_sharing", load_bike_sharing),
        ("wine_quality", load_wine_quality),
        ("abalone", load_abalone),
        ("concrete", load_concrete),
    ]

    all_results: dict[str, Any] = {}
    for name, loader in dataset_loaders:
        print(f"[{name}] preparing dataset...")
        dataset = _prepare_dataset(loader=loader, name=name)
        print(f"[{name}] running patching + steering...")
        all_results[name] = _run_dataset(dataset)

    payload = {
        "quick_run": bool(cfg.QUICK_RUN),
        "seed": int(cfg.SEED),
        "device": cfg.DEVICE,
        "noise_scale": float(NOISE_SCALE),
        "steering_layers": {"tabpfn": TABPFN_STEER_LAYER, "tabicl": TABICL_STEER_LAYER},
        "datasets": all_results,
    }

    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _plot(all_results, PLOT_PATH)

    print(f"Saved: {JSON_PATH}")
    print(f"Saved: {PLOT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
