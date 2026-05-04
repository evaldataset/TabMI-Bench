# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportImplicitStringConcatenation=false
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import real_world_datasets as rw_datasets

SEED = int(os.getenv("SEED", "42"))
DEVICE = os.getenv("DEVICE", "cpu")
QUICK_RUN = os.getenv("QUICK_RUN", "0") == "1"

RESULTS_DIR = ROOT / "results" / "phase7"
JSON_PATH = RESULTS_DIR / f"peak_investigation_seed{SEED}.json"


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


def _noising_sweep_tabpfn(
    model: TabPFNRegressor,
    X_test: np.ndarray,
    noise_scale: float,
    seed: int,
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
        layer_rng = np.random.default_rng(seed + layer_idx)

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


def _build_synthetic_dataset(
    seed: int, n_train: int, n_test: int, d: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    alpha = float(rng.uniform(1.0, 5.0))
    beta = float(rng.uniform(1.0, 5.0))

    x1_train = rng.normal(size=n_train).astype(np.float32)
    x2_train = rng.normal(size=n_train).astype(np.float32)
    x1_test = rng.normal(size=n_test).astype(np.float32)
    x2_test = rng.normal(size=n_test).astype(np.float32)

    if d > 2:
        noise_train = rng.normal(size=(n_train, d - 2)).astype(np.float32)
        noise_test = rng.normal(size=(n_test, d - 2)).astype(np.float32)
        X_train = np.column_stack([x1_train, x2_train, noise_train]).astype(np.float32)
        X_test = np.column_stack([x1_test, x2_test, noise_test]).astype(np.float32)
    else:
        X_train = np.column_stack([x1_train, x2_train]).astype(np.float32)
        X_test = np.column_stack([x1_test, x2_test]).astype(np.float32)

    y_train = (alpha * x1_train + beta * x2_train).astype(np.float32)
    y_test = (alpha * x1_test + beta * x2_test).astype(np.float32)

    return {
        "alpha": alpha,
        "beta": beta,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def _prepare_california_housing() -> dict[str, np.ndarray]:
    x_all, y_all = _extract_xy(
        loader=rw_datasets.load_california_housing, random_seed=SEED
    )
    X_train, X_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=0.2,
        random_state=SEED,
    )
    scaler = StandardScaler()
    X_train_scaled = np.asarray(scaler.fit_transform(X_train), dtype=np.float64)
    X_test_scaled = np.asarray(scaler.transform(X_test), dtype=np.float64)
    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": np.asarray(y_train, dtype=np.float64),
        "y_test": np.asarray(y_test, dtype=np.float64),
    }


def _run_test_a() -> dict[str, Any]:
    feature_counts = [2, 8, 11] if QUICK_RUN else [2, 3, 5, 8, 11]
    n_datasets = 3 if QUICK_RUN else 10
    noise_scale = 0.5

    by_feature_count: dict[str, Any] = {}
    for d in feature_counts:
        print(f"[Test A] d={d}: running {n_datasets} synthetic datasets")
        datasets_results: list[dict[str, Any]] = []
        for ds_idx in range(n_datasets):
            ds_seed = SEED + (d * 1000) + ds_idx
            dataset = _build_synthetic_dataset(
                seed=ds_seed,
                n_train=50,
                n_test=10,
                d=d,
            )

            model = TabPFNRegressor(
                device=DEVICE, model_path="tabpfn-v2-regressor.ckpt"
            )
            model.fit(dataset["X_train"], dataset["y_train"])
            sweep = _noising_sweep_tabpfn(
                model=model,
                X_test=np.asarray(dataset["X_test"], dtype=np.float64),
                noise_scale=noise_scale,
                seed=ds_seed,
            )

            datasets_results.append(
                {
                    "dataset_index": int(ds_idx),
                    "seed": int(ds_seed),
                    "alpha": float(dataset["alpha"]),
                    "beta": float(dataset["beta"]),
                    "peak_layer": int(sweep["peak_layer"]),
                    "normalized_sensitivity": sweep["normalized_sensitivity"],
                    "mse_increase_by_layer": sweep["mse_increase_by_layer"],
                }
            )
            print(
                f"  [Test A] d={d} dataset {ds_idx + 1}/{n_datasets} "
                f"peak=L{sweep['peak_layer']}"
            )

        peaks = [int(it["peak_layer"]) for it in datasets_results]
        by_feature_count[str(d)] = {
            "n_features": int(d),
            "n_datasets": int(n_datasets),
            "n_train": 50,
            "n_test": 10,
            "noise_scale": float(noise_scale),
            "peak_layers": peaks,
            "peak_layer_mean": float(np.mean(np.asarray(peaks, dtype=np.float64))),
            "datasets": datasets_results,
        }

    return {
        "hypothesis": (
            "Synthetic peak layer shifts later as feature count increases "
            "(d=2/3 to d=8/11)."
        ),
        "feature_count_sweep": by_feature_count,
    }


def _run_test_b() -> dict[str, Any]:
    noise_scales = [0.5] if QUICK_RUN else [0.1, 0.2, 0.3, 0.5, 1.0]
    dataset = _prepare_california_housing()

    model = TabPFNRegressor(device=DEVICE, model_path="tabpfn-v2-regressor.ckpt")
    model.fit(dataset["X_train"], dataset["y_train"])

    sweeps: list[dict[str, Any]] = []
    for sigma in noise_scales:
        print(f"[Test B] california_housing sigma={sigma}")
        sweep = _noising_sweep_tabpfn(
            model=model,
            X_test=np.asarray(dataset["X_test"], dtype=np.float64),
            noise_scale=float(sigma),
            seed=SEED,
        )
        sweeps.append(sweep)
        print(f"  [Test B] sigma={sigma} peak=L{sweep['peak_layer']}")

    return {
        "dataset": "california_housing",
        "n_train": int(np.asarray(dataset["X_train"]).shape[0]),
        "n_test": int(np.asarray(dataset["X_test"]).shape[0]),
        "n_features": int(np.asarray(dataset["X_train"]).shape[1]),
        "noise_scale_sweeps": sweeps,
        "peak_layer_by_sigma": {
            str(it["noise_scale"]): int(it["peak_layer"]) for it in sweeps
        },
    }


def main() -> int:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Phase 7 Experiment 1: TabPFN peak-layer investigation")
    print("=" * 72)
    print(f"seed={SEED}, device={DEVICE}, quick_run={QUICK_RUN}")

    test_a = _run_test_a()
    test_b = _run_test_b()

    payload = {
        "quick_run": bool(QUICK_RUN),
        "seed": int(SEED),
        "device": DEVICE,
        "model": "TabPFN v2",
        "model_path": "tabpfn-v2-regressor.ckpt",
        "experiment": "phase7_peak_investigation",
        "test_a_feature_count_effect": test_a,
        "test_b_noise_scale_realworld": test_b,
    }

    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
