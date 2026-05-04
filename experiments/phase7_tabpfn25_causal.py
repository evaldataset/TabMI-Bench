# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportImplicitStringConcatenation=false
from __future__ import annotations

import inspect
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
DEVICE = os.getenv("DEVICE", "cpu")
QUICK_RUN = os.getenv("QUICK_RUN", "0") == "1"
NOISE_SCALE = 0.5

MODEL_PATH = str(ROOT / "tabpfn-v2.5-regressor-v2.5_default.ckpt")
RESULTS_DIR = ROOT / "results" / "phase7"
JSON_PATH = RESULTS_DIR / f"tabpfn25_causal_seed{SEED}.json"


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
) -> dict[str, np.ndarray | str] | None:
    try:
        x_all, y_all = _extract_xy(loader=loader, random_seed=SEED)
    except Exception as exc:
        print(f"[{name}] skipped: dataset load failed ({exc})")
        return None

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
        "n_layers": int(len(layers)),
        "mse_increase_by_layer": [float(v) for v in mse_increase],
        "normalized_sensitivity": sensitivity,
        "peak_layer": int(np.argmax(sensitivity)),
    }


def main() -> int:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_dataset_loaders: list[tuple[str, Callable[..., Any]]] = [
        ("california_housing", rw_datasets.load_california_housing),
        ("diabetes", load_diabetes),
        ("bike_sharing", rw_datasets.load_bike_sharing),
        ("wine_quality", rw_datasets.load_wine_quality),
        ("abalone", rw_datasets.load_abalone),
        ("concrete", rw_datasets.load_concrete),
        ("adult_income", rw_datasets.load_adult_income),
        ("bank_marketing", rw_datasets.load_bank_marketing),
        ("credit_g", rw_datasets.load_credit_g),
        ("segment", rw_datasets.load_segment),
        ("vehicle", rw_datasets.load_vehicle),
    ]

    dataset_loaders = all_dataset_loaders
    if QUICK_RUN:
        quick_names = {"california_housing", "diabetes", "wine_quality"}
        dataset_loaders = [it for it in all_dataset_loaders if it[0] in quick_names]

    print("=" * 72)
    print("Phase 7 Experiment 2: TabPFN v2.5 real-world causal tracing")
    print("=" * 72)
    print(f"seed={SEED}, device={DEVICE}, quick_run={QUICK_RUN}")
    print(f"datasets={len(dataset_loaders)}, noise_scale={NOISE_SCALE}")

    all_results: dict[str, Any] = {}
    for idx, (name, loader) in enumerate(dataset_loaders, start=1):
        print(f"[{idx}/{len(dataset_loaders)}] [{name}] preparing dataset...")
        dataset = _prepare_dataset(loader=loader, name=name)
        if dataset is None:
            continue

        X_train = np.asarray(dataset["X_train"], dtype=np.float64)
        X_test = np.asarray(dataset["X_test"], dtype=np.float64)
        y_train = np.asarray(dataset["y_train"], dtype=np.float64)

        print(f"[{idx}/{len(dataset_loaders)}] [{name}] fit TabPFN v2.5 + noising...")
        model = TabPFNRegressor(device=DEVICE, model_path=MODEL_PATH)
        model.fit(X_train, y_train)
        sweep = _noising_sweep_tabpfn(
            model=model, X_test=X_test, noise_scale=NOISE_SCALE
        )

        all_results[name] = {
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "patching": sweep,
        }
        print(
            f"[{idx}/{len(dataset_loaders)}] [{name}] done "
            f"(layers={sweep['n_layers']}, peak=L{sweep['peak_layer']})"
        )

    payload = {
        "quick_run": bool(QUICK_RUN),
        "seed": int(SEED),
        "device": DEVICE,
        "model": "TabPFN v2.5",
        "model_path": "tabpfn-v2.5-regressor-v2.5_default.ckpt",
        "noise_scale": float(NOISE_SCALE),
        "datasets": all_results,
        "comparison_note": "Compare v2.5 peak layers against v2 real-world L11 pattern.",
    }

    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
