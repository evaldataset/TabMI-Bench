# pyright: reportMissingImports=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportImplicitStringConcatenation=false
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tabdpt import TabDPTRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.synthetic_generator import generate_linear_data
from src.hooks.tabdpt_hooker import TabDPTHookedModel


def _as_bool_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


QUICK_RUN = _as_bool_env(os.environ.get("QUICK_RUN", "0"))
SEED = int(os.environ.get("SEED", "42"))
DEVICE = os.environ.get(
    "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
).strip()

N_TRAIN = 50
N_TEST = 10
N_DATASETS = 5 if QUICK_RUN else 10
NOISE_SIGMA = 0.5
N_ENSEMBLES = 1

RESULTS_DIR = ROOT / "results" / "phase7" / "tabdpt_causal"
RESULTS_PATH = RESULTS_DIR / f"tabdpt_causal_seed{SEED}.json"


def _build_model() -> TabDPTRegressor:
    return TabDPTRegressor(device=DEVICE, compile=False, verbose=False)


def _to_tensor_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Unexpected layer output type for TabDPT hook")


def _compute_cka(x: np.ndarray, y: np.ndarray) -> float:
    n = x.shape[0]
    k_x = x @ x.T
    k_y = y @ y.T
    h = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / n
    k_x_c = h @ k_x @ h
    k_y_c = h @ k_y @ h

    hsic_xy = np.trace(k_x_c @ k_y_c) / (n - 1) ** 2
    hsic_xx = np.trace(k_x_c @ k_x_c) / (n - 1) ** 2
    hsic_yy = np.trace(k_y_c @ k_y_c) / (n - 1) ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom <= 1e-10:
        return 0.0
    return float(hsic_xy / denom)


def _compute_cka_matrix(layer_acts: list[np.ndarray]) -> np.ndarray:
    n_layers = len(layer_acts)
    cka = np.zeros((n_layers, n_layers), dtype=np.float64)
    for i in range(n_layers):
        for j in range(n_layers):
            cka[i, j] = _compute_cka(layer_acts[i], layer_acts[j])
    return cka


def _find_computation_block(
    adjacent_cka: list[float], threshold: float = 0.95
) -> dict[str, float | int]:
    best_start = -1
    best_end = -1
    best_len = 0
    run_start = -1

    for i, value in enumerate(adjacent_cka):
        if value > threshold:
            if run_start == -1:
                run_start = i
        elif run_start != -1:
            run_end = i - 1
            run_len = run_end - run_start + 1
            if run_len > best_len:
                best_len = run_len
                best_start = run_start
                best_end = run_end
            run_start = -1

    if run_start != -1:
        run_end = len(adjacent_cka) - 1
        run_len = run_end - run_start + 1
        if run_len > best_len:
            best_len = run_len
            best_start = run_start
            best_end = run_end

    if best_start == -1:
        return {"start": -1, "end": -1, "mean_cka": 0.0}

    return {
        "start": int(best_start),
        "end": int(best_end + 1),
        "mean_cka": float(np.mean(adjacent_cka[best_start : best_end + 1])),
    }


def _layer_stds(
    model: TabDPTRegressor, x_test: np.ndarray, n_layers: int
) -> list[float]:
    stds: list[float] = []
    for layer_idx in range(n_layers):
        bucket: list[float] = []

        def _stats_hook(_module: torch.nn.Module, _inp: Any, output: Any) -> None:
            out = _to_tensor_output(output)
            bucket.append(float(out.detach().std().item()))

        handle = model.model.transformer_encoder[layer_idx].register_forward_hook(
            _stats_hook
        )
        _ = model.predict(x_test, n_ensembles=N_ENSEMBLES)
        handle.remove()
        stds.append(float(np.mean(bucket)) if bucket else 1.0)
    return stds


def _noising_for_dataset(
    model: TabDPTRegressor,
    x_test: np.ndarray,
    noise_sigma: float,
    dataset_seed: int,
) -> list[float]:
    clean_preds = np.asarray(
        model.predict(x_test, n_ensembles=N_ENSEMBLES), dtype=np.float32
    )
    n_layers = len(model.model.transformer_encoder)
    act_stds = _layer_stds(model, x_test, n_layers)

    mse_by_layer: list[float] = []
    for layer_idx in range(n_layers):
        rng = np.random.default_rng(dataset_seed + layer_idx)

        def _noise_hook(
            _module: torch.nn.Module, _inp: Any, output: Any
        ) -> torch.Tensor:
            out = _to_tensor_output(output)
            noise = torch.from_numpy(
                rng.normal(
                    0.0, noise_sigma * act_stds[layer_idx], size=tuple(out.shape)
                ).astype(np.float32)
            ).to(out.device)
            return out + noise

        handle = model.model.transformer_encoder[layer_idx].register_forward_hook(
            _noise_hook
        )
        noised_preds = np.asarray(
            model.predict(x_test, n_ensembles=N_ENSEMBLES), dtype=np.float32
        )
        handle.remove()

        mse = float(np.mean((noised_preds - clean_preds) ** 2))
        mse_by_layer.append(mse)

    return mse_by_layer


def run_noising_causal_tracing() -> dict[str, Any]:
    mse_rows: list[np.ndarray] = []

    for ds_idx in range(N_DATASETS):
        alpha = 2.0
        beta = 3.0
        ds = generate_linear_data(
            alpha=alpha,
            beta=beta,
            n_train=N_TRAIN,
            n_test=N_TEST,
            random_seed=SEED + ds_idx,
        )

        model = _build_model()
        model.fit(ds.X_train, ds.y_train)
        mse_by_layer = _noising_for_dataset(
            model=model,
            x_test=ds.X_test,
            noise_sigma=NOISE_SIGMA,
            dataset_seed=SEED + 1000 + ds_idx,
        )
        mse_rows.append(np.asarray(mse_by_layer, dtype=np.float64))
        print(f"[causal-noising] dataset {ds_idx + 1}/{N_DATASETS} done")

    mse_matrix = np.vstack(mse_rows)
    mean_mse = np.mean(mse_matrix, axis=0)
    max_mse = float(np.max(mean_mse)) if float(np.max(mean_mse)) > 0 else 1.0
    normalized = (mean_mse / max_mse).tolist()

    return {
        "sigma": NOISE_SIGMA,
        "n_datasets": N_DATASETS,
        "n_layers": int(mean_mse.shape[0]),
        "mse_increase_by_layer": mean_mse.tolist(),
        "normalized_sensitivity": normalized,
        "most_sensitive_layer": int(np.argmax(mean_mse)),
    }


def run_cka_analysis() -> dict[str, Any]:
    pooled: dict[int, list[np.ndarray]] = {}
    layer_indices: list[int] | None = None

    for ds_idx in range(N_DATASETS):
        ds = generate_linear_data(
            alpha=2.0,
            beta=3.0,
            n_train=N_TRAIN,
            n_test=N_TEST,
            random_seed=SEED + 2000 + ds_idx,
        )

        model = _build_model()
        model.fit(ds.X_train, ds.y_train)
        hooker = TabDPTHookedModel(model, device=DEVICE)
        acts = hooker.get_activations(ds.X_train, ds.y_train, ds.X_test)

        if layer_indices is None:
            layer_indices = sorted(acts.keys())
            pooled = {idx: [] for idx in layer_indices}

        for layer_idx in layer_indices:
            pooled[layer_idx].append(np.asarray(acts[layer_idx], dtype=np.float64))

        print(f"[cka] dataset {ds_idx + 1}/{N_DATASETS} done")

    if layer_indices is None:
        raise RuntimeError("No datasets processed for CKA")

    pooled_layers = [np.vstack(pooled[idx]) for idx in layer_indices]
    cka_matrix = _compute_cka_matrix(pooled_layers)
    adjacent_cka = [float(cka_matrix[i, i + 1]) for i in range(cka_matrix.shape[0] - 1)]

    return {
        "n_layers": len(layer_indices),
        "layer_indices": layer_indices,
        "cka_matrix": cka_matrix.tolist(),
        "adjacent_cka": adjacent_cka,
        "computation_block": _find_computation_block(adjacent_cka),
    }


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(
        f"QUICK_RUN={QUICK_RUN}, seed={SEED}, n_datasets={N_DATASETS}, "
        f"n_train={N_TRAIN}, n_test={N_TEST}, sigma={NOISE_SIGMA}, "
        f"n_ensembles={N_ENSEMBLES}, device={DEVICE}"
    )

    results = {
        "meta": {
            "seed": SEED,
            "quick_run": QUICK_RUN,
            "n_datasets": N_DATASETS,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "model": "TabDPTRegressor",
            "device": DEVICE,
            "hidden_dim": 768,
            "n_layers": 16,
            "n_ensembles": N_ENSEMBLES,
        },
        "causal_tracing": run_noising_causal_tracing(),
        "cka": run_cka_analysis(),
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
