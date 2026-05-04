# pyright: reportMissingImports=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportImplicitStringConcatenation=false
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tabdpt import TabDPTRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.synthetic_generator import generate_linear_data, generate_quadratic_data
from src.hooks.tabdpt_hooker import TabDPTHookedModel
from src.probing.linear_probe import probe_layer


def _as_bool_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


QUICK_RUN = _as_bool_env(os.environ.get("QUICK_RUN", "0"))
SEED = int(os.environ.get("SEED", "42"))

N_TRAIN = 50
N_TEST = 10
N_DATASETS = 5 if QUICK_RUN else 50

RESULTS_DIR = ROOT / "results" / "phase7" / "tabdpt_probing"
RESULTS_PATH = RESULTS_DIR / f"tabdpt_probing_seed{SEED}.json"


def _build_model() -> TabDPTRegressor:
    return TabDPTRegressor(device="cpu")


def _probe_coefficients() -> dict[str, Any]:
    pooled_acts: dict[int, list[np.ndarray]] = {}
    pooled_alpha: list[np.ndarray] = []
    pooled_beta: list[np.ndarray] = []
    layer_indices: list[int] | None = None

    rng = np.random.default_rng(SEED)
    for ds_idx in range(N_DATASETS):
        alpha = float(rng.uniform(0.5, 5.0))
        beta = float(rng.uniform(0.5, 5.0))
        ds = generate_linear_data(
            alpha=alpha,
            beta=beta,
            n_train=N_TRAIN,
            n_test=N_TEST,
            random_seed=SEED + ds_idx,
        )

        model = _build_model()
        model.fit(ds.X_train, ds.y_train)
        hooker = TabDPTHookedModel(model)
        acts = hooker.get_activations(ds.X_train, ds.y_train, ds.X_test)

        if layer_indices is None:
            layer_indices = sorted(acts.keys())
            pooled_acts = {idx: [] for idx in layer_indices}

        for layer_idx in layer_indices:
            pooled_acts[layer_idx].append(np.asarray(acts[layer_idx], dtype=np.float32))

        n_rows = N_TRAIN + N_TEST
        pooled_alpha.append(np.full(n_rows, alpha, dtype=np.float32))
        pooled_beta.append(np.full(n_rows, beta, dtype=np.float32))

        print(
            f"[coefficient] dataset {ds_idx + 1}/{N_DATASETS} done "
            f"(alpha={alpha:.3f}, beta={beta:.3f})"
        )

    if layer_indices is None:
        raise RuntimeError("No datasets processed for coefficient probing")

    alpha_targets = np.concatenate(pooled_alpha)
    beta_targets = np.concatenate(pooled_beta)
    alpha_r2_by_layer: list[float] = []
    beta_r2_by_layer: list[float] = []

    for layer_idx in layer_indices:
        layer_acts = np.vstack(pooled_acts[layer_idx])
        alpha_probe = probe_layer(
            layer_acts,
            alpha_targets,
            complexities=[0],
            random_seed=SEED,
        )
        beta_probe = probe_layer(
            layer_acts,
            beta_targets,
            complexities=[0],
            random_seed=SEED,
        )
        alpha_r2_by_layer.append(float(alpha_probe[0]["r2"]))
        beta_r2_by_layer.append(float(beta_probe[0]["r2"]))

    return {
        "n_layers": len(layer_indices),
        "layer_indices": layer_indices,
        "alpha_r2_by_layer": alpha_r2_by_layer,
        "beta_r2_by_layer": beta_r2_by_layer,
        "peak_layer_alpha": int(np.argmax(np.asarray(alpha_r2_by_layer))),
        "peak_layer_beta": int(np.argmax(np.asarray(beta_r2_by_layer))),
    }


def _probe_intermediary() -> dict[str, Any]:
    pooled_acts: dict[int, list[np.ndarray]] = {}
    pooled_ab: list[np.ndarray] = []
    layer_indices: list[int] | None = None

    for ds_idx in range(N_DATASETS):
        ds = generate_quadratic_data(
            a_range=(0.5, 3.0),
            b_range=(0.5, 3.0),
            c_range=(0.5, 3.0),
            n_train=N_TRAIN,
            n_test=N_TEST,
            random_seed=SEED + ds_idx,
        )

        model = _build_model()
        model.fit(ds.X_train, ds.y_train)
        hooker = TabDPTHookedModel(model)
        acts = hooker.get_activations(ds.X_train, ds.y_train, ds.X_test)

        if layer_indices is None:
            layer_indices = sorted(acts.keys())
            pooled_acts = {idx: [] for idx in layer_indices}

        for layer_idx in layer_indices:
            pooled_acts[layer_idx].append(np.asarray(acts[layer_idx], dtype=np.float32))

        pooled_ab.append(
            np.concatenate(
                [
                    np.asarray(ds.intermediary_train, dtype=np.float32),
                    np.asarray(ds.intermediary_test, dtype=np.float32),
                ]
            )
        )

        print(f"[intermediary] dataset {ds_idx + 1}/{N_DATASETS} done")

    if layer_indices is None:
        raise RuntimeError("No datasets processed for intermediary probing")

    ab_targets = np.concatenate(pooled_ab)
    ab_r2_by_layer: list[float] = []

    for layer_idx in layer_indices:
        layer_acts = np.vstack(pooled_acts[layer_idx])
        probe_results = probe_layer(
            layer_acts,
            ab_targets,
            complexities=[0],
            random_seed=SEED,
        )
        ab_r2_by_layer.append(float(probe_results[0]["r2"]))

    return {
        "n_layers": len(layer_indices),
        "layer_indices": layer_indices,
        "ab_r2_by_layer": ab_r2_by_layer,
        "peak_layer_ab": int(np.argmax(np.asarray(ab_r2_by_layer))),
    }


def _probe_copy_mechanism() -> dict[str, Any]:
    pooled_acts: dict[int, list[np.ndarray]] = {}
    pooled_a: list[np.ndarray] = []
    pooled_b: list[np.ndarray] = []
    pooled_c: list[np.ndarray] = []
    layer_indices: list[int] | None = None

    for ds_idx in range(N_DATASETS):
        ds = generate_quadratic_data(
            a_range=(0.5, 3.0),
            b_range=(0.5, 3.0),
            c_range=(0.5, 3.0),
            n_train=N_TRAIN,
            n_test=N_TEST,
            random_seed=SEED + ds_idx,
        )

        model = _build_model()
        model.fit(ds.X_train, ds.y_train)
        hooker = TabDPTHookedModel(model)
        acts = hooker.get_activations(ds.X_train, ds.y_train, ds.X_test)

        if layer_indices is None:
            layer_indices = sorted(acts.keys())
            pooled_acts = {idx: [] for idx in layer_indices}

        for layer_idx in layer_indices:
            answer_acts = np.asarray(acts[layer_idx][-N_TEST:], dtype=np.float32)
            pooled_acts[layer_idx].append(answer_acts)

        pooled_a.append(np.asarray(ds.X_test[:, 0], dtype=np.float32))
        pooled_b.append(np.asarray(ds.X_test[:, 1], dtype=np.float32))
        pooled_c.append(np.asarray(ds.X_test[:, 2], dtype=np.float32))

        print(f"[copy] dataset {ds_idx + 1}/{N_DATASETS} done")

    if layer_indices is None:
        raise RuntimeError("No datasets processed for copy mechanism probing")

    targets = {
        "a": np.concatenate(pooled_a),
        "b": np.concatenate(pooled_b),
        "c": np.concatenate(pooled_c),
    }
    r2_by_target: dict[str, list[float]] = {"a": [], "b": [], "c": []}

    for layer_idx in layer_indices:
        layer_acts = np.vstack(pooled_acts[layer_idx])
        for target_name in ["a", "b", "c"]:
            probe_results = probe_layer(
                layer_acts,
                targets[target_name],
                complexities=[0],
                random_seed=SEED,
            )
            r2_by_target[target_name].append(float(probe_results[0]["r2"]))

    return {
        "n_layers": len(layer_indices),
        "layer_indices": layer_indices,
        "a_r2_by_layer": r2_by_target["a"],
        "b_r2_by_layer": r2_by_target["b"],
        "c_r2_by_layer": r2_by_target["c"],
        "peak_layer_a": int(np.argmax(np.asarray(r2_by_target["a"]))),
        "peak_layer_b": int(np.argmax(np.asarray(r2_by_target["b"]))),
        "peak_layer_c": int(np.argmax(np.asarray(r2_by_target["c"]))),
    }


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(
        f"QUICK_RUN={QUICK_RUN}, seed={SEED}, n_datasets={N_DATASETS}, "
        f"n_train={N_TRAIN}, n_test={N_TEST}"
    )

    results = {
        "meta": {
            "seed": SEED,
            "quick_run": QUICK_RUN,
            "n_datasets": N_DATASETS,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "model": "TabDPTRegressor",
            "hidden_dim": 768,
            "n_layers": 16,
        },
        "coefficient": _probe_coefficients(),
        "intermediary": _probe_intermediary(),
        "copy_mechanism": _probe_copy_mechanism(),
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
