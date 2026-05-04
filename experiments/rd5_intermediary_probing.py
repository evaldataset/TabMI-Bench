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
from typing import Any

import numpy as np
from iltm import iLTMRegressor
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

from src.data.synthetic_generator import generate_quadratic_data
from src.hooks.iltm_hooker import iLTMHookedModel
from src.hooks.tabicl_hooker import TabICLHookedModel
from src.hooks.tabpfn_hooker import TabPFNHookedModel
from src.probing.linear_probe import probe_layer
from rd5_config import cfg

QUICK_RUN = cfg.QUICK_RUN
SEED = cfg.SEED
N_TRAIN = cfg.N_TRAIN
N_TEST = cfg.N_TEST

RESULTS_DIR = ROOT / "results" / "rd5" / "intermediary_probing"


def _build_tabpfn() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _build_tabicl() -> TabICLRegressor:
    return TabICLRegressor(device=cfg.DEVICE, random_state=SEED)


def _build_iltm() -> iLTMRegressor:
    model = iLTMRegressor(device="cpu", n_ensemble=1, seed=SEED)
    model.n_ensemble = 1
    return model


def _dataset_count_for(model_name: str) -> int:
    if QUICK_RUN:
        if model_name == "iltm":
            return 3
        return 5
    if model_name == "iltm":
        return 10
    return 20


def _collect_model_layer_results(model_name: str) -> dict[str, Any]:
    n_datasets = _dataset_count_for(model_name)

    if model_name == "tabpfn":
        hooker_type = TabPFNHookedModel
        layer_indices = list(range(12))
    elif model_name == "tabicl":
        hooker_type = TabICLHookedModel
        layer_indices = list(range(12))
    elif model_name == "iltm":
        hooker_type = iLTMHookedModel
        layer_indices = [0, 1, 2]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pooled_activations: dict[int, list[np.ndarray]] = {i: [] for i in layer_indices}
    pooled_intermediary: list[np.ndarray] = []
    pooled_final: list[np.ndarray] = []

    for ds_idx in range(n_datasets):
        dataset = generate_quadratic_data(
            a_range=(0.5, 3.0),
            b_range=(0.5, 3.0),
            c_range=(0.5, 3.0),
            n_train=N_TRAIN,
            n_test=N_TEST,
            random_seed=SEED + ds_idx,
        )

        if model_name == "tabpfn":
            model = _build_tabpfn()
        elif model_name == "tabicl":
            model = _build_tabicl()
        else:
            model = _build_iltm()

        model.fit(dataset.X_train, dataset.y_train)
        hooker = hooker_type(model)
        _, cache = hooker.forward_with_cache(dataset.X_test)

        for layer_idx in layer_indices:
            if model_name == "tabpfn":
                act = hooker.get_test_label_token(cache, layer_idx)
            else:
                act = hooker.get_layer_activations(cache, layer_idx)
            pooled_activations[layer_idx].append(np.asarray(act, dtype=np.float32))

        pooled_intermediary.append(
            np.asarray(dataset.intermediary_test, dtype=np.float32).reshape(-1)
        )
        pooled_final.append(np.asarray(dataset.y_test, dtype=np.float32).reshape(-1))

        print(f"[{model_name}] dataset {ds_idx + 1}/{n_datasets} done")

    intermediary_targets = np.concatenate(pooled_intermediary, axis=0)
    final_targets = np.concatenate(pooled_final, axis=0)

    intermediary_r2_by_layer: list[float] = []
    final_r2_by_layer: list[float] = []

    for layer_idx in layer_indices:
        layer_acts = np.vstack(pooled_activations[layer_idx])

        intermediary_probe = probe_layer(
            layer_acts,
            intermediary_targets,
            complexities=[0],
            random_seed=SEED,
        )
        final_probe = probe_layer(
            layer_acts,
            final_targets,
            complexities=[0],
            random_seed=SEED,
        )

        intermediary_r2_by_layer.append(float(intermediary_probe[0]["r2"]))
        final_r2_by_layer.append(float(final_probe[0]["r2"]))

    peak_layer_intermediary = int(np.argmax(np.asarray(intermediary_r2_by_layer)))
    peak_layer_final = int(np.argmax(np.asarray(final_r2_by_layer)))

    return {
        "intermediary_r2_by_layer": intermediary_r2_by_layer,
        "final_r2_by_layer": final_r2_by_layer,
        "peak_layer_intermediary": peak_layer_intermediary,
        "peak_layer_final": peak_layer_final,
    }


def _plot_results(results: dict[str, dict[str, Any]], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    model_order = ["tabpfn", "tabicl", "iltm"]
    titles = {"tabpfn": "TabPFN", "tabicl": "TabICL", "iltm": "iLTM"}

    for ax, model_key in zip(axes, model_order, strict=True):
        intermediary = np.asarray(results[model_key]["intermediary_r2_by_layer"])
        final = np.asarray(results[model_key]["final_r2_by_layer"])
        layers = np.arange(intermediary.shape[0])

        ax.plot(layers, intermediary, marker="o", linewidth=2.0, label="R2(a*b)")
        ax.plot(layers, final, marker="s", linewidth=2.0, label="R2(z)")
        ax.set_title(titles[model_key])
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.25)
        if model_key == "tabpfn":
            ax.set_ylabel("R2")
        ax.legend(loc="best")

    fig.suptitle("RD5 Intermediary Probing: Layer-wise R2 (a*b vs z)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"QUICK_RUN={QUICK_RUN}, seed={SEED}")
    print(
        f"Datasets -> TabPFN: {_dataset_count_for('tabpfn')}, "
        f"TabICL: {_dataset_count_for('tabicl')}, iLTM: {_dataset_count_for('iltm')}"
    )

    results: dict[str, dict[str, Any]] = {}
    results["tabpfn"] = _collect_model_layer_results("tabpfn")
    results["tabicl"] = _collect_model_layer_results("tabicl")
    results["iltm"] = _collect_model_layer_results("iltm")

    json_path = RESULTS_DIR / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_path = RESULTS_DIR / "r2_comparison.png"
    _plot_results(results, plot_path)

    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
