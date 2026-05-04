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

RESULTS_DIR = ROOT / "results" / "rd5" / "copy_mechanism"


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


def _layer_indices_for(model_name: str) -> list[int]:
    if model_name == "iltm":
        return [0, 1, 2]
    return list(range(12))


def _collect_copy_r2_for_model(model_name: str) -> dict[str, Any]:
    n_datasets = _dataset_count_for(model_name)
    layer_indices = _layer_indices_for(model_name)

    pooled_activations: dict[int, list[np.ndarray]] = {i: [] for i in layer_indices}
    pooled_a: list[np.ndarray] = []
    pooled_b: list[np.ndarray] = []
    pooled_c: list[np.ndarray] = []
    pooled_ab: list[np.ndarray] = []

    for ds_idx in range(n_datasets):
        ds = generate_quadratic_data(
            n_train=N_TRAIN, n_test=N_TEST, random_seed=SEED + ds_idx
        )

        if model_name == "tabpfn":
            model = _build_tabpfn()
            hooker = TabPFNHookedModel
        elif model_name == "tabicl":
            model = _build_tabicl()
            hooker = TabICLHookedModel
        elif model_name == "iltm":
            model = _build_iltm()
            hooker = iLTMHookedModel
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(ds.X_train, ds.y_train)
        hooked_model = hooker(model)
        _, cache = hooked_model.forward_with_cache(ds.X_test)

        for layer_idx in layer_indices:
            if model_name == "tabpfn":
                act = hooked_model.get_test_label_token(cache, layer_idx)
            else:
                act = hooked_model.get_layer_activations(cache, layer_idx)
            pooled_activations[layer_idx].append(np.asarray(act, dtype=np.float32))

        pooled_a.append(np.asarray(ds.X_test[:, 0], dtype=np.float32))
        pooled_b.append(np.asarray(ds.X_test[:, 1], dtype=np.float32))
        pooled_c.append(np.asarray(ds.X_test[:, 2], dtype=np.float32))
        pooled_ab.append(np.asarray(ds.intermediary_test, dtype=np.float32))

        print(f"[{model_name}] dataset {ds_idx + 1}/{n_datasets} done")

    targets = {
        "a": np.concatenate(pooled_a, axis=0),
        "b": np.concatenate(pooled_b, axis=0),
        "c": np.concatenate(pooled_c, axis=0),
        "ab": np.concatenate(pooled_ab, axis=0),
    }

    r2_by_target: dict[str, list[float]] = {"a": [], "b": [], "c": [], "ab": []}
    for layer_idx in layer_indices:
        layer_acts = np.vstack(pooled_activations[layer_idx])
        for target_name in ["a", "b", "c", "ab"]:
            probe_results = probe_layer(
                layer_acts,
                targets[target_name],
                complexities=[0],
                random_seed=SEED,
            )
            r2_by_target[target_name].append(float(probe_results[0]["r2"]))

    return {
        "a_r2_by_layer": r2_by_target["a"],
        "b_r2_by_layer": r2_by_target["b"],
        "c_r2_by_layer": r2_by_target["c"],
        "ab_r2_by_layer": r2_by_target["ab"],
        "n_layers": len(layer_indices),
    }


def _plot_results(results: dict[str, dict[str, Any]], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    model_order = ["tabpfn", "tabicl", "iltm"]
    titles = {"tabpfn": "TabPFN", "tabicl": "TabICL", "iltm": "iLTM"}
    styles = {
        "a": {"label": "a", "color": "blue", "marker": "o"},
        "b": {"label": "b", "color": "orange", "marker": "s"},
        "c": {"label": "c", "color": "green", "marker": "^"},
        "ab": {"label": "a*b", "color": "red", "marker": "D"},
    }

    for ax, model_key in zip(axes, model_order, strict=True):
        model_res = results[model_key]
        layers = np.arange(model_res["n_layers"])
        for target_name in ["a", "b", "c", "ab"]:
            curve = np.asarray(model_res[f"{target_name}_r2_by_layer"])
            style = styles[target_name]
            ax.plot(
                layers,
                curve,
                label=f"R2({style['label']})",
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
            )

        ax.set_title(titles[model_key])
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.25)
        if model_key == "tabpfn":
            ax.set_ylabel("R2")
        ax.legend(loc="best")

    fig.suptitle("RD5 Copy Mechanism: Recoverability of a, b, c, a*b by Layer")
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

    results: dict[str, dict[str, Any]] = {
        "tabpfn": _collect_copy_r2_for_model("tabpfn"),
        "tabicl": _collect_copy_r2_for_model("tabicl"),
        "iltm": _collect_copy_r2_for_model("iltm"),
    }

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
