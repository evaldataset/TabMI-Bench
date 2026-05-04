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

from src.data.synthetic_generator import generate_linear_data
from src.hooks.iltm_hooker import iLTMHookedModel
from src.hooks.tabicl_hooker import TabICLHookedModel
from src.hooks.tabpfn_hooker import TabPFNHookedModel
from rd5_config import cfg

QUICK_RUN = cfg.QUICK_RUN
SEED = cfg.SEED
N_TRAIN = cfg.N_TRAIN
N_TEST = cfg.N_TEST
N_DATASETS = cfg.dataset_count("tabpfn")
CKA_THRESHOLD = 0.95
RESULTS_DIR = ROOT / "results" / "rd5" / "cka"


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    n = X.shape[0]
    K_X = X @ X.T
    K_Y = Y @ Y.T
    H = np.eye(n) - np.ones((n, n)) / n
    K_X_c = H @ K_X @ H
    K_Y_c = H @ K_Y @ H
    hsic_xy = np.trace(K_X_c @ K_Y_c) / (n - 1) ** 2
    hsic_xx = np.trace(K_X_c @ K_X_c) / (n - 1) ** 2
    hsic_yy = np.trace(K_Y_c @ K_Y_c) / (n - 1) ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 1e-10 else 0.0


def compute_cka_matrix(layer_acts: list[np.ndarray]) -> np.ndarray:
    n_layers = len(layer_acts)
    cka_matrix = np.zeros((n_layers, n_layers), dtype=np.float64)
    for i in range(n_layers):
        for j in range(n_layers):
            cka_matrix[i, j] = compute_cka(layer_acts[i], layer_acts[j])
    return cka_matrix


def find_computation_block(
    adjacent_cka: list[float],
    threshold: float = CKA_THRESHOLD,
) -> tuple[int, int, float]:
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
        return -1, -1, 0.0

    mean_cka = float(np.mean(adjacent_cka[best_start : best_end + 1]))
    return best_start, best_end + 1, mean_cka


def _build_tabpfn() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _build_tabicl() -> TabICLRegressor:
    return TabICLRegressor(device=cfg.DEVICE, random_state=SEED)


def _build_iltm() -> iLTMRegressor:
    model = iLTMRegressor(device="cpu", n_ensemble=1, seed=SEED)
    model.n_ensemble = 1
    return model


def _extract_layer_activations(
    hooker: Any, cache: dict[str, Any], layer_idx: int
) -> np.ndarray:
    if hasattr(hooker, "get_layer_activations"):
        return np.asarray(
            hooker.get_layer_activations(cache, layer_idx), dtype=np.float64
        )
    return np.asarray(hooker.get_test_label_token(cache, layer_idx), dtype=np.float64)


def collect_model_activations(model_name: str) -> list[np.ndarray]:
    if model_name == "tabpfn":
        build_model = _build_tabpfn
        hooker_cls = TabPFNHookedModel
        layer_indices = list(range(12))
    elif model_name == "tabicl":
        build_model = _build_tabicl
        hooker_cls = TabICLHookedModel
        layer_indices = list(range(12))
    elif model_name == "iltm":
        build_model = _build_iltm
        hooker_cls = iLTMHookedModel
        layer_indices = [0, 1, 2]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pooled_by_layer: dict[int, list[np.ndarray]] = {
        layer_idx: [] for layer_idx in layer_indices
    }

    for ds_idx in range(N_DATASETS):
        dataset = generate_linear_data(
            alpha=2.0,
            beta=3.0,
            n_train=N_TRAIN,
            n_test=N_TEST,
            random_seed=SEED + ds_idx,
        )

        model = build_model()
        model.fit(dataset.X_train, dataset.y_train)
        hooker = hooker_cls(model)
        _, cache = hooker.forward_with_cache(dataset.X_test)

        for layer_idx in layer_indices:
            layer_act = _extract_layer_activations(hooker, cache, layer_idx)
            pooled_by_layer[layer_idx].append(layer_act)

        print(f"[{model_name}] dataset {ds_idx + 1}/{N_DATASETS} done")

    pooled_layers: list[np.ndarray] = []
    for layer_idx in layer_indices:
        pooled_layers.append(np.vstack(pooled_by_layer[layer_idx]))

    return pooled_layers


def summarize_cka(cka_matrix: np.ndarray) -> dict[str, Any]:
    adjacent_cka = [float(cka_matrix[i, i + 1]) for i in range(cka_matrix.shape[0] - 1)]
    start, end, mean_cka = find_computation_block(adjacent_cka, threshold=CKA_THRESHOLD)
    return {
        "cka_matrix": cka_matrix.tolist(),
        "adjacent_cka": adjacent_cka,
        "computation_block": {
            "start": int(start),
            "end": int(end),
            "mean_cka": float(mean_cka),
        },
    }


def _plot_one_heatmap(ax: Any, cka_matrix: np.ndarray, title: str) -> None:
    n_layers = cka_matrix.shape[0]
    labels = [f"L{i}" for i in range(n_layers)]
    im = ax.imshow(cka_matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(n_layers))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_cka_heatmaps(results: dict[str, dict[str, Any]], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    tabpfn_matrix = np.asarray(results["tabpfn"]["cka_matrix"], dtype=np.float64)
    tabicl_matrix = np.asarray(results["tabicl"]["cka_matrix"], dtype=np.float64)
    iltm_matrix = np.asarray(results["iltm"]["cka_matrix"], dtype=np.float64)

    _plot_one_heatmap(axes[0], tabpfn_matrix, "TabPFN CKA (12x12)")
    _plot_one_heatmap(axes[1], tabicl_matrix, "TabICL CKA (12x12)")
    _plot_one_heatmap(axes[2], iltm_matrix, "iLTM CKA (3x3, L0-L2)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"QUICK_RUN={QUICK_RUN}, datasets={N_DATASETS}, seed={SEED}")
    print("Running CKA collection for TabPFN, TabICL, iLTM...")

    tabpfn_layers = collect_model_activations("tabpfn")
    tabicl_layers = collect_model_activations("tabicl")
    iltm_layers = collect_model_activations("iltm")

    tabpfn_cka = compute_cka_matrix(tabpfn_layers)
    tabicl_cka = compute_cka_matrix(tabicl_layers)
    iltm_cka = compute_cka_matrix(iltm_layers)

    results: dict[str, dict[str, Any]] = {
        "tabpfn": summarize_cka(tabpfn_cka),
        "tabicl": summarize_cka(tabicl_cka),
        "iltm": summarize_cka(iltm_cka),
    }

    json_path = RESULTS_DIR / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_path = RESULTS_DIR / "cka_heatmaps.png"
    plot_cka_heatmaps(results, plot_path)

    for model_name in ["tabpfn", "tabicl", "iltm"]:
        block = results[model_name]["computation_block"]
        print(
            f"{model_name}: computation_block=[{block['start']}, {block['end']}], mean_cka={block['mean_cka']:.4f}"
        )

    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
