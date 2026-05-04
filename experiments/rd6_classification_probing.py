# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from iltm import iLTMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from rd5_config import cfg
from src.data.real_world_datasets import (
    RealWorldDataset,
    load_adult_income,
    load_breast_cancer,
    load_iris_binary,
)
from src.hooks.iltm_hooker import iLTMHookedModel
from src.hooks.tabicl_hooker import TabICLHookedModel
from src.hooks.tabpfn_hooker import TabPFNHookedModel


RESULTS_DIR = ROOT / "results" / "rd6" / "classification_probing"
MODEL_ORDER = ["tabpfn", "tabicl", "iltm"]
MODEL_TITLE = {"tabpfn": "TabPFN", "tabicl": "TabICL", "iltm": "iLTM"}


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


def _mcfadden_pseudo_r2(
    y_true: np.ndarray,
    y_train: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    eps = 1e-12
    y_eval = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_ref = np.asarray(y_train, dtype=np.int64).reshape(-1)
    proba = np.asarray(y_pred_proba, dtype=np.float64)

    if proba.ndim != 2 or proba.shape[0] != y_eval.shape[0]:
        return float("nan")

    if proba.shape[1] < 2:
        return float("nan")

    probs_true = np.clip(proba[np.arange(y_eval.size), y_eval], eps, 1.0)
    ll_model = float(np.sum(np.log(probs_true)))

    p1 = float(np.mean(y_ref))
    p1 = min(max(p1, eps), 1.0 - eps)
    null_probs = np.where(y_eval == 1, p1, 1.0 - p1)
    ll_null = float(np.sum(np.log(np.clip(null_probs, eps, 1.0))))

    if abs(ll_null) <= eps:
        return float("nan")

    return 1.0 - (ll_model / ll_null)


def _probe_layer_classification(
    activations: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    x = np.asarray(activations, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)

    if x.shape[0] != y.shape[0] or x.shape[0] < 5:
        return {
            "accuracy": float("nan"),
            "f1_macro": float("nan"),
            "pseudo_r2_mcfadden": float("nan"),
        }

    unique, counts = np.unique(y, return_counts=True)
    if unique.size < 2:
        return {
            "accuracy": float("nan"),
            "f1_macro": float("nan"),
            "pseudo_r2_mcfadden": float("nan"),
        }

    stratify: np.ndarray | None = y if int(np.min(counts)) >= 2 else None
    try:
        x_tr, x_te, y_tr, y_te = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=cfg.SEED,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        x_tr, x_te, y_tr, y_te = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=cfg.SEED,
            shuffle=True,
            stratify=None,
        )

    if np.unique(y_tr).size < 2 or np.unique(y_te).size < 1:
        return {
            "accuracy": float("nan"),
            "f1_macro": float("nan"),
            "pseudo_r2_mcfadden": float("nan"),
        }

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_tr, y_tr)
    y_pred = clf.predict(x_te)
    y_proba = clf.predict_proba(x_te)

    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "pseudo_r2_mcfadden": float(_mcfadden_pseudo_r2(y_te, y_tr, y_proba)),
    }


def _probe_layer_classification_split(
    act_train: np.ndarray,
    y_train: np.ndarray,
    act_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Probe with proper train/test split: train probe on train activations,
    evaluate on test activations. Avoids re-splitting test data internally."""
    nan_result = {"accuracy": float("nan"), "f1_macro": float("nan"), "pseudo_r2_mcfadden": float("nan")}

    x_tr = np.asarray(act_train, dtype=np.float32)
    y_tr = np.asarray(y_train, dtype=np.int64).reshape(-1)
    x_te = np.asarray(act_test, dtype=np.float32)
    y_te = np.asarray(y_test, dtype=np.int64).reshape(-1)

    if x_tr.shape[0] < 5 or x_te.shape[0] < 2:
        return nan_result
    if np.unique(y_tr).size < 2 or np.unique(y_te).size < 1:
        return nan_result

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_tr, y_tr)
    y_pred = clf.predict(x_te)
    y_proba = clf.predict_proba(x_te)

    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "pseudo_r2_mcfadden": float(_mcfadden_pseudo_r2(y_te, y_tr, y_proba)),
    }


def _load_datasets() -> list[RealWorldDataset]:
    if cfg.QUICK_RUN:
        n_train = 50
        n_test = 20
        loader_specs: list[tuple[str, Callable[..., RealWorldDataset]]] = [
            ("breast_cancer", load_breast_cancer),
            ("iris_binary", load_iris_binary),
        ]
    else:
        n_train = cfg.N_TRAIN
        n_test = cfg.N_TEST
        loader_specs = [
            ("breast_cancer", load_breast_cancer),
            ("iris_binary", load_iris_binary),
            ("adult_income", load_adult_income),
        ]

    datasets: list[RealWorldDataset] = []
    for name, loader in loader_specs:
        try:
            ds = loader(n_train=n_train, n_test=n_test, random_seed=cfg.SEED)
            ds.name = name
            datasets.append(ds)
        except Exception as exc:
            print(f"[dataset:{name}] warning: skipped due to load error: {exc}")
    return datasets


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

    y_train = np.asarray(dataset.y_train, dtype=np.float64).reshape(-1)
    y_test = np.asarray(dataset.y_test, dtype=np.float64).reshape(-1)

    model.fit(dataset.X_train, y_train)
    hooker = hooker_cls(model)
    # Extract activations from BOTH splits: train for probe fitting, test for evaluation
    _, cache_train = hooker.forward_with_cache(dataset.X_train)
    _, cache_test = hooker.forward_with_cache(dataset.X_test)

    layers = cfg.layer_indices(model_name)
    accuracy_by_layer: list[float] = []
    f1_by_layer: list[float] = []
    pseudo_r2_by_layer: list[float] = []

    y_train_int = np.asarray(np.rint(y_train), dtype=np.int64)
    y_test_int = np.asarray(np.rint(y_test), dtype=np.int64)
    for layer_idx in layers:
        act_train = _extract_activations(model_name, hooker, cache_train, layer_idx)
        act_test = _extract_activations(model_name, hooker, cache_test, layer_idx)
        # Probe: train on train activations, evaluate on test activations
        metrics = _probe_layer_classification_split(act_train, y_train_int, act_test, y_test_int)
        accuracy_by_layer.append(metrics["accuracy"])
        f1_by_layer.append(metrics["f1_macro"])
        pseudo_r2_by_layer.append(metrics["pseudo_r2_mcfadden"])

    acc_arr = np.asarray(accuracy_by_layer, dtype=np.float64)
    best_idx = int(np.nanargmax(acc_arr)) if np.isfinite(acc_arr).any() else 0

    return {
        "layer_indices": [int(v) for v in layers],
        "accuracy_by_layer": [float(v) for v in accuracy_by_layer],
        "f1_macro_by_layer": [float(v) for v in f1_by_layer],
        "pseudo_r2_mcfadden_by_layer": [float(v) for v in pseudo_r2_by_layer],
        "best_layer_by_accuracy": int(layers[best_idx]),
        "best_accuracy": float(acc_arr[best_idx]),
    }


def _plot_metric_grid(
    flat_results: dict[str, dict[str, Any]],
    metric_key: str,
    y_label: str,
    title: str,
    save_path: Path,
) -> None:
    dataset_names = list(flat_results.keys())
    n_datasets = len(dataset_names)
    n_cols = 2 if n_datasets > 1 else 1
    n_rows = int(math.ceil(n_datasets / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.0 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )

    for idx, dataset_name in enumerate(dataset_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        for model_name in MODEL_ORDER:
            payload = flat_results.get(dataset_name, {}).get(model_name, {})
            if not payload or "error" in payload:
                continue

            layers = np.asarray(payload.get("layer_indices", []), dtype=np.int32)
            metric = np.asarray(payload.get(metric_key, []), dtype=np.float64)
            if layers.size == 0 or metric.size == 0:
                continue

            ax.plot(
                layers,
                metric,
                marker="o",
                linewidth=1.8,
                label=MODEL_TITLE[model_name],
            )

        ax.set_title(dataset_name)
        ax.set_xlabel("Layer")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best")

    for idx in range(n_datasets, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = _load_datasets()
    n_train = 50 if cfg.QUICK_RUN else cfg.N_TRAIN
    n_test = 20 if cfg.QUICK_RUN else cfg.N_TEST
    print(
        f"QUICK_RUN={cfg.QUICK_RUN}, seed={cfg.SEED}, n_train={n_train}, n_test={n_test}"
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
        "n_train": int(n_train),
        "n_test": int(n_test),
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

    accuracy_plot = RESULTS_DIR / "accuracy_by_dataset.png"
    _plot_metric_grid(
        flat_results=flat_results,
        metric_key="accuracy_by_layer",
        y_label="Accuracy",
        title="RD6 Classification Probing: Accuracy by Layer",
        save_path=accuracy_plot,
    )

    f1_plot = RESULTS_DIR / "f1_macro_by_dataset.png"
    _plot_metric_grid(
        flat_results=flat_results,
        metric_key="f1_macro_by_layer",
        y_label="F1 (macro)",
        title="RD6 Classification Probing: F1 by Layer",
        save_path=f1_plot,
    )

    print(f"Saved: {json_path}")
    print(f"Saved: {accuracy_plot}")
    print(f"Saved: {f1_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
