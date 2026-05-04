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
from typing import Any

import numpy as np
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

from src.data.synthetic_generator import generate_linear_data
from src.hooks.steering_vector import TabPFNSteeringVector, compute_steering_effect
from src.hooks.tabicl_steering import (
    TabICLSteeringVector,
    compute_steering_effect as tabicl_compute_effect,
)
from rd5_config import cfg

QUICK_RUN = cfg.QUICK_RUN
RANDOM_SEED = cfg.SEED

N_TRAIN = cfg.N_TRAIN
N_TEST = cfg.N_TEST

TARGET_LAYER_TABPFN = 6
TARGET_LAYER_TABICL = 5
TABICL_LAYER_CANDIDATES = [2, 4, 5, 6, 8, 10]
LAMBDA_VALUES = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]


def _to_json_mean_preds(mean_preds: dict[float, float]) -> dict[str, float]:
    return {str(float(k)): float(v) for k, v in mean_preds.items()}


def _run_tabpfn(
    ds_high: Any,
    ds_low: Any,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    model = TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")
    steerer = TabPFNSteeringVector(model)

    n_val = max(int(len(ds_high.y_train) * 0.2), 2)
    X_val = ds_high.X_train[:n_val]
    direction = steerer.extract_direction(
        ds_high.X_train[n_val:],
        ds_high.y_train[n_val:],
        ds_low.X_train[n_val:],
        ds_low.y_train[n_val:],
        ds_high.X_test,
        layer=TARGET_LAYER_TABPFN,
        X_val=X_val,
    )
    sweep = steerer.sweep_lambda(
        ds_high.X_test,
        layer=TARGET_LAYER_TABPFN,
        direction=direction,
        lambdas=LAMBDA_VALUES,
    )
    effect = compute_steering_effect(sweep["lambdas"], sweep["mean_preds"])

    payload = {
        "layer": TARGET_LAYER_TABPFN,
        "direction_shape": [int(dim) for dim in direction.shape],
        "lambda_sweep": {
            "lambdas": [float(v) for v in sweep["lambdas"]],
            "mean_preds": _to_json_mean_preds(sweep["mean_preds"]),
        },
        "effect": {
            "pearson_r": float(effect["pearson_r"]),
            "slope": float(effect["slope"]),
            "prediction_range": float(effect["prediction_range"]),
        },
    }
    return payload, sweep, direction


def _run_tabicl(
    ds_high: Any,
    ds_low: Any,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], int, dict[str, Any]]:
    model = TabICLRegressor(device=cfg.DEVICE, random_state=RANDOM_SEED)
    steerer = TabICLSteeringVector(model)

    layer_sweep_for_json: dict[str, dict[str, float]] = {}
    layer_run_artifacts: dict[int, dict[str, Any]] = {}

    n_val = max(int(len(ds_high.y_train) * 0.2), 2)
    X_val_icl = ds_high.X_train[:n_val]
    for layer in TABICL_LAYER_CANDIDATES:
        direction = steerer.extract_direction(
            ds_high.X_train[n_val:],
            ds_high.y_train[n_val:],
            ds_low.X_train[n_val:],
            ds_low.y_train[n_val:],
            ds_high.X_test,
            layer=layer,
            X_val=X_val_icl,
        )
        sweep = steerer.sweep_lambda(
            ds_high.X_test,
            layer=layer,
            direction=direction,
            lambdas=LAMBDA_VALUES,
        )
        effect = tabicl_compute_effect(sweep["lambdas"], sweep["mean_preds"])

        layer_run_artifacts[layer] = {
            "direction": direction,
            "sweep": sweep,
            "effect": effect,
        }
        layer_sweep_for_json[str(layer)] = {
            "pearson_r": float(effect["pearson_r"]),
            "slope": float(effect["slope"]),
            "prediction_range": float(effect["prediction_range"]),
        }

    best_layer = max(
        TABICL_LAYER_CANDIDATES,
        key=lambda l: (
            abs(float(layer_run_artifacts[l]["effect"]["pearson_r"])),
            float(layer_run_artifacts[l]["effect"]["prediction_range"]),
        ),
    )
    best_effect = layer_run_artifacts[best_layer]["effect"]

    payload = {
        "target_layer_tested": TARGET_LAYER_TABICL,
        "best_layer": int(best_layer),
        "layer_sweep": layer_sweep_for_json,
        "best_effect": {
            "pearson_r": float(best_effect["pearson_r"]),
            "slope": float(best_effect["slope"]),
            "prediction_range": float(best_effect["prediction_range"]),
        },
    }
    return payload, layer_run_artifacts, best_layer, layer_run_artifacts[best_layer]


def _build_hypotheses(
    tabpfn_effect: dict[str, float],
    tabicl_best_layer: int,
    tabicl_best_effect: dict[str, float],
) -> dict[str, dict[str, Any]]:
    h4_supported = (
        abs(float(tabicl_best_effect["pearson_r"])) >= 0.7
        and abs(float(tabicl_best_effect["slope"])) > 0.0
        and float(tabicl_best_effect["prediction_range"]) > 0.0
    )
    h4_evidence = (
        f"Best TabICL layer={tabicl_best_layer} with "
        f"r={float(tabicl_best_effect['pearson_r']):.4f}, "
        f"slope={float(tabicl_best_effect['slope']):.4f}, "
        f"range={float(tabicl_best_effect['prediction_range']):.4f}."
    )

    tabpfn_strength = float(abs(tabpfn_effect["slope"]))
    tabicl_strength = float(abs(tabicl_best_effect["slope"]))
    h5_supported = tabicl_strength > tabpfn_strength
    direction_word = "larger" if h5_supported else "smaller"
    h5_evidence = (
        "Using |slope| as steering magnitude proxy, TabICL (hidden_dim=512) "
        f"shows {direction_word} effect than TabPFN (hidden_dim=192): "
        f"|slope|_TabICL={tabicl_strength:.4f} vs |slope|_TabPFN={tabpfn_strength:.4f}."
    )

    return {
        "H4": {
            "description": "TabICL steering possible",
            "supported": bool(h4_supported),
            "evidence": h4_evidence,
        },
        "H5": {
            "description": "Model size vs steering effect",
            "supported": bool(h5_supported),
            "evidence": h5_evidence,
        },
    }


def _plot_comparison(
    tabpfn_sweep: dict[str, Any],
    tabicl_best_sweep: dict[str, Any],
    tabicl_layer_effects: dict[int, dict[str, Any]],
    tabicl_best_layer: int,
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.2])

    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, :])

    lambdas = np.asarray(tabpfn_sweep["lambdas"], dtype=np.float64)
    pfn_means = np.asarray(
        [tabpfn_sweep["mean_preds"][float(lam)] for lam in lambdas], dtype=np.float64
    )
    ax_left.scatter(lambdas, pfn_means, color="#1f77b4", s=45, zorder=5)
    ax_left.plot(lambdas, pfn_means, color="#1f77b4", linewidth=2, zorder=4)
    ax_left.set_title("TabPFN (Layer 6)", fontweight="bold")
    ax_left.set_xlabel("lambda")
    ax_left.set_ylabel("Mean prediction")
    ax_left.grid(True, alpha=0.25)

    tabicl_lambdas = np.asarray(tabicl_best_sweep["lambdas"], dtype=np.float64)
    tabicl_means = np.asarray(
        [tabicl_best_sweep["mean_preds"][float(lam)] for lam in tabicl_lambdas],
        dtype=np.float64,
    )
    ax_right.scatter(tabicl_lambdas, tabicl_means, color="#d62728", s=45, zorder=5)
    ax_right.plot(tabicl_lambdas, tabicl_means, color="#d62728", linewidth=2, zorder=4)
    ax_right.set_title(f"TabICL (Best Layer {tabicl_best_layer})", fontweight="bold")
    ax_right.set_xlabel("lambda")
    ax_right.set_ylabel("Mean prediction")
    ax_right.grid(True, alpha=0.25)

    layers = np.asarray(TABICL_LAYER_CANDIDATES, dtype=np.int64)
    layer_rs = np.asarray(
        [tabicl_layer_effects[int(layer)]["effect"]["pearson_r"] for layer in layers],
        dtype=np.float64,
    )
    ax_bottom.plot(
        layers,
        layer_rs,
        marker="o",
        linewidth=2,
        color="#2ca02c",
        label="TabICL Pearson r",
    )
    ax_bottom.axhline(0.0, linestyle="--", linewidth=1, color="gray")
    ax_bottom.set_xticks(TABICL_LAYER_CANDIDATES)
    ax_bottom.set_title("TabICL Layer Sweep (Pearson r)", fontweight="bold")
    ax_bottom.set_xlabel("Layer")
    ax_bottom.set_ylabel("Pearson r")
    ax_bottom.grid(True, alpha=0.25)
    ax_bottom.legend()

    fig.suptitle("RD5 Steering Comparison: TabPFN vs TabICL", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results_dir = ROOT / "results" / "rd5" / "steering"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RD5 Steering Vector Comparison: TabPFN vs TabICL")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}")

    ds_high = generate_linear_data(
        alpha=5.0,
        beta=1.0,
        n_train=N_TRAIN,
        n_test=N_TEST,
        random_seed=RANDOM_SEED,
    )
    ds_low = generate_linear_data(
        alpha=1.0,
        beta=1.0,
        n_train=N_TRAIN,
        n_test=N_TEST,
        random_seed=RANDOM_SEED,
    )

    print("\n[1/4] Running TabPFN steering...")
    tabpfn_payload, tabpfn_sweep, _ = _run_tabpfn(ds_high, ds_low)
    print(
        "  TabPFN effect: "
        f"r={tabpfn_payload['effect']['pearson_r']:.4f}, "
        f"slope={tabpfn_payload['effect']['slope']:.4f}, "
        f"range={tabpfn_payload['effect']['prediction_range']:.4f}"
    )

    print("\n[2/4] Running TabICL steering layer sweep...")
    tabicl_payload, tabicl_runs, tabicl_best_layer, tabicl_best_artifact = _run_tabicl(
        ds_high,
        ds_low,
    )
    best_effect = tabicl_payload["best_effect"]
    print(
        "  TabICL best effect: "
        f"layer={tabicl_best_layer}, "
        f"r={best_effect['pearson_r']:.4f}, "
        f"slope={best_effect['slope']:.4f}, "
        f"range={best_effect['prediction_range']:.4f}"
    )

    print("\n[3/4] Building hypotheses and saving JSON...")
    hypotheses = _build_hypotheses(
        tabpfn_effect=tabpfn_payload["effect"],
        tabicl_best_layer=tabicl_best_layer,
        tabicl_best_effect=tabicl_payload["best_effect"],
    )

    payload: dict[str, Any] = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "lambdas": [float(v) for v in LAMBDA_VALUES],
        "tabpfn": tabpfn_payload,
        "tabicl": tabicl_payload,
        "hypotheses": hypotheses,
    }

    json_path = results_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved JSON: {json_path}")

    print("\n[4/4] Saving comparison plot...")
    plot_path = results_dir / "steering_comparison.png"
    _plot_comparison(
        tabpfn_sweep=tabpfn_sweep,
        tabicl_best_sweep=tabicl_best_artifact["sweep"],
        tabicl_layer_effects=tabicl_runs,
        tabicl_best_layer=tabicl_best_layer,
        save_path=plot_path,
    )
    print(f"  Saved plot: {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
