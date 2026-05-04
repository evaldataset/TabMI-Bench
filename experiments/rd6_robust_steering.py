# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportImplicitStringConcatenation=false

from rd5_config import cfg
from src.data.synthetic_generator import generate_linear_data
from src.hooks.steering_vector import TabPFNSteeringVector, compute_steering_effect
from src.hooks.tabicl_steering import TabICLSteeringVector

RANDOM_SEED = cfg.SEED
N_TRAIN = cfg.N_TRAIN
N_TEST = cfg.N_TEST

FULL_PAIRS = [(5.0, 1.0), (4.0, 1.0), (5.0, 2.0)]  # Reduced from 5 to 3 for tractability
FULL_LAYERS = [0, 4, 6, 8, 11]  # Reduced from 9 to 5 key layers
QUICK_PAIRS = FULL_PAIRS[:2]
QUICK_LAYERS = [0, 4, 6, 10]

CONTRASTIVE_PAIRS = QUICK_PAIRS if cfg.QUICK_RUN else FULL_PAIRS
TARGET_LAYERS = QUICK_LAYERS if cfg.QUICK_RUN else FULL_LAYERS

BETA_FIXED = 1.0
ALPHA_BASELINE = 3.0
LAMBDA_VALUES = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if np.isclose(norm, 0.0, atol=1e-12):
        raise ValueError("Averaged direction has near-zero norm; cannot normalize.")
    return (vec / norm).astype(np.float64)


def _to_json_effect(effect: dict[str, float]) -> dict[str, float]:
    return {
        "pearson_r": float(effect["pearson_r"]),
        "pearson_p": float(effect["pearson_p"]),
        "slope": float(effect["slope"]),
        "prediction_range": float(effect["prediction_range"]),
    }


def _generate_dataset(alpha: float) -> Any:
    return generate_linear_data(
        alpha=alpha,
        beta=BETA_FIXED,
        n_train=N_TRAIN,
        n_test=N_TEST,
        random_seed=RANDOM_SEED,
    )


def _build_pair_datasets() -> list[tuple[tuple[float, float], Any, Any]]:
    pair_datasets: list[tuple[tuple[float, float], Any, Any]] = []
    for alpha_high, alpha_low in CONTRASTIVE_PAIRS:
        ds_high = _generate_dataset(alpha=alpha_high)
        ds_low = _generate_dataset(alpha=alpha_low)
        pair_datasets.append(((alpha_high, alpha_low), ds_high, ds_low))
    return pair_datasets


def _extract_tabpfn_robust_direction(
    steerer: TabPFNSteeringVector,
    pair_datasets: list[tuple[tuple[float, float], Any, Any]],
    layer: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    directions: list[np.ndarray] = []
    pair_details: list[dict[str, Any]] = []

    for (alpha_high, alpha_low), ds_high, ds_low in pair_datasets:
        # Use 20% of high-condition train data as validation for direction extraction
        n_val = max(int(len(ds_high.y_train) * 0.2), 2)
        X_val = ds_high.X_train[:n_val]
        direction = steerer.extract_direction(
            ds_high.X_train[n_val:],
            ds_high.y_train[n_val:],
            ds_low.X_train[n_val:],
            ds_low.y_train[n_val:],
            ds_high.X_test,
            layer=layer,
            token_idx=-1,
            X_val=X_val,
        )
        directions.append(np.asarray(direction, dtype=np.float64))
        pair_details.append(
            {
                "alpha_high": float(alpha_high),
                "alpha_low": float(alpha_low),
                "direction_norm": float(np.linalg.norm(direction)),
            }
        )

    avg_direction = np.mean(np.stack(directions, axis=0), axis=0)
    return _normalize(avg_direction), pair_details


def _extract_tabicl_robust_direction(
    steerer: TabICLSteeringVector,
    pair_datasets: list[tuple[tuple[float, float], Any, Any]],
    layer: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    directions: list[np.ndarray] = []
    pair_details: list[dict[str, Any]] = []

    for (alpha_high, alpha_low), ds_high, ds_low in pair_datasets:
        n_val = max(int(len(ds_high.y_train) * 0.2), 2)
        X_val = ds_high.X_train[:n_val]
        direction = steerer.extract_direction(
            ds_high.X_train[n_val:],
            ds_high.y_train[n_val:],
            ds_low.X_train[n_val:],
            ds_low.y_train[n_val:],
            ds_high.X_test,
            layer=layer,
            X_val=X_val,
        )
        directions.append(np.asarray(direction, dtype=np.float64))
        pair_details.append(
            {
                "alpha_high": float(alpha_high),
                "alpha_low": float(alpha_low),
                "direction_norm": float(np.linalg.norm(direction)),
            }
        )

    avg_direction = np.mean(np.stack(directions, axis=0), axis=0)
    return _normalize(avg_direction), pair_details


def _run_tabpfn(
    pair_datasets: list[tuple[tuple[float, float], Any, Any]],
    baseline_ds: Any,
) -> dict[str, Any]:
    model = TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")
    steerer = TabPFNSteeringVector(model)

    robust_layer_results: dict[str, dict[str, Any]] = {}
    single_layer_results: dict[str, dict[str, Any]] = {}
    first_pair = [pair_datasets[0]]

    for layer in TARGET_LAYERS:
        robust_direction, robust_pairs = _extract_tabpfn_robust_direction(
            steerer=steerer,
            pair_datasets=pair_datasets,
            layer=layer,
        )

        model.fit(baseline_ds.X_train, baseline_ds.y_train)
        robust_sweep = steerer.sweep_lambda(
            baseline_ds.X_test,
            layer=layer,
            direction=robust_direction,
            lambdas=LAMBDA_VALUES,
        )
        robust_effect = compute_steering_effect(
            robust_sweep["lambdas"], robust_sweep["mean_preds"]
        )

        single_direction, single_pairs = _extract_tabpfn_robust_direction(
            steerer=steerer,
            pair_datasets=first_pair,
            layer=layer,
        )
        model.fit(baseline_ds.X_train, baseline_ds.y_train)
        single_sweep = steerer.sweep_lambda(
            baseline_ds.X_test,
            layer=layer,
            direction=single_direction,
            lambdas=LAMBDA_VALUES,
        )
        single_effect = compute_steering_effect(
            single_sweep["lambdas"], single_sweep["mean_preds"]
        )

        robust_layer_results[str(layer)] = {
            "effect": _to_json_effect(robust_effect),
            "abs_pearson_r": float(abs(robust_effect["pearson_r"])),
            "direction_shape": [int(v) for v in robust_direction.shape],
            "pair_details": robust_pairs,
        }
        single_layer_results[str(layer)] = {
            "effect": _to_json_effect(single_effect),
            "abs_pearson_r": float(abs(single_effect["pearson_r"])),
            "direction_shape": [int(v) for v in single_direction.shape],
            "pair_details": single_pairs,
        }

    best_layer_robust = max(
        TARGET_LAYERS,
        key=lambda l: (
            robust_layer_results[str(l)]["abs_pearson_r"],
            robust_layer_results[str(l)]["effect"]["prediction_range"],
        ),
    )
    best_layer_single = max(
        TARGET_LAYERS,
        key=lambda l: (
            single_layer_results[str(l)]["abs_pearson_r"],
            single_layer_results[str(l)]["effect"]["prediction_range"],
        ),
    )

    return {
        "layers_tested": TARGET_LAYERS,
        "robust": {
            "n_pairs": len(pair_datasets),
            "per_layer": robust_layer_results,
            "best_layer": int(best_layer_robust),
            "best_effect": robust_layer_results[str(best_layer_robust)]["effect"],
        },
        "single_pair_baseline": {
            "pair": {
                "alpha_high": float(first_pair[0][0][0]),
                "alpha_low": float(first_pair[0][0][1]),
            },
            "per_layer": single_layer_results,
            "best_layer": int(best_layer_single),
            "best_effect": single_layer_results[str(best_layer_single)]["effect"],
        },
        "comparison": {
            "best_abs_r_robust": robust_layer_results[str(best_layer_robust)][
                "abs_pearson_r"
            ],
            "best_abs_r_single_pair": single_layer_results[str(best_layer_single)][
                "abs_pearson_r"
            ],
            "delta_best_abs_r": robust_layer_results[str(best_layer_robust)][
                "abs_pearson_r"
            ]
            - single_layer_results[str(best_layer_single)]["abs_pearson_r"],
        },
    }


def _run_tabicl(
    pair_datasets: list[tuple[tuple[float, float], Any, Any]],
    baseline_ds: Any,
) -> dict[str, Any]:
    model = TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)
    steerer = TabICLSteeringVector(model)

    robust_layer_results: dict[str, dict[str, Any]] = {}
    single_layer_results: dict[str, dict[str, Any]] = {}
    first_pair = [pair_datasets[0]]

    for layer in TARGET_LAYERS:
        robust_direction, robust_pairs = _extract_tabicl_robust_direction(
            steerer=steerer,
            pair_datasets=pair_datasets,
            layer=layer,
        )

        model.fit(baseline_ds.X_train, baseline_ds.y_train)
        robust_sweep = steerer.sweep_lambda(
            baseline_ds.X_test,
            layer=layer,
            direction=robust_direction,
            lambdas=LAMBDA_VALUES,
        )
        robust_effect = compute_steering_effect(
            robust_sweep["lambdas"], robust_sweep["mean_preds"]
        )

        single_direction, single_pairs = _extract_tabicl_robust_direction(
            steerer=steerer,
            pair_datasets=first_pair,
            layer=layer,
        )
        model.fit(baseline_ds.X_train, baseline_ds.y_train)
        single_sweep = steerer.sweep_lambda(
            baseline_ds.X_test,
            layer=layer,
            direction=single_direction,
            lambdas=LAMBDA_VALUES,
        )
        single_effect = compute_steering_effect(
            single_sweep["lambdas"], single_sweep["mean_preds"]
        )

        robust_layer_results[str(layer)] = {
            "effect": _to_json_effect(robust_effect),
            "abs_pearson_r": float(abs(robust_effect["pearson_r"])),
            "direction_shape": [int(v) for v in robust_direction.shape],
            "pair_details": robust_pairs,
        }
        single_layer_results[str(layer)] = {
            "effect": _to_json_effect(single_effect),
            "abs_pearson_r": float(abs(single_effect["pearson_r"])),
            "direction_shape": [int(v) for v in single_direction.shape],
            "pair_details": single_pairs,
        }

    best_layer_robust = max(
        TARGET_LAYERS,
        key=lambda l: (
            robust_layer_results[str(l)]["abs_pearson_r"],
            robust_layer_results[str(l)]["effect"]["prediction_range"],
        ),
    )
    best_layer_single = max(
        TARGET_LAYERS,
        key=lambda l: (
            single_layer_results[str(l)]["abs_pearson_r"],
            single_layer_results[str(l)]["effect"]["prediction_range"],
        ),
    )

    return {
        "layers_tested": TARGET_LAYERS,
        "robust": {
            "n_pairs": len(pair_datasets),
            "per_layer": robust_layer_results,
            "best_layer": int(best_layer_robust),
            "best_effect": robust_layer_results[str(best_layer_robust)]["effect"],
        },
        "single_pair_baseline": {
            "pair": {
                "alpha_high": float(first_pair[0][0][0]),
                "alpha_low": float(first_pair[0][0][1]),
            },
            "per_layer": single_layer_results,
            "best_layer": int(best_layer_single),
            "best_effect": single_layer_results[str(best_layer_single)]["effect"],
        },
        "comparison": {
            "best_abs_r_robust": robust_layer_results[str(best_layer_robust)][
                "abs_pearson_r"
            ],
            "best_abs_r_single_pair": single_layer_results[str(best_layer_single)][
                "abs_pearson_r"
            ],
            "delta_best_abs_r": robust_layer_results[str(best_layer_robust)][
                "abs_pearson_r"
            ]
            - single_layer_results[str(best_layer_single)]["abs_pearson_r"],
        },
    }


def _plot_layer_abs_r_comparison(
    tabpfn_payload: dict[str, Any],
    tabicl_payload: dict[str, Any],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    model_payloads = [
        ("TabPFN", tabpfn_payload, "#1f77b4", "#1f77b4"),
        ("TabICL", tabicl_payload, "#d62728", "#d62728"),
    ]

    for ax, (name, payload, color_curve, color_line) in zip(axes, model_payloads):
        layers = np.asarray(payload["layers_tested"], dtype=np.int64)
        robust_abs_r = np.asarray(
            [
                payload["robust"]["per_layer"][str(int(layer))]["abs_pearson_r"]
                for layer in layers
            ],
            dtype=np.float64,
        )
        single_best_abs_r = float(payload["comparison"]["best_abs_r_single_pair"])

        ax.plot(
            layers,
            robust_abs_r,
            marker="o",
            linewidth=2,
            color=color_curve,
            label="Robust multi-pair |r|",
        )
        ax.axhline(
            single_best_abs_r,
            linestyle="--",
            linewidth=1.7,
            color=color_line,
            alpha=0.8,
            label="Single-pair baseline (best |r|)",
        )
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_xticks(layers.tolist())
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("|Pearson r|")
    fig.suptitle(
        "RD6 Robust Steering: Layer Sweep vs Single-Pair Baseline", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results_dir = ROOT / "results" / "rd6" / "robust_steering"
    results_dir.mkdir(parents=True, exist_ok=True)

    pair_datasets = _build_pair_datasets()
    baseline_ds = _generate_dataset(alpha=ALPHA_BASELINE)

    print("=" * 72)
    print("RD6 Robust Steering: Multi-Pair Direction Averaging + Layer Sweep")
    print("=" * 72)
    print(f"QUICK_RUN={cfg.QUICK_RUN}, SEED={RANDOM_SEED}")
    print(f"pairs={CONTRASTIVE_PAIRS}")
    print(f"layers={TARGET_LAYERS}")

    print("\n[1/4] Running TabPFN robust steering...")
    tabpfn_payload = _run_tabpfn(pair_datasets=pair_datasets, baseline_ds=baseline_ds)
    print(
        "  TabPFN best robust: "
        f"L{tabpfn_payload['robust']['best_layer']}, "
        f"|r|={abs(float(tabpfn_payload['robust']['best_effect']['pearson_r'])):.4f}"
    )

    print("\n[2/4] Running TabICL robust steering...")
    tabicl_payload = _run_tabicl(pair_datasets=pair_datasets, baseline_ds=baseline_ds)
    print(
        "  TabICL best robust: "
        f"L{tabicl_payload['robust']['best_layer']}, "
        f"|r|={abs(float(tabicl_payload['robust']['best_effect']['pearson_r'])):.4f}"
    )

    print("\n[3/4] Saving JSON results...")
    payload: dict[str, Any] = {
        "quick_run": bool(cfg.QUICK_RUN),
        "random_seed": int(RANDOM_SEED),
        "n_train": int(N_TRAIN),
        "n_test": int(N_TEST),
        "beta_fixed": float(BETA_FIXED),
        "alpha_baseline": float(ALPHA_BASELINE),
        "contrastive_pairs": [
            {"alpha_high": float(h), "alpha_low": float(l)}
            for h, l in CONTRASTIVE_PAIRS
        ],
        "layers_tested": [int(layer) for layer in TARGET_LAYERS],
        "lambda_values": [float(v) for v in LAMBDA_VALUES],
        "tabpfn": tabpfn_payload,
        "tabicl": tabicl_payload,
    }
    json_path = results_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved JSON: {json_path}")

    print("\n[4/4] Saving comparison plot...")
    plot_path = results_dir / "robust_vs_single_pair.png"
    _plot_layer_abs_r_comparison(
        tabpfn_payload=tabpfn_payload,
        tabicl_payload=tabicl_payload,
        save_path=plot_path,
    )
    print(f"  Saved plot: {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
