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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rd5_config import cfg  # noqa: E402
from src.data.synthetic_generator import generate_linear_data  # noqa: E402
from src.hooks.steering_vector import compute_steering_effect  # noqa: E402
from src.hooks.tabicl_steering import TabICLSteeringVector  # noqa: E402

RESULTS_DIR = ROOT / "results" / "rd6" / "tabicl_l6_anomaly"
SEEDS = [42, 123, 456]
LAYERS = list(range(12))
LAMBDA_VALUES = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

CONTRASTIVE_PAIRS = [
    (5.0, 1.0),
    (4.0, 1.0),
    (5.0, 2.0),
    (4.0, 2.0),
    (3.5, 1.0),
]


def _build_dataset(alpha: float, seed: int) -> Any:
    return generate_linear_data(
        alpha=alpha,
        beta=1.0,
        n_train=cfg.N_TRAIN,
        n_test=cfg.N_TEST,
        random_seed=seed,
    )


def _run_seed(seed: int) -> dict[str, Any]:
    model = TabICLRegressor(device=cfg.DEVICE, random_state=seed)
    steerer = TabICLSteeringVector(model)

    baseline_ds = _build_dataset(alpha=3.0, seed=seed)

    per_layer_pair_abs_r: dict[int, list[float]] = {layer: [] for layer in LAYERS}
    per_layer_pair_slope: dict[int, list[float]] = {layer: [] for layer in LAYERS}
    per_layer_pair_range: dict[int, list[float]] = {layer: [] for layer in LAYERS}
    per_layer_pair_dir_norm: dict[int, list[float]] = {layer: [] for layer in LAYERS}

    for alpha_high, alpha_low in CONTRASTIVE_PAIRS:
        ds_high = _build_dataset(alpha=alpha_high, seed=seed)
        ds_low = _build_dataset(alpha=alpha_low, seed=seed)

        for layer in LAYERS:
            direction = steerer.extract_direction(
                ds_high.X_train,
                ds_high.y_train,
                ds_low.X_train,
                ds_low.y_train,
                baseline_ds.X_test,
                layer=layer,
                X_val=baseline_ds.X_train,  # Use train features (no test-set leakage).
            )

            model.fit(baseline_ds.X_train, baseline_ds.y_train)
            sweep = steerer.sweep_lambda(
                baseline_ds.X_test,
                layer=layer,
                direction=direction,
                lambdas=LAMBDA_VALUES,
            )
            effect = compute_steering_effect(sweep["lambdas"], sweep["mean_preds"])

            per_layer_pair_abs_r[layer].append(float(abs(effect["pearson_r"])))
            per_layer_pair_slope[layer].append(float(effect["slope"]))
            per_layer_pair_range[layer].append(float(effect["prediction_range"]))
            per_layer_pair_dir_norm[layer].append(float(np.linalg.norm(direction)))

    layer_summary: dict[str, Any] = {}
    for layer in LAYERS:
        abs_r_vals = np.asarray(per_layer_pair_abs_r[layer], dtype=np.float64)
        slope_vals = np.asarray(per_layer_pair_slope[layer], dtype=np.float64)
        range_vals = np.asarray(per_layer_pair_range[layer], dtype=np.float64)
        dir_norm_vals = np.asarray(per_layer_pair_dir_norm[layer], dtype=np.float64)
        layer_summary[str(layer)] = {
            "abs_r_values": abs_r_vals.tolist(),
            "abs_r_mean": float(np.mean(abs_r_vals)),
            "abs_r_std": float(np.std(abs_r_vals)),
            "slope_mean": float(np.mean(slope_vals)),
            "slope_std": float(np.std(slope_vals)),
            "range_mean": float(np.mean(range_vals)),
            "range_std": float(np.std(range_vals)),
            "direction_norm_mean": float(np.mean(dir_norm_vals)),
            "direction_norm_std": float(np.std(dir_norm_vals)),
        }

    anomaly_scores = {
        str(layer): float(
            layer_summary[str(layer)]["abs_r_std"]
            + max(0.0, 0.9 - layer_summary[str(layer)]["abs_r_mean"])
        )
        for layer in LAYERS
    }
    best_anomaly_layer = int(
        max(LAYERS, key=lambda layer_idx: anomaly_scores[str(layer_idx)])
    )

    return {
        "seed": int(seed),
        "layers": LAYERS,
        "contrastive_pairs": [
            {"alpha_high": float(high), "alpha_low": float(low)}
            for high, low in CONTRASTIVE_PAIRS
        ],
        "per_layer": layer_summary,
        "anomaly_scores": anomaly_scores,
        "best_anomaly_layer": best_anomaly_layer,
    }


def _plot(seed_payloads: list[dict[str, Any]], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    layers = np.asarray(LAYERS)

    mean_abs_r_per_seed = []
    std_abs_r_per_seed = []
    for payload in seed_payloads:
        means = [
            payload["per_layer"][str(layer_idx)]["abs_r_mean"] for layer_idx in LAYERS
        ]
        stds = [
            payload["per_layer"][str(layer_idx)]["abs_r_std"] for layer_idx in LAYERS
        ]
        mean_abs_r_per_seed.append(means)
        std_abs_r_per_seed.append(stds)

    mean_abs_r_arr = np.asarray(mean_abs_r_per_seed, dtype=np.float64)
    std_abs_r_arr = np.asarray(std_abs_r_per_seed, dtype=np.float64)

    mean_curve = mean_abs_r_arr.mean(axis=0)
    mean_curve_std = mean_abs_r_arr.std(axis=0)
    var_curve = std_abs_r_arr.mean(axis=0)
    var_curve_std = std_abs_r_arr.std(axis=0)

    ax = axes[0]
    ax.plot(layers, mean_curve, marker="o", linewidth=2, label="mean |r|")
    ax.fill_between(
        layers,
        mean_curve - mean_curve_std,
        mean_curve + mean_curve_std,
        alpha=0.2,
        label="across-seed std",
    )
    ax.axvline(6, linestyle="--", color="red", alpha=0.8, label="L6")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Steering strength |r|")
    ax.set_title("TabICL steering strength by layer")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(
        layers, var_curve, marker="s", linewidth=2, label="pair variance (std of |r|)"
    )
    ax.fill_between(
        layers,
        var_curve - var_curve_std,
        var_curve + var_curve_std,
        alpha=0.2,
        label="across-seed std",
    )
    ax.axvline(6, linestyle="--", color="red", alpha=0.8, label="L6")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Instability")
    ax.set_title("TabICL layer instability (pair variance)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("RD6: TabICL L6 anomaly diagnosis")
    print(f"QUICK_RUN={cfg.QUICK_RUN}, n_train={cfg.N_TRAIN}, n_test={cfg.N_TEST}")
    print(f"seeds={SEEDS}, pairs={len(CONTRASTIVE_PAIRS)}, layers={len(LAYERS)}")

    seed_payloads: list[dict[str, Any]] = []
    for seed in SEEDS:
        print(f"\n[seed={seed}] running...")
        payload = _run_seed(seed)
        seed_payloads.append(payload)
        print(f"  best anomaly layer: L{payload['best_anomaly_layer']}")

    agg_best_layers = [p["best_anomaly_layer"] for p in seed_payloads]
    l6_rank_scores = [p["anomaly_scores"]["6"] for p in seed_payloads]

    per_layer_mean_abs_r: dict[str, float] = {}
    per_layer_mean_std_abs_r: dict[str, float] = {}
    for layer in LAYERS:
        mean_vals = [p["per_layer"][str(layer)]["abs_r_mean"] for p in seed_payloads]
        std_vals = [p["per_layer"][str(layer)]["abs_r_std"] for p in seed_payloads]
        per_layer_mean_abs_r[str(layer)] = float(np.mean(mean_vals))
        per_layer_mean_std_abs_r[str(layer)] = float(np.mean(std_vals))

    final_payload = {
        "quick_run": bool(cfg.QUICK_RUN),
        "n_train": int(cfg.N_TRAIN),
        "n_test": int(cfg.N_TEST),
        "seeds": SEEDS,
        "layers": LAYERS,
        "contrastive_pairs": [
            {"alpha_high": float(high), "alpha_low": float(low)}
            for high, low in CONTRASTIVE_PAIRS
        ],
        "per_seed": seed_payloads,
        "aggregated": {
            "best_anomaly_layers": agg_best_layers,
            "mean_l6_anomaly_score": float(np.mean(l6_rank_scores)),
            "std_l6_anomaly_score": float(np.std(l6_rank_scores)),
            "per_layer_mean_abs_r": per_layer_mean_abs_r,
            "per_layer_mean_std_abs_r": per_layer_mean_std_abs_r,
        },
    }

    json_path = RESULTS_DIR / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    plot_path = RESULTS_DIR / "l6_anomaly_profile.png"
    _plot(seed_payloads, plot_path)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
