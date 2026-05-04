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
from src.hooks.tabicl_hooker import TabICLHookedModel  # noqa: E402
from src.hooks.tabicl_steering import TabICLSteeringVector  # noqa: E402

RESULTS_DIR = ROOT / "results" / "rd6" / "tabicl_l5l6_zoom"
SEEDS = [42, 123, 456]
LAYERS = [4, 5, 6, 7]
LAMBDA_VALUES = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
CONTRASTIVE_PAIRS = [(5.0, 1.0), (4.0, 1.0), (5.0, 2.0), (4.0, 2.0), (3.5, 1.0)]


def _build_dataset(alpha: float, seed: int) -> Any:
    return generate_linear_data(
        alpha=alpha,
        beta=1.0,
        n_train=cfg.N_TRAIN,
        n_test=cfg.N_TEST,
        random_seed=seed,
    )


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    un = float(np.linalg.norm(u))
    vn = float(np.linalg.norm(v))
    if un < 1e-12 or vn < 1e-12:
        return 0.0
    return float(np.dot(u, v) / (un * vn))


def _layer_activation_profile(
    seed: int,
    layer: int,
    ds_high: Any,
    ds_low: Any,
    x_ref: np.ndarray,
) -> dict[str, float]:
    model = TabICLRegressor(device=cfg.DEVICE, random_state=seed)

    model.fit(ds_high.X_train, ds_high.y_train)
    hook_high = TabICLHookedModel(model)
    _, cache_high = hook_high.forward_with_cache(x_ref)
    act_high = np.asarray(
        hook_high.get_layer_activations(cache_high, layer), dtype=np.float64
    )

    model.fit(ds_low.X_train, ds_low.y_train)
    hook_low = TabICLHookedModel(model)
    _, cache_low = hook_low.forward_with_cache(x_ref)
    act_low = np.asarray(
        hook_low.get_layer_activations(cache_low, layer), dtype=np.float64
    )

    mean_high = act_high.mean(axis=0)
    mean_low = act_low.mean(axis=0)
    delta = mean_high - mean_low

    return {
        "mean_norm_high": float(np.linalg.norm(mean_high)),
        "mean_norm_low": float(np.linalg.norm(mean_low)),
        "delta_norm": float(np.linalg.norm(delta)),
        "mean_cosine": _cosine(mean_high, mean_low),
    }


def _run_seed(seed: int) -> dict[str, Any]:
    model = TabICLRegressor(device=cfg.DEVICE, random_state=seed)
    steerer = TabICLSteeringVector(model)
    baseline_ds = _build_dataset(alpha=3.0, seed=seed)

    per_layer_abs_r: dict[int, list[float]] = {layer: [] for layer in LAYERS}
    per_layer_slope: dict[int, list[float]] = {layer: [] for layer in LAYERS}
    per_layer_range: dict[int, list[float]] = {layer: [] for layer in LAYERS}
    per_layer_activation: dict[int, list[dict[str, float]]] = {
        layer: [] for layer in LAYERS
    }

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

            profile = _layer_activation_profile(
                seed=seed,
                layer=layer,
                ds_high=ds_high,
                ds_low=ds_low,
                x_ref=baseline_ds.X_test,
            )

            per_layer_abs_r[layer].append(float(abs(effect["pearson_r"])))
            per_layer_slope[layer].append(float(effect["slope"]))
            per_layer_range[layer].append(float(effect["prediction_range"]))
            per_layer_activation[layer].append(profile)

    per_layer: dict[str, Any] = {}
    for layer in LAYERS:
        abs_r = np.asarray(per_layer_abs_r[layer], dtype=np.float64)
        slopes = np.asarray(per_layer_slope[layer], dtype=np.float64)
        ranges = np.asarray(per_layer_range[layer], dtype=np.float64)
        delta_norms = np.asarray(
            [p["delta_norm"] for p in per_layer_activation[layer]], dtype=np.float64
        )
        cosines = np.asarray(
            [p["mean_cosine"] for p in per_layer_activation[layer]], dtype=np.float64
        )
        per_layer[str(layer)] = {
            "abs_r_mean": float(np.mean(abs_r)),
            "abs_r_std": float(np.std(abs_r)),
            "slope_mean": float(np.mean(slopes)),
            "slope_std": float(np.std(slopes)),
            "range_mean": float(np.mean(ranges)),
            "range_std": float(np.std(ranges)),
            "delta_norm_mean": float(np.mean(delta_norms)),
            "delta_norm_std": float(np.std(delta_norms)),
            "mean_cosine_mean": float(np.mean(cosines)),
            "mean_cosine_std": float(np.std(cosines)),
        }

    return {
        "seed": seed,
        "layers": LAYERS,
        "contrastive_pairs": [
            {"alpha_high": float(high), "alpha_low": float(low)}
            for high, low in CONTRASTIVE_PAIRS
        ],
        "per_layer": per_layer,
    }


def _plot(seed_payloads: list[dict[str, Any]], out_dir: Path) -> None:
    layers = np.asarray(LAYERS, dtype=np.int64)

    abs_r_seed = np.asarray(
        [
            [payload["per_layer"][str(layer)]["abs_r_mean"] for layer in LAYERS]
            for payload in seed_payloads
        ],
        dtype=np.float64,
    )
    var_seed = np.asarray(
        [
            [payload["per_layer"][str(layer)]["abs_r_std"] for layer in LAYERS]
            for payload in seed_payloads
        ],
        dtype=np.float64,
    )
    delta_seed = np.asarray(
        [
            [payload["per_layer"][str(layer)]["delta_norm_mean"] for layer in LAYERS]
            for payload in seed_payloads
        ],
        dtype=np.float64,
    )

    def _make(curve_arr: np.ndarray, title: str, ylabel: str, filename: str) -> None:
        mean = curve_arr.mean(axis=0)
        std = curve_arr.std(axis=0)
        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.plot(layers, mean, marker="o", linewidth=2)
        ax.fill_between(layers, mean - std, mean + std, alpha=0.2)
        ax.axvline(5, linestyle="--", color="gray", alpha=0.7)
        ax.axvline(6, linestyle="--", color="red", alpha=0.7)
        ax.set_xticks(layers)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    _make(abs_r_seed, "TabICL L4-L7: mean |r|", "mean |r|", "mean_abs_r_zoom.png")
    _make(
        var_seed,
        "TabICL L4-L7: instability (std of |r|)",
        "std(|r|)",
        "instability_zoom.png",
    )
    _make(
        delta_seed,
        "TabICL L4-L7: activation delta norm",
        "||mean_high-mean_low||",
        "delta_norm_zoom.png",
    )


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("RD6: TabICL L5-L6 zoom")
    print(f"QUICK_RUN={cfg.QUICK_RUN}, n_train={cfg.N_TRAIN}, n_test={cfg.N_TEST}")

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        print(f"[seed={seed}] running...")
        payload = _run_seed(seed)
        per_seed.append(payload)

    aggregated: dict[str, dict[str, float]] = {}
    for layer in LAYERS:
        abs_r_vals = [
            payload["per_layer"][str(layer)]["abs_r_mean"] for payload in per_seed
        ]
        abs_r_var_vals = [
            payload["per_layer"][str(layer)]["abs_r_std"] for payload in per_seed
        ]
        delta_vals = [
            payload["per_layer"][str(layer)]["delta_norm_mean"] for payload in per_seed
        ]
        cosine_vals = [
            payload["per_layer"][str(layer)]["mean_cosine_mean"] for payload in per_seed
        ]
        aggregated[str(layer)] = {
            "abs_r_mean_over_seeds": float(np.mean(abs_r_vals)),
            "abs_r_std_over_seeds": float(np.std(abs_r_vals)),
            "instability_mean_over_seeds": float(np.mean(abs_r_var_vals)),
            "instability_std_over_seeds": float(np.std(abs_r_var_vals)),
            "delta_norm_mean_over_seeds": float(np.mean(delta_vals)),
            "delta_norm_std_over_seeds": float(np.std(delta_vals)),
            "mean_cosine_mean_over_seeds": float(np.mean(cosine_vals)),
            "mean_cosine_std_over_seeds": float(np.std(cosine_vals)),
        }

    out = {
        "quick_run": bool(cfg.QUICK_RUN),
        "seeds": SEEDS,
        "layers": LAYERS,
        "contrastive_pairs": [
            {"alpha_high": float(high), "alpha_low": float(low)}
            for high, low in CONTRASTIVE_PAIRS
        ],
        "per_seed": per_seed,
        "aggregated": aggregated,
    }

    with (RESULTS_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    _plot(per_seed, RESULTS_DIR)
    print(f"Saved: {RESULTS_DIR / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
