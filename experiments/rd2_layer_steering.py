# pyright: reportMissingImports=false
"""RD-2 M9-T3: Layer-by-Layer Steering Effect Comparison.

Applies the same steering direction (extracted from α-contrastive pair)
at each of the 12 transformer layers to identify where steering is most
effective. Confirms that Layer 5-8 (the core computation zone identified
in Phases 1-2) shows the strongest steering effect.

Methodology:
    1. Extract α-direction from contrastive pair at each target layer.
    2. For each layer: sweep λ and compute Pearson r(λ, prediction_shift).
    3. Compare steering effectiveness across layers.

Reference:
    - Turner et al. "Activation Addition" (2023)
    - Gupta et al. "TabPFN Through The Looking Glass" arXiv:2601.08181
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rd5_config import cfg
from src.hooks.steering_vector import (  # noqa: E402
    TabPFNSteeringVector,
    compute_steering_effect,
)

QUICK_RUN = True
RANDOM_SEED = 42
N_LAYERS = 12

# Contrastive pair
ALPHA_HIGH = 5.0
ALPHA_LOW = 2.0
BETA_SHARED = 3.0

# Baseline
ALPHA_BASELINE = 2.0
BETA_BASELINE = 3.0

# Dataset sizes
N_TRAIN = 50
N_TEST = 10

# Layers to compare
TARGET_LAYERS = [0, 2, 4, 6, 8, 10]  # representative layers across 12
LAMBDA_VALUES = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def _build_model() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _generate_shared_data() -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    n_total = N_TRAIN + N_TEST
    X = rng.standard_normal((n_total, 2))

    X_train = X[:N_TRAIN]
    X_test = X[n_total - N_TEST :]

    y_train_high = ALPHA_HIGH * X_train[:, 0] + BETA_SHARED * X_train[:, 1]
    y_train_low = ALPHA_LOW * X_train[:, 0] + BETA_SHARED * X_train[:, 1]
    y_train_baseline = ALPHA_BASELINE * X_train[:, 0] + BETA_BASELINE * X_train[:, 1]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_high": y_train_high,
        "y_train_low": y_train_low,
        "y_train_baseline": y_train_baseline,
    }


def _run_layer_steering(
    steerer: TabPFNSteeringVector,
    model: TabPFNRegressor,
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train_high = data["y_train_high"]
    y_train_low = data["y_train_low"]
    y_train_baseline = data["y_train_baseline"]

    layer_results: list[dict[str, Any]] = []

    for idx, layer in enumerate(TARGET_LAYERS):
        step = f"[{idx + 1}/{len(TARGET_LAYERS)}]"
        print(f"\n{step} Layer {layer}: extract direction and sweep steering")

        direction_layer = steerer.extract_direction(
            X_train_high=X_train,
            y_train_high=y_train_high,
            X_train_low=X_train,
            y_train_low=y_train_low,
            X_test=X_test,
            layer=layer,
            X_val=X_train,  # Use train features (no test-set leakage).
        )

        # Critical: refit baseline after extraction and before steering sweep.
        model.fit(X_train, y_train_baseline)

        sweep_results = steerer.sweep_lambda(
            X_test=X_test,
            layer=layer,
            direction=direction_layer,
            lambdas=LAMBDA_VALUES,
        )
        effect = compute_steering_effect(
            sweep_results["lambdas"],
            sweep_results["mean_preds"],
        )

        layer_results.append(
            {
                "layer": layer,
                "pearson_r": effect["pearson_r"],
                "pearson_p": effect["pearson_p"],
                "slope": effect["slope"],
                "prediction_range": effect["prediction_range"],
                "lambdas": sweep_results["lambdas"],
                "mean_preds": {
                    str(k): float(v) for k, v in sweep_results["mean_preds"].items()
                },
            }
        )

        print(
            "  "
            f"r={effect['pearson_r']:+.4f}, "
            f"slope={effect['slope']:+.4f}, "
            f"range={effect['prediction_range']:.4f}"
        )

    return layer_results


def _plot_layer_steering_comparison(
    layer_results: list[dict[str, Any]],
    save_path: Path,
) -> None:
    layers = np.asarray([res["layer"] for res in layer_results], dtype=np.int32)
    pearson_r = np.asarray(
        [res["pearson_r"] for res in layer_results], dtype=np.float64
    )
    colors = plt.cm.viridis(np.linspace(0, 1, len(TARGET_LAYERS)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.axvspan(4.5, 8.5, color="#FFFACD", alpha=0.6)
    ax.bar(layers, pearson_r, color=colors, width=1.25, edgecolor="black")
    ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    peak_idx = int(np.argmax(np.abs(pearson_r)))
    peak_layer = int(layers[peak_idx])
    peak_val = float(pearson_r[peak_idx])
    ax.annotate(
        f"Peak: L{peak_layer} ({peak_val:+.3f})",
        xy=(peak_layer, peak_val),
        xytext=(peak_layer + 0.8, peak_val + 0.04),
        fontsize=9,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "black"},
        color="black",
    )
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Pearson r(λ, mean prediction)", fontsize=11)
    ax.set_title("(a) Steering Effectiveness by Layer", fontsize=12, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_ylim(min(-1.05, float(np.min(pearson_r) - 0.05)), 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    for color, res in zip(colors, layer_results):
        lambda_values = [float(v) for v in res["lambdas"]]
        mean_preds_map = res["mean_preds"]
        mean_preds = [mean_preds_map[str(v)] for v in lambda_values]
        ax.plot(
            lambda_values,
            mean_preds,
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"L{res['layer']} (r={res['pearson_r']:+.2f})",
        )

    ax.axvline(x=0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Steering strength λ", fontsize=11)
    ax.set_ylabel("Mean prediction", fontsize=11)
    ax.set_title("(b) Mean Prediction vs λ (per layer)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "RD-2 Layer Steering Comparison: Direction Extracted and Applied at Same Layer",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _analyze_layer_groups(layer_results: list[dict[str, Any]]) -> dict[str, Any]:
    core_rs = [
        float(res["pearson_r"]) for res in layer_results if 5 <= int(res["layer"]) <= 8
    ]
    early_rs = [
        float(res["pearson_r"]) for res in layer_results if 0 <= int(res["layer"]) <= 4
    ]

    core_mean_r = float(np.mean(core_rs)) if core_rs else float("nan")
    early_mean_r = float(np.mean(early_rs)) if early_rs else float("nan")
    mean_gap = core_mean_r - early_mean_r

    print("\nLayer-wise summary")
    print("-" * 72)
    print(f"{'Layer':>6} | {'Pearson r':>10} | {'Slope':>10} | {'Pred range':>10}")
    print("-" * 72)
    for res in layer_results:
        print(
            f"{res['layer']:>6} | "
            f"{res['pearson_r']:>+10.4f} | "
            f"{res['slope']:>+10.4f} | "
            f"{res['prediction_range']:>10.4f}"
        )
    print("-" * 72)
    print(f"Mean Pearson r (Layer 5-8 subset): {core_mean_r:+.4f}")
    print(f"Mean Pearson r (Layer 0-4 subset): {early_mean_r:+.4f}")
    print(f"Gap (Layer 5-8 minus Layer 0-4):  {mean_gap:+.4f}")

    return {
        "layer_5_8_mean_pearson_r": core_mean_r,
        "layer_0_4_mean_pearson_r": early_mean_r,
        "mean_gap": mean_gap,
        "layer_5_8_layers": [
            int(res["layer"]) for res in layer_results if 5 <= res["layer"] <= 8
        ],
        "layer_0_4_layers": [
            int(res["layer"]) for res in layer_results if 0 <= res["layer"] <= 4
        ],
    }


def main() -> None:
    results_dir = ROOT / "results" / "rd2" / "layer_steering"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RD-2 M9-T3: Layer-by-Layer Steering Effect Comparison")
    print("=" * 72)
    print(
        f"Contrastive pair: (alpha_high={ALPHA_HIGH}, alpha_low={ALPHA_LOW}, beta_shared={BETA_SHARED})"
    )
    print(
        f"Baseline: alpha={ALPHA_BASELINE}, beta={BETA_BASELINE}; TARGET_LAYERS={TARGET_LAYERS}"
    )

    print("\n[1/5] Generating contrastive and baseline datasets ...")
    data = _generate_shared_data()
    print(f"  X_train: {data['X_train'].shape}, X_test: {data['X_test'].shape}")

    model = _build_model()
    steerer = TabPFNSteeringVector(model)

    print("\n[2/5] Running layer-wise steering sweep ...")
    layer_results = _run_layer_steering(steerer, model, data)

    print("\n[3/5] Plotting comparison figure ...")
    plot_path = results_dir / "layer_steering_comparison.png"
    _plot_layer_steering_comparison(layer_results, plot_path)
    print(f"  Saved: {plot_path}")

    print("\n[4/5] Analyzing layer groups ...")
    group_analysis = _analyze_layer_groups(layer_results)

    print("\n[5/5] Saving results JSON ...")
    payload: dict[str, Any] = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "n_layers": N_LAYERS,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "contrastive": {
            "alpha_high": ALPHA_HIGH,
            "alpha_low": ALPHA_LOW,
            "beta_shared": BETA_SHARED,
        },
        "baseline": {
            "alpha": ALPHA_BASELINE,
            "beta": BETA_BASELINE,
        },
        "target_layers": TARGET_LAYERS,
        "lambda_values": LAMBDA_VALUES,
        "per_layer": {
            f"layer_{res['layer']}": {
                "pearson_r": res["pearson_r"],
                "pearson_p": res["pearson_p"],
                "slope": res["slope"],
                "prediction_range": res["prediction_range"],
                "lambdas": res["lambdas"],
                "mean_predictions": res["mean_preds"],
            }
            for res in layer_results
        },
        "group_analysis": group_analysis,
    }

    json_path = results_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\nDone. All outputs in: {results_dir}")


if __name__ == "__main__":
    main()
