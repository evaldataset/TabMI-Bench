# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAny=false, reportExplicitAny=false
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rd5_config import cfg

from src.data.synthetic_generator import generate_quadratic_data  # noqa: E402
from src.hooks.attention_extractor import (  # noqa: E402
    TabPFNAttentionExtractor,
    compute_layer_entropy_curve,
)
from src.visualization.plots import plot_attention_heatmap  # noqa: E402


QUICK_RUN = True
RANDOM_SEED = 42
N_LAYERS = 12


def _build_model() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _plot_entropy_curve(
    sample_entropy: np.ndarray,
    feature_entropy: np.ndarray,
    save_path: Path,
) -> None:
    layers = np.arange(len(sample_entropy), dtype=np.int32)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        layers,
        sample_entropy,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="Sample attention entropy",
    )
    ax.plot(
        layers,
        feature_entropy,
        marker="s",
        linewidth=2,
        color="#ff7f0e",
        label="Feature attention entropy",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean entropy")
    ax.set_title("Attention Entropy Across Layers", fontweight="bold")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _layers_to_save(n_layers: int) -> list[int]:
    if QUICK_RUN:
        return [layer for layer in [0, 5, 8, 11] if layer < n_layers]
    return list(range(n_layers))


def _save_layer_heatmaps(
    attention_result: dict[str, Any],
    results_dir: Path,
) -> dict[str, Any]:
    sample_attn = attention_result["sample_attn"]
    feature_attn = attention_result["feature_attn"]

    n_layers = len(sample_attn)
    layers = _layers_to_save(n_layers)

    for layer in layers:
        sample_avg = np.asarray(sample_attn[layer], dtype=np.float64).mean(axis=0)
        feature_avg = np.asarray(feature_attn[layer], dtype=np.float64).mean(axis=0)

        sample_path = results_dir / f"layer_{layer}_sample_attn.png"
        feature_path = results_dir / f"layer_{layer}_feature_attn.png"

        sample_fig = plot_attention_heatmap(
            sample_avg,
            layer=layer,
            head=-1,
            attn_type="sample",
            title=f"Sample Attention (Head-avg) - Layer {layer}",
            save_path=str(sample_path),
        )
        plt.close(sample_fig)

        feature_fig = plot_attention_heatmap(
            feature_avg,
            layer=layer,
            head=-1,
            attn_type="feature",
            title=f"Feature Attention (Head-avg) - Layer {layer}",
            save_path=str(feature_path),
        )
        plt.close(feature_fig)

    return {
        "layers_saved": layers,
        "sample_shape_per_layer": [
            list(np.asarray(layer).shape) for layer in sample_attn
        ],
        "feature_shape_per_layer": [
            list(np.asarray(layer).shape) for layer in feature_attn
        ],
    }


def main() -> None:
    results_dir = ROOT / "results" / "rd8" / "attention_heatmaps"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RD-8 Attention Heatmap Visualization")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}")

    dataset = generate_quadratic_data(n_train=50, n_test=10, random_seed=RANDOM_SEED)
    model = _build_model()
    model.fit(dataset.X_train, dataset.y_train)

    extractor = TabPFNAttentionExtractor(model)
    attention_result = extractor.extract(dataset.X_test)

    heatmap_meta = _save_layer_heatmaps(attention_result, results_dir)

    sample_entropy = compute_layer_entropy_curve(
        attention_result, attn_type="sample_attn"
    )
    feature_entropy = compute_layer_entropy_curve(
        attention_result, attn_type="feature_attn"
    )
    _plot_entropy_curve(
        sample_entropy, feature_entropy, results_dir / "entropy_curve.png"
    )

    payload = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "n_layers": int(attention_result["n_layers"]),
        "n_heads": int(attention_result["n_heads"]),
        "n_train": int(dataset.X_train.shape[0]),
        "n_test": int(dataset.X_test.shape[0]),
        "layers_saved": heatmap_meta["layers_saved"],
        "sample_attention_entropy": sample_entropy.tolist(),
        "feature_attention_entropy": feature_entropy.tolist(),
        "sample_shape_per_layer": heatmap_meta["sample_shape_per_layer"],
        "feature_shape_per_layer": heatmap_meta["feature_shape_per_layer"],
    }

    with (results_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved outputs to: {results_dir}")


if __name__ == "__main__":
    main()
