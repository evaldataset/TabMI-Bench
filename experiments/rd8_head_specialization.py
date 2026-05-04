# pyright: reportMissingImports=false
"""RD-8 M3-T4: Head Specialization Analysis.

Computes pairwise Jensen-Shannon divergence between attention heads per layer
to measure head specialization/diversity. Saves diversity curves and per-head
attention heatmaps for the most interesting layer.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
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

from src.data.synthetic_generator import generate_quadratic_data  # noqa: E402
from src.hooks.attention_extractor import TabPFNAttentionExtractor  # noqa: E402
from src.visualization.plots import plot_attention_heatmap  # noqa: E402

QUICK_RUN = True
RANDOM_SEED = 42
N_LAYERS = 12
N_HEADS = 6
EPS = 1e-10


def _build_model() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability distributions."""
    p = p + EPS
    q = q + EPS
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def compute_pairwise_jsd(attn_layer: np.ndarray) -> float:
    """Compute mean pairwise JSD between all heads in a layer.

    Args:
        attn_layer: [n_heads, N, N] attention weights for one layer.

    Returns:
        Mean pairwise JSD across all C(n_heads, 2) pairs.
    """
    weights = np.asarray(attn_layer, dtype=np.float64)
    n_heads = weights.shape[0]
    if n_heads < 2:
        return 0.0

    # Flatten each head's attention to a probability vector
    head_dists = weights.reshape(n_heads, -1)
    # Normalize to valid probability distributions
    head_dists = head_dists + EPS
    head_dists = head_dists / head_dists.sum(axis=1, keepdims=True)

    pairwise_values: list[float] = []
    for i, j in combinations(range(n_heads), 2):
        pairwise_values.append(jsd(head_dists[i], head_dists[j]))

    return float(np.mean(pairwise_values))


def _plot_diversity_curve(
    sample_jsd: np.ndarray,
    feature_jsd: np.ndarray,
    save_path: Path,
) -> None:
    """Plot head diversity (JSD) across layers for both attention types."""
    layers = np.arange(len(sample_jsd), dtype=np.int32)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        layers,
        sample_jsd,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="Sample attention JSD",
    )
    ax.plot(
        layers,
        feature_jsd,
        marker="s",
        linewidth=2,
        color="#ff7f0e",
        label="Feature attention JSD",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean pairwise JSD")
    ax.set_title("Head Specialization: Pairwise JSD Across Layers", fontweight="bold")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _plot_per_head_heatmaps(
    attn_layer: np.ndarray,
    layer_idx: int,
    attn_type: str,
    save_path: Path,
) -> None:
    """Plot per-head attention heatmaps as 2x3 subplots in one figure.

    Args:
        attn_layer: [n_heads, N, N] attention weights.
        layer_idx: Layer index for title.
        attn_type: 'sample' or 'feature'.
        save_path: Path to save the combined figure.
    """
    n_heads = attn_layer.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for h in range(min(n_heads, 6)):
        ax = axes_flat[h]
        head_attn = attn_layer[h]  # [N, N]
        im = ax.imshow(head_attn, cmap="Blues", vmin=0.0, aspect="auto")
        ax.set_title(f"Head {h}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{attn_type.capitalize()} Attention — Layer {layer_idx} (Per-Head)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results_dir = ROOT / "results" / "rd8" / "head_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RD-8 M3-T4: Head Specialization Analysis")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}")

    # Generate data and fit model
    dataset = generate_quadratic_data(n_train=50, n_test=10, random_seed=RANDOM_SEED)
    model = _build_model()
    model.fit(dataset.X_train, dataset.y_train)

    # Extract attention
    extractor = TabPFNAttentionExtractor(model)
    attention_result = extractor.extract(dataset.X_test)

    sample_attn: list[Any] = attention_result["sample_attn"]
    feature_attn: list[Any] = attention_result["feature_attn"]
    n_layers = len(sample_attn)

    # Compute pairwise JSD per layer
    sample_jsd_per_layer = np.array(
        [compute_pairwise_jsd(sample_attn[layer]) for layer in range(n_layers)]
    )
    feature_jsd_per_layer = np.array(
        [compute_pairwise_jsd(feature_attn[layer]) for layer in range(n_layers)]
    )

    print(f"\nSample JSD per layer:  {np.round(sample_jsd_per_layer, 4).tolist()}")
    print(f"Feature JSD per layer: {np.round(feature_jsd_per_layer, 4).tolist()}")

    peak_sample = int(np.argmax(sample_jsd_per_layer))
    peak_feature = int(np.argmax(feature_jsd_per_layer))
    print(
        f"\nPeak sample JSD:  layer {peak_sample} ({sample_jsd_per_layer[peak_sample]:.4f})"
    )
    print(
        f"Peak feature JSD: layer {peak_feature} ({feature_jsd_per_layer[peak_feature]:.4f})"
    )

    # Plot diversity curve
    _plot_diversity_curve(
        sample_jsd_per_layer,
        feature_jsd_per_layer,
        results_dir / "head_diversity_curve.png",
    )

    # Per-head heatmaps for layer 5 (most interesting per paper findings)
    target_layer = 5
    _plot_per_head_heatmaps(
        np.asarray(sample_attn[target_layer], dtype=np.float64),
        layer_idx=target_layer,
        attn_type="sample",
        save_path=results_dir / f"layer_{target_layer}_per_head_sample.png",
    )
    _plot_per_head_heatmaps(
        np.asarray(feature_attn[target_layer], dtype=np.float64),
        layer_idx=target_layer,
        attn_type="feature",
        save_path=results_dir / f"layer_{target_layer}_per_head_feature.png",
    )

    # Save results JSON
    payload = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "n_layers": n_layers,
        "n_heads": N_HEADS,
        "n_train": int(dataset.X_train.shape[0]),
        "n_test": int(dataset.X_test.shape[0]),
        "sample_jsd_per_layer": sample_jsd_per_layer.tolist(),
        "feature_jsd_per_layer": feature_jsd_per_layer.tolist(),
        "peak_sample_jsd_layer": peak_sample,
        "peak_feature_jsd_layer": peak_feature,
        "peak_sample_jsd_value": float(sample_jsd_per_layer[peak_sample]),
        "peak_feature_jsd_value": float(feature_jsd_per_layer[peak_feature]),
        "target_heatmap_layer": target_layer,
    }

    with (results_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved outputs to: {results_dir}")


if __name__ == "__main__":
    main()
