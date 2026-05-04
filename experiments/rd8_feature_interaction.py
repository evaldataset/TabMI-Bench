# pyright: reportMissingImports=false
"""RD-8 M3-T5: Feature Attention Pattern → Feature Interaction Structure Analysis.

Analyzes whether feature attention weights reflect the multiplicative structure
(a·b) in z = a·b + c.  With 3 features the feature-attention blocks are:

    block 0 = [a, b]   (multiplicative pair)
    block 1 = [c]       (additive term)
    block 2 = label token

Hypothesis: self-attention within the multiplicative pair (attn[0,0]) is higher
than cross-block attention to the additive term (attn[0,1]) in the computation
layers 5-8 — indicating that the model learns to focus on interacting features.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tabpfn import TabPFNRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rd5_config import cfg

from src.data.synthetic_generator import generate_quadratic_data  # noqa: E402
from src.hooks.attention_extractor import TabPFNAttentionExtractor  # noqa: E402

QUICK_RUN = True
RANDOM_SEED = 42
N_LAYERS = 12
N_HEADS = 6
HYPOTHESIS_LAYERS = [5, 6, 7, 8]
HEATMAP_LAYERS = [0, 5, 8, 11]


def _build_model() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _compute_block_attention(
    feature_attn: list[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute head-averaged inter-block attention values per layer.

    For each layer, averages over heads (axis=0) to get [fb+1, fb+1] matrix,
    then extracts attn[0,0], attn[0,1], attn[1,1].

    Returns:
        (attn_00, attn_01, attn_11) — each [n_layers] array.
    """
    n_layers = len(feature_attn)
    attn_00 = np.zeros(n_layers, dtype=np.float64)
    attn_01 = np.zeros(n_layers, dtype=np.float64)
    attn_11 = np.zeros(n_layers, dtype=np.float64)

    for layer in range(n_layers):
        # feature_attn[layer] shape: [n_heads, fb+1, fb+1]
        avg = np.asarray(feature_attn[layer], dtype=np.float64).mean(axis=0)
        attn_00[layer] = float(avg[0, 0])
        attn_01[layer] = float(avg[0, 1])
        attn_11[layer] = float(avg[1, 1])

    return attn_00, attn_01, attn_11


def _test_hypothesis(
    attn_00: np.ndarray,
    attn_01: np.ndarray,
    layers: list[int],
) -> bool:
    """Test whether attn[0,0] > attn[0,1] in the majority of given layers."""
    count_supported = sum(1 for layer in layers if attn_00[layer] > attn_01[layer])
    return count_supported > len(layers) / 2


def _plot_inter_block_curve(
    attn_00: np.ndarray,
    attn_01: np.ndarray,
    attn_11: np.ndarray,
    save_path: Path,
) -> None:
    """Line chart of inter-block attention values across layers."""
    layers = np.arange(len(attn_00), dtype=np.int32)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        layers,
        attn_00,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="attn[0,0]: self [a,b]↔[a,b]",
    )
    ax.plot(
        layers,
        attn_01,
        marker="s",
        linewidth=2,
        color="#ff7f0e",
        label="attn[0,1]: cross [a,b]→[c]",
    )
    ax.plot(
        layers,
        attn_11,
        marker="^",
        linewidth=2,
        color="#2ca02c",
        label="attn[1,1]: self [c]↔[c]",
    )

    # Shade hypothesis region (layers 5-8)
    ax.axvspan(
        HYPOTHESIS_LAYERS[0] - 0.3,
        HYPOTHESIS_LAYERS[-1] + 0.3,
        alpha=0.10,
        color="blue",
        label="Hypothesis region (L5-8)",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Head-averaged attention weight")
    ax.set_title(
        "Feature Interaction: Inter-Block Attention Across Layers",
        fontweight="bold",
    )
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_feature_attention_heatmaps(
    feature_attn: list[Any],
    layers: list[int],
    save_path: Path,
) -> None:
    """Plot head-averaged feature attention heatmaps for selected layers."""
    n_plots = len(layers)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 3.5))

    if n_plots == 1:
        axes = [axes]

    block_labels = ["[a,b]", "[c]", "label"]

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        avg = np.asarray(feature_attn[layer], dtype=np.float64).mean(axis=0)

        im = ax.imshow(avg, cmap="Blues", vmin=0.0, aspect="equal")
        ax.set_title(f"Layer {layer}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Key block")
        ax.set_ylabel("Query block")

        n_blocks = avg.shape[0]
        ticks = list(range(n_blocks))
        labels = block_labels[:n_blocks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=9)

        # Annotate cell values
        for i in range(n_blocks):
            for j in range(n_blocks):
                ax.text(
                    j,
                    i,
                    f"{avg[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if avg[i, j] > avg.max() * 0.6 else "black",
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Feature Attention Heatmaps (Head-Averaged)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    results_dir = ROOT / "results" / "rd8" / "feature_interaction"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RD-8 M3-T5: Feature Interaction Structure Analysis")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}")
    print("Hypothesis: attn[0,0] (self [a,b]) > attn[0,1] (cross [a,b]→[c])")
    print(f"Hypothesis layers: {HYPOTHESIS_LAYERS}")

    # Generate data and fit model
    dataset = generate_quadratic_data(n_train=50, n_test=10, random_seed=RANDOM_SEED)
    model = _build_model()
    model.fit(dataset.X_train, dataset.y_train)

    # Extract attention
    extractor = TabPFNAttentionExtractor(model)
    attention_result = extractor.extract(dataset.X_test)

    feature_attn: list[Any] = attention_result["feature_attn"]
    n_layers = len(feature_attn)
    fb_plus_1 = np.asarray(feature_attn[0], dtype=np.float64).shape[1]
    print(f"\nExtracted {n_layers} layers, feature block size fb+1={fb_plus_1}")

    # Compute inter-block attention values
    attn_00, attn_01, attn_11 = _compute_block_attention(feature_attn)

    print("\nPer-layer head-averaged attention:")
    print(f"  attn[0,0] (self [a,b]):    {np.round(attn_00, 4).tolist()}")
    print(f"  attn[0,1] (cross [a,b]→c): {np.round(attn_01, 4).tolist()}")
    print(f"  attn[1,1] (self [c]):      {np.round(attn_11, 4).tolist()}")

    # Test hypothesis
    valid_hyp_layers = [l for l in HYPOTHESIS_LAYERS if l < n_layers]
    hypothesis_supported = _test_hypothesis(attn_00, attn_01, valid_hyp_layers)
    print(f"\nHypothesis (attn[0,0] > attn[0,1] in L5-8): {hypothesis_supported}")
    for layer in valid_hyp_layers:
        marker = "✓" if attn_00[layer] > attn_01[layer] else "✗"
        print(
            f"  Layer {layer}: attn[0,0]={attn_00[layer]:.4f} vs "
            f"attn[0,1]={attn_01[layer]:.4f}  {marker}"
        )

    # Plot 1: Inter-block attention curve
    _plot_inter_block_curve(
        attn_00, attn_01, attn_11, results_dir / "inter_block_attention.png"
    )

    # Plot 2: Feature attention heatmaps for selected layers
    valid_heatmap_layers = [l for l in HEATMAP_LAYERS if l < n_layers]
    _plot_feature_attention_heatmaps(
        feature_attn, valid_heatmap_layers, results_dir / "feature_attn_heatmaps.png"
    )

    # Save results JSON
    payload = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "n_layers": n_layers,
        "n_heads": N_HEADS,
        "n_train": int(dataset.X_train.shape[0]),
        "n_test": int(dataset.X_test.shape[0]),
        "fb_plus_1": fb_plus_1,
        "block_mapping": {
            "0": "[a,b] (multiplicative pair)",
            "1": "[c] (additive term)",
            "2": "label token",
        },
        "attn_block00_per_layer": attn_00.tolist(),
        "attn_block01_per_layer": attn_01.tolist(),
        "attn_block11_per_layer": attn_11.tolist(),
        "hypothesis_layers": valid_hyp_layers,
        "hypothesis_supported": hypothesis_supported,
    }

    with (results_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved outputs to: {results_dir}")


if __name__ == "__main__":
    main()
