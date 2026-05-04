# pyright: reportMissingImports=false
"""RD-8 M3-T6: Q-K Joint Embedding Space Analysis.

Extracts Query and Key vectors from TabPFN's sample attention
(self_attn_between_items) using forward pre-hooks on the attention modules.
Projects Q and K jointly to 2D via PCA and saves scatter plots showing how
queries and keys cluster in the embedding space at selected layers.
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
import torch
from sklearn.decomposition import PCA
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rd5_config import cfg

from src.data.synthetic_generator import generate_quadratic_data  # noqa: E402

QUICK_RUN = True
RANDOM_SEED = 42
LAYERS_TO_ANALYZE = [0, 5, 8, 11]


def _build_model() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _make_qk_hook(
    qk_cache: dict[int, dict[str, np.ndarray]],
    layer_idx: int,
) -> Any:
    """Create a forward pre-hook that computes Q and K from attention inputs.

    The hook intercepts the input to self_attn_between_items, applies the
    in_proj_weight to compute Q and K projections, and stores them in the cache.

    Args:
        qk_cache: Shared dict to store Q/K arrays keyed by layer index.
        layer_idx: Transformer layer index for cache key.
    """

    def hook(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        x = args[0]  # [batch*(fb+1), N_train, emsize]
        W = module.in_proj_weight.detach()  # [3*emsize, emsize]
        b = module.in_proj_bias.detach()  # [3*emsize]
        emsize = W.shape[0] // 3

        W_Q = W[:emsize, :]
        W_K = W[emsize : 2 * emsize, :]
        b_Q = b[:emsize]
        b_K = b[emsize : 2 * emsize]

        # Project: [batch*(fb+1), N_train, emsize]
        Q = (x @ W_Q.T + b_Q).detach().cpu().numpy()
        K = (x @ W_K.T + b_K).detach().cpu().numpy()

        # Take first batch element
        Q_first = Q[0]  # [N_train, emsize]
        K_first = K[0]  # [N_train, emsize]

        qk_cache[layer_idx] = {"Q": Q_first, "K": K_first}

    return hook


def _plot_qk_scatter(
    Q_2d: np.ndarray,
    K_2d: np.ndarray,
    layer_idx: int,
    explained_var: np.ndarray,
    save_path: Path,
) -> None:
    """Plot Q and K points in a joint 2D PCA space.

    Args:
        Q_2d: PCA-projected Q vectors, shape [N_train, 2].
        K_2d: PCA-projected K vectors, shape [N_train, 2].
        layer_idx: Layer index for title.
        explained_var: PCA explained variance ratio [2].
        save_path: Output path for the figure.
    """
    n_train = Q_2d.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        Q_2d[:, 0],
        Q_2d[:, 1],
        c="tab:blue",
        marker="o",
        s=60,
        alpha=0.7,
        label="Q (query)",
        edgecolors="white",
        linewidths=0.5,
    )
    ax.scatter(
        K_2d[:, 0],
        K_2d[:, 1],
        c="tab:orange",
        marker="^",
        s=60,
        alpha=0.7,
        label="K (key)",
        edgecolors="white",
        linewidths=0.5,
    )

    # Label each point with its sample index
    for i in range(n_train):
        ax.annotate(str(i), (Q_2d[i, 0], Q_2d[i, 1]), fontsize=6, alpha=0.6)
        ax.annotate(str(i), (K_2d[i, 0], K_2d[i, 1]), fontsize=6, alpha=0.6)

    ax.set_xlabel(f"PC1 ({explained_var[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({explained_var[1]:.1%} var)")
    ax.set_title(
        f"Q-K Joint Embedding — Layer {layer_idx}",
        fontweight="bold",
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    results_dir = ROOT / "results" / "rd8" / "qk_embeddings"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RD-8 M3-T6: Q-K Joint Embedding Space Analysis")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}")

    # Generate data and fit model
    dataset = generate_quadratic_data(
        n_train=50,
        n_test=10,
        random_seed=RANDOM_SEED,
    )
    model = _build_model()
    model.fit(dataset.X_train, dataset.y_train)

    pytorch_model = model.model_

    # Prepare Q-K cache and register forward pre-hooks
    qk_cache: dict[int, dict[str, np.ndarray]] = {}
    handles: list[torch.utils.hooks.RemovableHook] = []

    try:
        for layer_idx in LAYERS_TO_ANALYZE:
            attn_module = pytorch_model.transformer_encoder.layers[
                layer_idx
            ].self_attn_between_items
            handle = attn_module.register_forward_pre_hook(
                _make_qk_hook(qk_cache, layer_idx),
            )
            handles.append(handle)

        # Trigger forward pass to populate cache
        _ = model.predict(dataset.X_test)
    finally:
        for h in handles:
            h.remove()

    # PCA projection and plotting
    pca_results: dict[str, Any] = {}

    for layer_idx in LAYERS_TO_ANALYZE:
        entry = qk_cache[layer_idx]
        Q = entry["Q"]  # [N_train, emsize]
        K = entry["K"]  # [N_train, emsize]

        # Joint embedding: stack Q and K
        QK = np.vstack([Q, K])  # [2*N_train, emsize]

        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        QK_2d = pca.fit_transform(QK)  # [2*N_train, 2]

        n_train = Q.shape[0]
        Q_2d = QK_2d[:n_train]  # [N_train, 2]
        K_2d = QK_2d[n_train:]  # [N_train, 2]

        explained_var = pca.explained_variance_ratio_

        print(
            f"\nLayer {layer_idx}: PCA explained variance = "
            f"[{explained_var[0]:.4f}, {explained_var[1]:.4f}]"
        )

        pca_results[str(layer_idx)] = {
            "explained_variance_ratio": explained_var.tolist(),
        }

        _plot_qk_scatter(
            Q_2d,
            K_2d,
            layer_idx=layer_idx,
            explained_var=explained_var,
            save_path=results_dir / f"layer_{layer_idx}_qk_pca.png",
        )

    # Save results JSON
    payload = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "layers_analyzed": LAYERS_TO_ANALYZE,
        "n_train": int(dataset.X_train.shape[0]),
        "n_features": int(dataset.X_train.shape[1]),
        "pca_explained_variance_ratio": pca_results,
    }

    with (results_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved outputs to: {results_dir}")


if __name__ == "__main__":
    main()
