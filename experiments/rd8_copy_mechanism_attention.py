# pyright: reportMissingImports=false
"""RD-8 M3-T3: Copy Mechanism Attention Analysis.

Analyzes whether test samples attend to similar training samples,
testing the hypothesis that TabPFN uses a copy mechanism.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rd5_config import cfg

from src.data.synthetic_generator import generate_quadratic_data  # noqa: E402
from src.hooks.tabpfn_hooker import TabPFNHookedModel  # noqa: E402
from src.visualization.plots import plot_layer_r2  # noqa: E402

QUICK_RUN = True
RANDOM_SEED = 42
N_LAYERS = 12


def _build_model() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_similarity_attention_correlation(
    X_train: np.ndarray,
    X_test: np.ndarray,
    cache: dict[str, Any],
    layer_idx: int,
) -> dict[str, float]:
    """Compute correlation between sample similarity and attention-proxy.

    Since test→train attention is not directly accessible (multiquery mode
    captures train self-attention), we use a proxy: cosine similarity between
    test and train label token activations as an attention proxy.

    Args:
        X_train: [N_train, n_features]
        X_test: [N_test, n_features]
        cache: Activation cache from forward_with_cache
        layer_idx: Layer index

    Returns:
        {'euclidean_corr': float, 'cosine_corr': float}
    """
    single_eval_pos = int(cache["single_eval_pos"])
    layer_act = _to_numpy(cache["layers"][layer_idx][0])  # [seq_len, fb+1, 192]

    # Train label tokens: [N_train, 192]
    train_label_tok = layer_act[:single_eval_pos, -1, :]
    # Test label tokens: [N_test, 192]
    test_label_tok = layer_act[single_eval_pos:, -1, :]

    # Proxy attention: cosine similarity between test and train label tokens
    # [N_test, N_train]
    proxy_attn = cosine_similarity(test_label_tok, train_label_tok)
    # Normalize rows to sum to 1 (like softmax)
    proxy_attn = proxy_attn - proxy_attn.min(axis=1, keepdims=True)
    row_sums = proxy_attn.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    proxy_attn = proxy_attn / row_sums  # [N_test, N_train]

    # Input similarity: Euclidean distance (inverted) and cosine similarity
    # [N_test, N_train]
    euc_dist = euclidean_distances(X_test, X_train)
    euc_sim = 1.0 / (1.0 + euc_dist)  # Convert distance to similarity

    cos_sim = cosine_similarity(X_test, X_train)  # [N_test, N_train]

    # Flatten and compute correlation
    proxy_flat = proxy_attn.flatten()
    euc_flat = euc_sim.flatten()
    cos_flat = cos_sim.flatten()

    euc_corr = float(np.corrcoef(proxy_flat, euc_flat)[0, 1])
    cos_corr = float(np.corrcoef(proxy_flat, cos_flat)[0, 1])

    return {"euclidean_corr": euc_corr, "cosine_corr": cos_corr}


def main() -> None:
    results_dir = ROOT / "results" / "rd8" / "copy_mechanism_attention"
    results_dir.mkdir(parents=True, exist_ok=True)

    n_datasets = 5 if QUICK_RUN else 20
    n_train = 50
    n_test = 10

    print("=" * 72)
    print("RD-8 M3-T3: Copy Mechanism Attention Analysis")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}, n_datasets={n_datasets}")

    euc_corr_per_layer: list[list[float]] = [[] for _ in range(N_LAYERS)]
    cos_corr_per_layer: list[list[float]] = [[] for _ in range(N_LAYERS)]

    start = time.time()
    for ds_idx in range(n_datasets):
        ds = generate_quadratic_data(
            n_train=n_train, n_test=n_test, random_seed=RANDOM_SEED + ds_idx
        )
        model = _build_model()
        model.fit(ds.X_train, ds.y_train)

        hooker = TabPFNHookedModel(model)
        _, cache = hooker.forward_with_cache(ds.X_test)

        for layer_idx in range(N_LAYERS):
            corrs = compute_similarity_attention_correlation(
                ds.X_train, ds.X_test, cache, layer_idx
            )
            euc_corr_per_layer[layer_idx].append(corrs["euclidean_corr"])
            cos_corr_per_layer[layer_idx].append(corrs["cosine_corr"])

        elapsed = time.time() - start
        print(f"  Dataset {ds_idx + 1}/{n_datasets} done (total={elapsed / 60:.1f}m)")

    # Average correlations per layer
    euc_mean = np.array([np.mean(v) for v in euc_corr_per_layer])
    cos_mean = np.array([np.mean(v) for v in cos_corr_per_layer])

    # Plot correlation curves
    r2_matrix = np.column_stack([euc_mean, cos_mean])
    fig = plot_layer_r2(
        r2_matrix,
        title="Copy Mechanism: Similarity-Attention Correlation per Layer",
        save_path=str(results_dir / "correlation_curve.png"),
        complexity_labels=["Euclidean similarity", "Cosine similarity"],
    )
    plt.close(fig)

    # Success criterion: r > 0.5 in layer 5-8
    success = bool(np.any(np.abs(cos_mean[5:9]) > 0.5))

    payload = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "n_datasets": n_datasets,
        "euclidean_corr_per_layer": euc_mean.tolist(),
        "cosine_corr_per_layer": cos_mean.tolist(),
        "success_criterion_r_gt_0.5_layer5to8": success,
    }
    with (results_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        f"\nMax cosine corr: {float(np.max(np.abs(cos_mean))):.4f} at layer {int(np.argmax(np.abs(cos_mean)))}"
    )
    print(f"Success criterion (|r| > 0.5 in layer 5-8): {success}")
    print(f"Saved: {results_dir}")


if __name__ == "__main__":
    main()
