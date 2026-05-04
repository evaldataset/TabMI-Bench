from __future__ import annotations

from typing import Any

import numpy as np  # pyright: ignore[reportMissingImports]

from .tabpfn_hooker import AttentionExtractor, TabPFNHookedModel


EPS = 1e-12


def compute_attention_entropy(attn_weights: np.ndarray) -> np.ndarray:
    """Compute entropy over the last dimension of attention weights.

    Args:
        attn_weights: Attention weights where the last axis is a probability
            distribution (sums to 1).

    Returns:
        Entropy values with the same shape as ``attn_weights`` except for
        the last dimension.
    """
    weights = np.asarray(attn_weights, dtype=np.float64)
    return -np.sum(weights * np.log(weights + EPS), axis=-1)


def compute_layer_entropy_curve(
    attention_dict: dict[str, Any], attn_type: str = "sample_attn"
) -> np.ndarray:
    """Compute mean attention entropy per layer.

    Args:
        attention_dict: Output from ``TabPFNAttentionExtractor.extract``.
        attn_type: ``"sample_attn"`` or ``"feature_attn"``.

    Returns:
        Array of shape ``[n_layers]`` with mean entropy per layer.
    """
    if attn_type not in attention_dict:
        raise KeyError(f"Unknown attention type: {attn_type}")

    layer_values: list[float] = []
    for layer_attn in attention_dict[attn_type]:
        per_head = compute_attention_entropy(layer_attn).mean(axis=-1)
        layer_values.append(float(np.mean(per_head)))
    return np.asarray(layer_values, dtype=np.float64)


class TabPFNAttentionExtractor:
    """High-level wrapper for extracting TabPFN attention matrices."""

    def __init__(self, fitted_model: Any) -> None:
        """Wrap a fitted TabPFN model for attention extraction."""
        self._hooked_model = TabPFNHookedModel(fitted_model)
        self._extractor = AttentionExtractor(self._hooked_model)

    def extract(self, X_test: np.ndarray) -> dict[str, Any]:
        """Extract attention weights for all layers and heads.

        Returns:
            {
                'sample_attn': list of 12 arrays, each [n_heads, N_train, N_train],
                'feature_attn': list of 12 arrays, each [n_heads, fb+1, fb+1],
                'n_layers': 12,
                'n_heads': 6,
            }
        """
        raw = self._extractor.extract(X_test)

        sample_attn = [
            np.asarray(layer_weights.mean(dim=0).cpu().numpy(), dtype=np.float64)
            for layer_weights in raw["sample_attn_weights"]
        ]
        feature_attn = [
            np.asarray(layer_weights.mean(dim=0).cpu().numpy(), dtype=np.float64)
            for layer_weights in raw["feature_attn_weights"]
        ]

        n_layers = len(sample_attn)
        n_heads = int(sample_attn[0].shape[0]) if n_layers > 0 else 0

        return {
            "sample_attn": sample_attn,
            "feature_attn": feature_attn,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "predictions": raw["predictions"],
            "cache": raw["cache"],
        }

    def compute_entropy(self, attn_weights: np.ndarray) -> np.ndarray:
        """Compute mean attention entropy per head.

        Args:
            attn_weights: ``[n_heads, N, N]`` attention weights.

        Returns:
            ``[n_heads]`` entropy values.
        """
        entropy_per_query = compute_attention_entropy(attn_weights)
        return np.mean(entropy_per_query, axis=-1)

    def compute_head_diversity(self, attn_weights: np.ndarray) -> float:
        """Compute average pairwise Jensen-Shannon divergence between heads.

        Args:
            attn_weights: ``[n_heads, N, N]`` attention weights.

        Returns:
            Average pairwise JSD between heads.
        """
        weights = np.asarray(attn_weights, dtype=np.float64)
        if weights.ndim != 3:
            raise ValueError(
                f"Expected 3D attention tensor [n_heads, N, N], got {weights.shape}"
            )

        n_heads = weights.shape[0]
        if n_heads < 2:
            return 0.0

        head_distributions = weights.reshape(n_heads, -1)
        head_distributions /= head_distributions.sum(axis=1, keepdims=True) + EPS

        def _kl(p: np.ndarray, q: np.ndarray) -> float:
            return float(np.sum(p * np.log((p + EPS) / (q + EPS))))

        pairwise_jsd: list[float] = []
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                p = head_distributions[i]
                q = head_distributions[j]
                m = 0.5 * (p + q)
                jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
                pairwise_jsd.append(jsd)

        return float(np.mean(pairwise_jsd))
