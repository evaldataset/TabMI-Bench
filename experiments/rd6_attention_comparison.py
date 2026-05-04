# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportUnusedParameter=false
from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rd5_config import cfg
from src.data.synthetic_generator import generate_linear_data
from src.hooks.attention_extractor import TabPFNAttentionExtractor
from src.hooks.tabicl_hooker import TabICLHookedModel

RESULTS_DIR = ROOT / "results" / "rd6" / "attention_comparison"
EPS = 1e-12


def _mean_entropy_per_layer(attn_layers: list[np.ndarray]) -> np.ndarray:
    layer_values: list[float] = []
    for layer_attn in attn_layers:
        weights = np.asarray(layer_attn, dtype=np.float64)
        entropy = -np.sum(weights * np.log(weights + EPS), axis=-1)
        layer_values.append(float(np.mean(entropy)))
    return np.asarray(layer_values, dtype=np.float64)


def _pairwise_jsd_per_layer(attn_layers: list[np.ndarray]) -> np.ndarray:
    def _jsd(p: np.ndarray, q: np.ndarray) -> float:
        p = p + EPS
        q = q + EPS
        p = p / np.sum(p)
        q = q / np.sum(q)
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log((p + EPS) / (m + EPS)))
        kl_qm = np.sum(q * np.log((q + EPS) / (m + EPS)))
        return float(0.5 * kl_pm + 0.5 * kl_qm)

    values: list[float] = []
    for layer_attn in attn_layers:
        weights = np.asarray(layer_attn, dtype=np.float64)
        n_heads = int(weights.shape[0])
        if n_heads < 2:
            values.append(0.0)
            continue

        head_distributions = weights.reshape(n_heads, -1)
        head_distributions = head_distributions / (
            np.sum(head_distributions, axis=1, keepdims=True) + EPS
        )

        pairwise_jsd: list[float] = []
        for i, j in combinations(range(n_heads), 2):
            pairwise_jsd.append(_jsd(head_distributions[i], head_distributions[j]))
        values.append(float(np.mean(pairwise_jsd)))

    return np.asarray(values, dtype=np.float64)


def _extract_tabicl_attention(
    model: TabICLRegressor, X_test: np.ndarray
) -> list[np.ndarray]:
    import tabicl.model.layers as tabicl_layers

    blocks = model.model_.icl_predictor.tf_icl.blocks
    n_layers = len(blocks)

    captured: list[torch.Tensor | None] = [None] * n_layers
    handles: list[torch.utils.hooks.RemovableHandle] = []
    current_layer: dict[str, int | None] = {"idx": None}

    original_multi_head_attention_forward = tabicl_layers.multi_head_attention_forward

    def _make_pre_hook(layer_idx: int) -> Any:
        def _pre_hook(module: torch.nn.Module, inputs: Any) -> None:
            current_layer["idx"] = layer_idx

        return _pre_hook

    def _patched_multi_head_attention_forward(*args: Any, **kwargs: Any) -> Any:
        out = original_multi_head_attention_forward(*args, **kwargs)

        layer_idx = current_layer["idx"]
        if layer_idx is None:
            return out

        query = args[0]
        num_heads = int(args[1])
        in_proj_weight = args[2]
        in_proj_bias = args[3]
        key = kwargs.get("key")
        value = kwargs.get("value")
        cached_kv = kwargs.get("cached_kv")
        key_padding_mask = kwargs.get("key_padding_mask")
        attn_mask = kwargs.get("attn_mask")
        rope = kwargs.get("rope")

        *batch_shape, tgt_len, embed_dim = query.shape
        head_dim = embed_dim // num_heads

        if cached_kv is None:
            if key is None or value is None:
                return out
            src_len = key.shape[-2]
            q, k, _ = F._in_projection_packed(
                query, key, value, in_proj_weight, in_proj_bias
            )
            q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)
            k = k.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)
            if rope is not None:
                q = rope.rotate_queries_or_keys(q)
                k = rope.rotate_queries_or_keys(k)
        else:
            k = cached_kv.key
            src_len = k.shape[-2]
            q_proj_weight = in_proj_weight[:embed_dim]
            q_proj_bias = in_proj_bias[:embed_dim] if in_proj_bias is not None else None
            q = F.linear(query, q_proj_weight, q_proj_bias)
            q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)
            if rope is not None:
                q = rope.rotate_queries_or_keys(q)

        merged_mask = attn_mask
        if key_padding_mask is not None:
            key_pad = key_padding_mask.view(*batch_shape, 1, 1, src_len).expand(
                *batch_shape, num_heads, tgt_len, src_len
            )
            merged_mask = key_pad if merged_mask is None else merged_mask + key_pad

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(head_dim))
        if merged_mask is not None:
            scores = scores + merged_mask

        captured[layer_idx] = torch.softmax(scores, dim=-1).detach().cpu()
        return out

    try:
        tabicl_layers.multi_head_attention_forward = (
            _patched_multi_head_attention_forward
        )
        for i, block in enumerate(blocks):
            handles.append(block.attn.register_forward_pre_hook(_make_pre_hook(i)))

        hooker = TabICLHookedModel(model)
        _pred, _cache = hooker.forward_with_cache(X_test)
    finally:
        tabicl_layers.multi_head_attention_forward = (
            original_multi_head_attention_forward
        )
        for handle in handles:
            handle.remove()

    attn_layers: list[np.ndarray] = []
    for layer_idx, attn_weights in enumerate(captured):
        if attn_weights is None:
            raise RuntimeError(
                f"Failed to capture TabICL attention at layer {layer_idx}"
            )

        arr = np.asarray(attn_weights, dtype=np.float64)
        while arr.ndim > 3:
            arr = arr.mean(axis=0)
        if arr.ndim != 3:
            raise ValueError(
                f"Unexpected TabICL attention shape at layer {layer_idx}: {arr.shape}"
            )
        attn_layers.append(arr)

    return attn_layers


def _plot_comparison(
    entropy_tabpfn_sample: np.ndarray,
    entropy_tabpfn_feature: np.ndarray,
    entropy_tabicl: np.ndarray,
    jsd_tabpfn_sample: np.ndarray,
    jsd_tabpfn_feature: np.ndarray,
    jsd_tabicl: np.ndarray,
    save_path: Path,
) -> None:
    layers = np.arange(entropy_tabpfn_sample.shape[0], dtype=np.int32)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_entropy = axes[0]
    ax_entropy.plot(
        layers, entropy_tabpfn_sample, marker="o", linewidth=2, label="TabPFN-sample"
    )
    ax_entropy.plot(
        layers, entropy_tabpfn_feature, marker="s", linewidth=2, label="TabPFN-feature"
    )
    ax_entropy.plot(layers, entropy_tabicl, marker="^", linewidth=2, label="TabICL")
    ax_entropy.set_xlabel("Layer")
    ax_entropy.set_ylabel("Attention entropy")
    ax_entropy.set_title("(a) Entropy by layer")
    ax_entropy.set_xticks(layers)
    ax_entropy.grid(True, alpha=0.3)
    ax_entropy.legend(loc="best")

    ax_jsd = axes[1]
    ax_jsd.plot(
        layers, jsd_tabpfn_sample, marker="o", linewidth=2, label="TabPFN-sample"
    )
    ax_jsd.plot(
        layers, jsd_tabpfn_feature, marker="s", linewidth=2, label="TabPFN-feature"
    )
    ax_jsd.plot(layers, jsd_tabicl, marker="^", linewidth=2, label="TabICL")
    ax_jsd.set_xlabel("Layer")
    ax_jsd.set_ylabel("Head specialization (JSD)")
    ax_jsd.set_title("(b) JSD by layer")
    ax_jsd.set_xticks(layers)
    ax_jsd.grid(True, alpha=0.3)
    ax_jsd.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = generate_linear_data(
        alpha=3.0,
        beta=2.0,
        n_train=cfg.N_TRAIN,
        n_test=cfg.N_TEST,
        random_seed=cfg.SEED,
    )

    tabpfn_model = TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")
    tabpfn_model.fit(dataset.X_train, dataset.y_train)
    tabpfn_extractor = TabPFNAttentionExtractor(tabpfn_model)
    tabpfn_attention = tabpfn_extractor.extract(dataset.X_test)
    tabpfn_sample = [
        np.asarray(x, dtype=np.float64) for x in tabpfn_attention["sample_attn"]
    ]
    tabpfn_feature = [
        np.asarray(x, dtype=np.float64) for x in tabpfn_attention["feature_attn"]
    ]

    tabicl_model = TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)
    tabicl_model.fit(dataset.X_train, dataset.y_train)
    tabicl_attn = _extract_tabicl_attention(tabicl_model, dataset.X_test)

    entropy_tabpfn_sample = _mean_entropy_per_layer(tabpfn_sample)
    entropy_tabpfn_feature = _mean_entropy_per_layer(tabpfn_feature)
    entropy_tabicl = _mean_entropy_per_layer(tabicl_attn)

    jsd_tabpfn_sample = _pairwise_jsd_per_layer(tabpfn_sample)
    jsd_tabpfn_feature = _pairwise_jsd_per_layer(tabpfn_feature)
    jsd_tabicl = _pairwise_jsd_per_layer(tabicl_attn)

    plot_path = RESULTS_DIR / "attention_comparison.png"
    _plot_comparison(
        entropy_tabpfn_sample=entropy_tabpfn_sample,
        entropy_tabpfn_feature=entropy_tabpfn_feature,
        entropy_tabicl=entropy_tabicl,
        jsd_tabpfn_sample=jsd_tabpfn_sample,
        jsd_tabpfn_feature=jsd_tabpfn_feature,
        jsd_tabicl=jsd_tabicl,
        save_path=plot_path,
    )

    payload = {
        "quick_run": bool(cfg.QUICK_RUN),
        "seed": int(cfg.SEED),
        "n_train": int(cfg.N_TRAIN),
        "n_test": int(cfg.N_TEST),
        "data": {"formula": "z = alpha*x + beta*y", "alpha": 3.0, "beta": 2.0},
        "models": {
            "tabpfn": {
                "n_layers": int(tabpfn_attention["n_layers"]),
                "n_heads": int(tabpfn_attention["n_heads"]),
                "sample_entropy": entropy_tabpfn_sample.tolist(),
                "feature_entropy": entropy_tabpfn_feature.tolist(),
                "sample_head_jsd": jsd_tabpfn_sample.tolist(),
                "feature_head_jsd": jsd_tabpfn_feature.tolist(),
            },
            "tabicl": {
                "n_layers": int(len(tabicl_attn)),
                "n_heads": int(tabicl_attn[0].shape[0]) if tabicl_attn else 0,
                "entropy": entropy_tabicl.tolist(),
                "head_jsd": jsd_tabicl.tolist(),
            },
        },
        "artifacts": {
            "results_json": str(RESULTS_DIR / "results.json"),
            "comparison_plot": str(plot_path),
        },
    }

    json_path = RESULTS_DIR / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
