"""TabPFN activation extraction infrastructure for mechanistic interpretability.

Provides hook-based activation caching for TabPFN's 12-layer dual-axis
transformer (PerFeatureTransformer) architecture. Extracts:
- Layer outputs (residual stream)
- Feature attention outputs (self_attn_between_features)
- Sample attention outputs (self_attn_between_items)
- Output head activations
- Logit lens projections

Architecture reference:
    - 12 PerFeatureEncoderLayer layers
    - Each layer: Feature Attention → Sample Attention → MLP (all post-norm)
    - Dual-axis: feature attention over feature blocks, sample attention over items
    - Output head: decoder_dict["standard"] = Linear → GELU → Linear
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TabPFNHookedModel:
    """Hook-based activation extractor for fitted TabPFN models.

    Registers forward hooks on all 12 transformer layers and their
    sub-modules (feature attention, sample attention) to cache activations
    during a forward pass.

    Usage:
        model = TabPFNRegressor(device='cpu')
        model.fit(X_train, y_train)

        hooker = TabPFNHookedModel(model)
        pred, cache = hooker.forward_with_cache(X_test)

        # Layer i activation
        layer_act = cache['layers'][i]  # [batch, seq_len, feature_blocks+1, 192]

        # Feature attention output (includes residual, pre-layernorm)
        feat_attn = cache['feature_attn'][i]  # [batch, seq_len, feature_blocks+1, 192]

        # Sample attention output (last call; in multiquery mode = train self-attn)
        samp_attn = cache['sample_attn'][i]  # [batch, feature_blocks+1, N_train, 192]

    Note:
        In multiquery mode (default for TabPFN v2), self_attn_between_items is
        called twice per layer: once for test→train cross-attention, once for
        train self-attention. The hook captures the LAST call (train self-attn).
        This matches the shape [batch, feature_blocks+1, N_train, emsize].
    """

    NUM_LAYERS = 12

    def __init__(self, fitted_model: Any) -> None:
        """Initialize with a fitted TabPFN model.

        Args:
            fitted_model: A fitted TabPFNRegressor or TabPFNClassifier.
                Must have been .fit() already.

        Raises:
            AttributeError: If model has not been fitted (no model_ attribute).
        """
        if not hasattr(fitted_model, "model_"):
            raise AttributeError(
                "Model must be fitted before hooking. Call model.fit(X, y) first."
            )
        self.model = fitted_model
        self._pytorch_model: nn.Module = fitted_model.model_
        self._n_layers: int = len(self._pytorch_model.transformer_encoder.layers)

        # Cache output head reference for logit lens
        self._output_head: nn.Sequential = self._pytorch_model.decoder_dict["standard"]

    def _get_layer(self, idx: int) -> nn.Module:
        """Get transformer layer by index."""
        return self._pytorch_model.transformer_encoder.layers[idx]

    def forward_with_cache(
        self,
        X_test: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], dict[str, Any]]:
        """Run prediction while caching all layer activations.

        Registers hooks, runs model.predict(X_test), collects activations,
        and removes hooks. Thread-safe hook cleanup via try/finally.

        Args:
            X_test: Test features, shape [N_test, n_features].

        Returns:
            Tuple of (predictions, cache) where:
                predictions: np.ndarray of shape [N_test] or [N_test, ...]
                cache: dict with keys:
                    'layers': list of 12 torch.Tensor, each
                        [batch, seq_len, feature_blocks+1, emsize]
                    'feature_attn': list of 12 torch.Tensor
                    'sample_attn': list of 12 torch.Tensor
                    'output': torch.Tensor [N_test, batch, n_output_bins]
                    'single_eval_pos': int (number of training samples)
        """
        cache: dict[str, Any] = {
            "layers": [None] * self._n_layers,
            "feature_attn": [None] * self._n_layers,
            "sample_attn": [None] * self._n_layers,
            "output": None,
            "single_eval_pos": None,
        }
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def _make_hook(store: list[torch.Tensor | None], idx: int) -> Any:
            """Create a forward hook that stores detached output."""

            def hook(
                module: nn.Module,
                input: Any,
                output: torch.Tensor,
            ) -> None:
                store[idx] = output.detach().cpu()

            return hook

        try:
            # Register hooks on each transformer layer
            for i in range(self._n_layers):
                layer = self._get_layer(i)

                # Layer output hook (full residual stream after all sublayers)
                handles.append(
                    layer.register_forward_hook(_make_hook(cache["layers"], i))
                )

                # Feature attention output hook
                handles.append(
                    layer.self_attn_between_features.register_forward_hook(
                        _make_hook(cache["feature_attn"], i)
                    )
                )

                # Sample attention output hook
                # NOTE: In multiquery mode, this module is called twice per
                # layer. The hook captures the last call (train self-attn),
                # giving shape [batch, feature_blocks+1, N_train, emsize].
                handles.append(
                    layer.self_attn_between_items.register_forward_hook(
                        _make_hook(cache["sample_attn"], i)
                    )
                )

            # Output head hook
            def _output_hook(
                module: nn.Module,
                input: Any,
                output: torch.Tensor,
            ) -> None:
                cache["output"] = output.detach().cpu()

            handles.append(self._output_head.register_forward_hook(_output_hook))

            # Run prediction — triggers the full forward pass
            predictions = self.model.predict(X_test)

        finally:
            # Always remove hooks to prevent memory leaks
            for h in handles:
                h.remove()

        # Compute single_eval_pos from shapes
        N_test = X_test.shape[0]
        seq_len = cache["layers"][0].shape[1]
        cache["single_eval_pos"] = seq_len - N_test

        return predictions, cache

    def get_label_token_activations(
        self, cache: dict[str, Any], layer_idx: int
    ) -> NDArray[np.floating]:
        """Extract label token (last feature block) activations for all samples.

        The label token is the last position in the feature_blocks+1 dimension,
        corresponding to the y-embedding column. This is the primary channel
        through which predictions are formed.

        Args:
            cache: Cache dict from forward_with_cache.
            layer_idx: Transformer layer index (0-11).

        Returns:
            np.ndarray of shape [seq_len, emsize] (squeezed batch dim).
        """
        # cache['layers'][layer_idx]: [batch=1, seq_len, feature_blocks+1, emsize]
        activation = cache["layers"][layer_idx]
        # Last feature block = label token
        label_tok = activation[0, :, -1, :]  # [seq_len, emsize]
        if isinstance(label_tok, torch.Tensor):
            return label_tok.numpy()
        return np.asarray(label_tok)

    def get_test_label_token(
        self, cache: dict[str, Any], layer_idx: int
    ) -> NDArray[np.floating]:
        """Extract test samples' label token activations only.

        These are the representations from which predictions are decoded.
        Useful for probing experiments that focus on test-time representations.

        Args:
            cache: Cache dict from forward_with_cache.
            layer_idx: Transformer layer index (0-11).

        Returns:
            np.ndarray of shape [N_test, emsize].
        """
        single_eval_pos = cache["single_eval_pos"]
        activation = cache["layers"][layer_idx]
        # [batch=1, single_eval_pos:, -1, :] → [N_test, emsize]
        test_tok = activation[0, single_eval_pos:, -1, :]
        if isinstance(test_tok, torch.Tensor):
            return test_tok.numpy()
        return np.asarray(test_tok)

    def apply_logit_lens(
        self, cache: dict[str, Any], layer_idx: int
    ) -> NDArray[np.floating]:
        """Apply logit lens: run intermediate activations through the output head.

        Takes the test samples' label token activations from a given layer
        and passes them through the full output head (Linear → GELU → Linear)
        to see what the model would predict from that layer's representation.

        This implements the logit lens technique from Nostalgebraist (2020),
        adapted for TabPFN's architecture.

        Args:
            cache: Cache dict from forward_with_cache.
            layer_idx: Transformer layer index (0-11).

        Returns:
            np.ndarray of shape [N_test, n_output_bins].
        """
        single_eval_pos = cache["single_eval_pos"]
        activation = cache["layers"][layer_idx]

        # Get test label token: [batch=1, N_test, emsize]
        test_tok = activation[:, single_eval_pos:, -1, :]

        # Ensure tensor for forward pass
        if isinstance(test_tok, np.ndarray):
            test_tok = torch.from_numpy(test_tok)

        # Transpose to decoder input format: [N_test, batch, emsize]
        test_tok = test_tok.transpose(0, 1).contiguous()

        # Move to model device for computation
        device = next(self._output_head.parameters()).device
        test_tok = test_tok.to(device)

        # Apply output head (Linear → GELU → Linear)
        with torch.no_grad():
            logits = self._output_head(test_tok)

        # [N_test, batch=1, n_output_bins] → [N_test, n_output_bins]
        return logits[:, 0, :].detach().cpu().numpy()


class AttentionExtractor:
    """Extract actual attention weight patterns from TabPFN's dual-axis attention.

    While TabPFNHookedModel captures post-attention activations (outputs),
    this class computes the actual attention weight matrices
    (softmax(QK^T / sqrt(d_k))) for visualization and analysis.

    TabPFN uses 6-head attention with d_k=d_v=32, emsize=192.
    Feature attention: self-attention over feature blocks (+ label token).
    Sample attention: cross-attention from all samples to training samples.

    Usage:
        hooker = TabPFNHookedModel(model)
        extractor = AttentionExtractor(hooker)

        attn_data = extractor.extract(X_test)

        # Feature attention weights for layer 5
        feat_weights = attn_data['feature_attn_weights'][5]
        # shape: [batch*seq_len, n_heads, feature_blocks+1, feature_blocks+1]

        # Sample attention weights for layer 5
        samp_weights = attn_data['sample_attn_weights'][5]
        # shape: [batch*(feature_blocks+1), n_heads, N_train, N_train]
        # (last call in multiquery mode = train self-attention)
    """

    def __init__(self, hooked_model: TabPFNHookedModel) -> None:
        """Initialize with a TabPFNHookedModel.

        Args:
            hooked_model: An initialized TabPFNHookedModel instance.
        """
        self._hooked_model = hooked_model
        self._pytorch_model = hooked_model._pytorch_model
        self._n_layers = hooked_model._n_layers

    def extract(
        self,
        X_test: NDArray[np.floating],
    ) -> dict[str, Any]:
        """Run forward pass and extract attention weight patterns.

        Hooks into attention module inputs via forward pre-hooks to capture
        pre-attention activations, then manually computes QKV projections
        and attention weight patterns.

        Args:
            X_test: Test features, shape [N_test, n_features].

        Returns:
            dict with keys:
                'predictions': np.ndarray of predictions
                'cache': full activation cache from TabPFNHookedModel
                'feature_attn_weights': list of 12 tensors
                    Each: [flat_batch, n_heads, seq_q, seq_k]
                'sample_attn_weights': list of 12 tensors
                    Each: [flat_batch, n_heads, seq_q, seq_k]
        """
        # Storage for attention inputs (x, x_kv)
        feat_inputs: list[tuple[torch.Tensor, torch.Tensor | None] | None] = [
            None
        ] * self._n_layers
        samp_inputs: list[tuple[torch.Tensor, torch.Tensor | None] | None] = [
            None
        ] * self._n_layers

        handles: list[torch.utils.hooks.RemovableHandle] = []

        def _make_input_hook(
            store: list[tuple[torch.Tensor, torch.Tensor | None] | None],
            idx: int,
        ) -> Any:
            """Capture attention module inputs (x and optional x_kv)."""

            def hook(
                module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
            ) -> None:
                x = args[0].detach().cpu()
                x_kv = (
                    args[1].detach().cpu()
                    if len(args) > 1 and args[1] is not None
                    else None
                )
                store[idx] = (x, x_kv)

            return hook

        try:
            # Register forward pre-hooks to capture attention inputs
            for i in range(self._n_layers):
                layer = self._hooked_model._get_layer(i)

                handles.append(
                    layer.self_attn_between_features.register_forward_pre_hook(
                        _make_input_hook(feat_inputs, i),
                        with_kwargs=True,
                    )
                )
                handles.append(
                    layer.self_attn_between_items.register_forward_pre_hook(
                        _make_input_hook(samp_inputs, i),
                        with_kwargs=True,
                    )
                )

            # Run forward with full activation cache
            predictions, cache = self._hooked_model.forward_with_cache(X_test)

        finally:
            for h in handles:
                h.remove()

        # Compute attention weight patterns
        feat_weights: list[torch.Tensor] = []
        samp_weights: list[torch.Tensor] = []

        with torch.no_grad():
            for i in range(self._n_layers):
                layer = self._hooked_model._get_layer(i)

                # Feature attention weights
                feat_entry = feat_inputs[i]
                assert feat_entry is not None, f"No feature attn input for layer {i}"
                feat_x, feat_xkv = feat_entry
                feat_w = self._compute_attention_weights(
                    layer.self_attn_between_features,
                    feat_x,
                    feat_xkv,
                )
                feat_weights.append(feat_w)

                # Sample attention weights (last call in multiquery mode)
                samp_entry = samp_inputs[i]
                assert samp_entry is not None, f"No sample attn input for layer {i}"
                samp_x, samp_xkv = samp_entry
                samp_w = self._compute_attention_weights(
                    layer.self_attn_between_items,
                    samp_x,
                    samp_xkv,
                )
                samp_weights.append(samp_w)

        return {
            "predictions": predictions,
            "cache": cache,
            "feature_attn_weights": feat_weights,
            "sample_attn_weights": samp_weights,
        }

    @staticmethod
    def _compute_attention_weights(
        attn_module: nn.Module,
        x: torch.Tensor,
        x_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Manually compute attention weight pattern from input and weights.

        Applies QKV projections and computes softmax(QK^T / sqrt(d_k)).

        Args:
            attn_module: The MultiHeadAttention module.
            x: Query input tensor.
            x_kv: Key/Value input tensor, or None for self-attention.

        Returns:
            Attention weights of shape [flat_batch, n_heads, seq_len_q, seq_len_k].
        """
        # Flatten batch dims like the attention module does internally
        x_flat = x.reshape(-1, *x.shape[-2:])
        if x_kv is not None:
            xkv_flat = x_kv.reshape(-1, *x_kv.shape[-2:])
        else:
            xkv_flat = x_flat

        device = attn_module._w_out.device

        # Compute Q from x, K from x_kv (or x if self-attention)
        if attn_module._w_qkv is not None:
            w_qkv = attn_module._w_qkv  # [3, n_heads, d_k, emsize]
            j, nhead, d_k, input_size = w_qkv.shape
            w_flat = w_qkv.reshape(-1, input_size)

            # Q from query input
            q_flat = torch.matmul(x_flat.to(device), w_flat.T)
            q_all = q_flat.reshape(*x_flat.shape[:-1], j, nhead, d_k)
            q = q_all[..., 0, :, :]  # [flat_batch, seq_q, nhead, d_k]

            # K from kv input
            kv_flat = torch.matmul(xkv_flat.to(device), w_flat.T)
            kv_all = kv_flat.reshape(*xkv_flat.shape[:-1], j, nhead, d_k)
            k = kv_all[..., 1, :, :]  # [flat_batch, seq_k, nhead, d_k]
        else:
            w_q = attn_module._w_q[0]  # [n_heads, d_k, emsize]
            q = torch.einsum("... s, h d s -> ... h d", x_flat.to(device), w_q)

            if attn_module._w_kv is not None:
                w_kv = attn_module._w_kv  # [2, n_heads_kv, d_k, emsize]
                kv = torch.einsum(
                    "... s, j h d s -> ... j h d",
                    xkv_flat.to(device),
                    w_kv,
                )
                k = kv[..., 0, :, :]
            else:
                w_k = attn_module._w_k
                k = torch.einsum(
                    "... s, h d s -> ... h d",
                    xkv_flat.to(device),
                    w_k,
                )

        # Handle grouped query attention (broadcast kv heads)
        d_k = q.shape[-1]
        n_heads_q = q.shape[-2]
        n_heads_k = k.shape[-2]
        if n_heads_k < n_heads_q:
            share_factor = n_heads_q // n_heads_k
            k = (
                k.unsqueeze(-2)
                .expand(*k.shape[:-1], share_factor, d_k)
                .reshape(*k.shape[:-2], n_heads_q, d_k)
            )

        # Compute attention weights: softmax(QK^T / sqrt(d_k))
        # q: [flat_batch, seq_q, n_heads, d_k]
        # k: [flat_batch, seq_k, n_heads, d_k]
        attn_logits = torch.einsum("b q h d, b k h d -> b h q k", q, k) / math.sqrt(d_k)

        attn_weights = torch.softmax(attn_logits, dim=-1)

        return attn_weights.detach().cpu()


# ---------------------------------------------------------------------------
# Verification script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from tabpfn import TabPFNRegressor

    print("=" * 60)
    print("TabPFNHookedModel Verification")
    print("=" * 60)

    # Setup dummy data
    np.random.seed(42)
    X_train = np.random.randn(20, 3)
    y_train = 2 * X_train[:, 0] + 3 * X_train[:, 1] + np.random.randn(20) * 0.1
    X_test = np.random.randn(5, 3)

    # Fit model
    print("\n[1] Fitting TabPFNRegressor...")
    model = TabPFNRegressor(device="cpu", model_path="tabpfn-v2-regressor.ckpt")
    model.fit(X_train, y_train)
    print("    Done.")

    # Create hooked model and run forward
    print("\n[2] Running forward_with_cache...")
    hooker = TabPFNHookedModel(model)
    pred, cache = hooker.forward_with_cache(X_test)
    print("    Done.")

    # Verify shapes
    print("\n[3] Shape verification:")
    print(f"    Predictions shape: {pred.shape}")
    print(f"    Number of cached layers: {len(cache['layers'])}")
    print(f"    single_eval_pos: {cache['single_eval_pos']}")

    for i in [0, 5, 11]:
        print(f"\n    Layer {i}:")
        print(f"      layer output shape:  {cache['layers'][i].shape}")
        print(f"      feature_attn shape:  {cache['feature_attn'][i].shape}")
        print(f"      sample_attn shape:   {cache['sample_attn'][i].shape}")

    print(f"\n    Output head shape: {cache['output'].shape}")

    # Verify helper methods
    print("\n[4] Helper method verification:")
    label_tok = hooker.get_label_token_activations(cache, layer_idx=5)
    print(f"    Label token (layer 5) shape: {label_tok.shape}")

    test_tok = hooker.get_test_label_token(cache, layer_idx=5)
    print(f"    Test label token (layer 5) shape: {test_tok.shape}")

    logit_lens = hooker.apply_logit_lens(cache, layer_idx=5)
    print(f"    Logit lens (layer 5) shape: {logit_lens.shape}")

    # Verify AttentionExtractor
    print("\n[5] AttentionExtractor verification:")
    extractor = AttentionExtractor(hooker)
    attn_data = extractor.extract(X_test)
    for i in [0, 5, 11]:
        fw = attn_data["feature_attn_weights"][i]
        sw = attn_data["sample_attn_weights"][i]
        print(f"    Layer {i}:")
        print(f"      feature_attn_weights shape: {fw.shape}")
        print(f"      sample_attn_weights shape:  {sw.shape}")
        # Verify weights sum to 1 along key dim
        feat_sum = fw[0, 0, 0, :].sum().item()
        samp_sum = sw[0, 0, 0, :].sum().item()
        print(f"      feature weight row sum: {feat_sum:.6f}")
        print(f"      sample weight row sum:  {samp_sum:.6f}")

    # Basic consistency checks
    print("\n[6] Consistency checks:")
    assert pred.shape == (5,), f"Expected (5,), got {pred.shape}"
    assert len(cache["layers"]) == 12, "Expected 12 layers"
    assert cache["single_eval_pos"] == 20, (
        f"Expected 20, got {cache['single_eval_pos']}"
    )
    assert label_tok.shape[0] == 25, f"Expected seq_len=25, got {label_tok.shape[0]}"
    assert label_tok.shape[1] == 192, f"Expected emsize=192, got {label_tok.shape[1]}"
    assert test_tok.shape == (5, 192), f"Expected (5, 192), got {test_tok.shape}"
    assert logit_lens.shape[0] == 5, f"Expected N_test=5, got {logit_lens.shape[0]}"
    assert logit_lens.shape[1] == 5000, f"Expected 5000 bins, got {logit_lens.shape[1]}"

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
