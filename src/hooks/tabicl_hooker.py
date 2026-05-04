# pyright: reportMissingImports=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TabICLHookedModel:
    """Hook-based activation extractor for fitted TabICL models.

    Registers forward hooks on all 12 ICL transformer blocks to cache
    residual stream activations during inference.

    Usage:
        model = TabICLRegressor(device="cpu", random_state=42)
        model.fit(X_train, y_train)

        hooker = TabICLHookedModel(model)
        pred, cache = hooker.forward_with_cache(X_test)

        act = hooker.get_layer_activations(cache, i)
    """

    NUM_LAYERS = 12
    HIDDEN_DIM = 512

    def __init__(self, fitted_model: Any) -> None:
        """Initialize with a fitted TabICLRegressor.

        Args:
            fitted_model: A fitted TabICLRegressor instance.

        Raises:
            AttributeError: If model has not been fitted (no model_ attribute).
        """
        if not hasattr(fitted_model, "model_"):
            raise AttributeError(
                "Model must be fitted before hooking. Call model.fit(X, y) first."
            )

        self.model = fitted_model
        self._pytorch_model: nn.Module = fitted_model.model_

        self._blocks: nn.ModuleList = self._pytorch_model.icl_predictor.tf_icl.blocks
        self._n_layers: int = len(self._blocks)
        self._decoder: nn.Sequential = self._pytorch_model.icl_predictor.decoder
        self._ln: nn.LayerNorm = self._pytorch_model.icl_predictor.ln

    def _get_layer(self, idx: int) -> nn.Module:
        """Get ICL block by index."""
        return self._blocks[idx]

    def forward_with_cache(
        self,
        X_test: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], dict[str, Any]]:
        """Run inference while caching ICL block activations.

        Args:
            X_test: Test features, shape [N_test, n_features].

        Returns:
            Tuple of (predictions, cache) where:
                predictions: np.ndarray of shape [N_test]
                cache: dict with keys:
                    'layers': list of 12 torch.Tensor, each [1, seq_len, 512]
                    'output': torch.Tensor decoder output
                    'train_size': int number of in-context training samples
        """
        cache: dict[str, Any] = {
            "layers": [None] * self._n_layers,
            "output": None,
            "train_size": None,
        }
        handles: list[torch.utils.hooks.RemovableHandle] = []

        X_test_encoded = self.model.X_encoder_.transform(X_test)
        data = self.model.ensemble_generator_.transform(X_test_encoded, mode="both")
        Xs, ys = next(iter(data.values()))

        X_tensor = torch.from_numpy(Xs[0:1]).float()
        y_tensor = torch.from_numpy(ys[0:1]).float()

        def _make_hook(store: list[torch.Tensor | None], idx: int) -> Any:
            """Create a forward hook that stores detached output."""

            def hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
                store[idx] = output.detach().cpu()

            return hook

        try:
            for i in range(self._n_layers):
                block = self._get_layer(i)
                handles.append(
                    block.register_forward_hook(_make_hook(cache["layers"], i))
                )

            def _decoder_hook(
                module: nn.Module, input: Any, output: torch.Tensor
            ) -> None:
                cache["output"] = output.detach().cpu()

            handles.append(self._decoder.register_forward_hook(_decoder_hook))

            device = self.model.device_
            self._pytorch_model.eval()
            with torch.no_grad():
                raw_quantiles = self._pytorch_model._inference_forward(
                    X_tensor.to(device),
                    y_tensor.to(device),
                    inference_config=self.model.inference_config_,
                )

        finally:
            for h in handles:
                h.remove()

        cache["train_size"] = int(y_tensor.shape[1])

        pred = raw_quantiles.mean(dim=-1).detach().cpu().numpy()
        pred = self.model.y_scaler_.inverse_transform(pred.reshape(-1, 1)).reshape(-1)

        return pred, cache

    def get_layer_activations(
        self,
        cache: dict[str, Any],
        layer_idx: int,
    ) -> NDArray[np.floating]:
        """Extract test-sample activations from a specific ICL block.

        Args:
            cache: Cache dict from forward_with_cache.
            layer_idx: ICL block index (0-11).

        Returns:
            np.ndarray of shape [N_test, 512].
        """
        train_size = cache["train_size"]
        activation = cache["layers"][layer_idx]
        test_act = activation[:, train_size:, :].squeeze(0)
        if isinstance(test_act, torch.Tensor):
            return test_act.numpy()
        return np.asarray(test_act)

    def get_test_label_token(
        self,
        cache: dict[str, Any],
        layer_idx: int,
    ) -> NDArray[np.floating]:
        """Alias for test activations (TabICL has no separate label token)."""
        return self.get_layer_activations(cache, layer_idx)

    def apply_logit_lens(
        self,
        cache: dict[str, Any],
        layer_idx: int,
    ) -> NDArray[np.floating]:
        """Apply TabICL logit lens using LN + decoder on layer activations.

        Args:
            cache: Cache dict from forward_with_cache.
            layer_idx: ICL block index (0-11).

        Returns:
            np.ndarray of shape [N_test, 999].
        """
        train_size = cache["train_size"]
        act = cache["layers"][layer_idx][:, train_size:, :]

        if isinstance(act, np.ndarray):
            act_tensor = torch.from_numpy(act).float()
        else:
            act_tensor = act

        device = next(self._decoder.parameters()).device
        act_tensor = act_tensor.to(device)

        with torch.no_grad():
            act_tensor = self._ln(act_tensor)
            logits = self._decoder(act_tensor)

        return logits[0].detach().cpu().numpy()

    @property
    def num_layers(self) -> int:
        """Number of hookable ICL blocks."""
        return self.NUM_LAYERS

    @property
    def hidden_dim(self) -> int:
        """Hidden dimension of ICL block activations."""
        return self.HIDDEN_DIM


if __name__ == "__main__":
    from tabicl import TabICLRegressor

    print("=" * 60)
    print("TabICLHookedModel Verification")
    print("=" * 60)

    np.random.seed(42)
    X_train = np.random.randn(20, 3)
    y_train = 2 * X_train[:, 0] + 3 * X_train[:, 1] + np.random.randn(20) * 0.1
    X_test = np.random.randn(5, 3)

    print("\n[1] Fitting TabICLRegressor...")
    model = TabICLRegressor(device="cpu", random_state=42)
    model.fit(X_train, y_train)
    print("    Done.")

    print("\n[2] Running forward_with_cache...")
    hooker = TabICLHookedModel(model)
    pred, cache = hooker.forward_with_cache(X_test)
    print("    Done.")

    print("\n[3] Shape verification:")
    print(f"    Predictions shape: {pred.shape}")
    print(f"    Number of cached layers: {len(cache['layers'])}")
    print(f"    train_size: {cache['train_size']}")

    for i in [0, 5, 11]:
        print(f"\n    Layer {i}:")
        print(f"      layer output shape: {cache['layers'][i].shape}")

    print(f"\n    Decoder output shape: {cache['output'].shape}")

    print("\n[4] Helper method verification:")
    layer_act = hooker.get_layer_activations(cache, layer_idx=5)
    print(f"    Layer activations (layer 5) shape: {layer_act.shape}")

    test_tok = hooker.get_test_label_token(cache, layer_idx=5)
    print(f"    Test label token (layer 5) shape: {test_tok.shape}")

    logit_lens = hooker.apply_logit_lens(cache, layer_idx=5)
    print(f"    Logit lens (layer 5) shape: {logit_lens.shape}")

    print("\n[5] Consistency checks:")
    assert pred.shape == (5,), f"Expected (5,), got {pred.shape}"
    assert len(cache["layers"]) == 12, "Expected 12 cached layers"

    for i in range(12):
        layer = cache["layers"][i]
        assert layer is not None, f"Layer {i} activation is None"
        assert layer.shape[0] == 1, (
            f"Expected batch=1 for layer {i}, got {layer.shape[0]}"
        )
        assert layer.shape[-1] == 512, (
            f"Expected hidden_dim=512 for layer {i}, got {layer.shape[-1]}"
        )

    assert layer_act.shape == (5, 512), f"Expected (5, 512), got {layer_act.shape}"
    assert test_tok.shape == (5, 512), f"Expected (5, 512), got {test_tok.shape}"
    assert logit_lens.shape == (5, 999), f"Expected (5, 999), got {logit_lens.shape}"

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
