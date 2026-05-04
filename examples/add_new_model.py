"""Example: How to add a new TFM to TabMI-Bench.

This file demonstrates the minimal interface required to integrate
a new tabular foundation model into the benchmark. Implementing
the HookedModel protocol (~100-200 lines) enables all existing
evaluation scripts to run on the new model without modification.

Usage:
    python examples/add_new_model.py
"""
from __future__ import annotations

import numpy as np


class HookedModelProtocol:
    """The interface that all TabMI-Bench hook implementations must follow.

    Required methods:
        forward_with_cache(X_test) -> (predictions, cache)
        get_layer_activations(cache, layer_idx) -> np.ndarray
        num_layers -> int
        hidden_dim -> int

    Optional methods (model-specific):
        get_test_label_token(cache, layer_idx) -> np.ndarray  # TabPFN-style
        apply_logit_lens(cache, layer_idx) -> np.ndarray      # if applicable
    """

    def __init__(self, fitted_model):
        """Initialize with a fitted (trained) model instance."""
        self.model = fitted_model
        # Extract PyTorch module and layer references
        # self._layers = model.some_attribute.layers
        raise NotImplementedError

    def forward_with_cache(self, X_test: np.ndarray) -> tuple:
        """Run forward pass, caching activations at each layer.

        Returns:
            (predictions: np.ndarray, cache: dict[int, np.ndarray])
        """
        # 1. Register forward hooks on each layer
        # 2. Run model.predict(X_test)
        # 3. Remove hooks (use try/finally)
        # 4. Return (predictions, activation_cache)
        raise NotImplementedError

    def get_layer_activations(self, cache: dict, layer_idx: int) -> np.ndarray:
        """Extract activations for a specific layer from the cache.

        Returns:
            np.ndarray of shape [n_test, hidden_dim]
        """
        return cache[layer_idx]

    @property
    def num_layers(self) -> int:
        """Number of hookable layers."""
        raise NotImplementedError

    @property
    def hidden_dim(self) -> int:
        """Hidden dimension of layer activations."""
        raise NotImplementedError


# --- Example: Minimal hook for a hypothetical "MyTFM" model ---

class MyTFMHookedModel:
    """Example hook implementation for a hypothetical model.

    This demonstrates the pattern used by all existing hooks:
    - tabpfn_hooker.py (594 lines, handles dual-axis attention)
    - tabicl_hooker.py (267 lines, handles column→row blocks)
    - iltm_hooker.py (192 lines, handles tree+MLP)
    - tabdpt_hooker.py (91 lines, simplest example)
    """

    def __init__(self, fitted_model):
        self.model = fitted_model
        # Access the PyTorch module inside the sklearn-like wrapper
        self._pytorch_model = fitted_model.model_
        self._layers = self._pytorch_model.encoder.layers  # example
        self._n_layers = len(self._layers)
        self._hidden_dim = self._layers[0].linear1.in_features  # example

    def forward_with_cache(self, X_test: np.ndarray) -> tuple:
        import torch

        cache: dict[int, np.ndarray] = {}
        handles = []

        try:
            for i, layer in enumerate(self._layers):
                def make_hook(idx):
                    def hook(module, input, output):
                        # Store activation as numpy array
                        if isinstance(output, torch.Tensor):
                            cache[idx] = output.detach().cpu().numpy()
                    return hook
                handles.append(layer.register_forward_hook(make_hook(i)))

            predictions = self.model.predict(X_test)
        finally:
            for h in handles:
                h.remove()

        return predictions, cache

    def get_layer_activations(self, cache: dict, layer_idx: int) -> np.ndarray:
        act = cache[layer_idx]
        # Reshape to [n_samples, hidden_dim] if needed
        if act.ndim == 3:
            act = act[:, -1, :]  # take last token (common for ICL models)
        return act

    @property
    def num_layers(self) -> int:
        return self._n_layers

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim


if __name__ == "__main__":
    print("TabMI-Bench: Adding a New Model")
    print("=" * 50)
    print()
    print("Steps to add your TFM:")
    print("1. Create src/hooks/your_model_hooker.py")
    print("2. Implement HookedModelProtocol (see above)")
    print("3. Run: python experiments/rd5_intermediary_probing.py")
    print("4. Compare against Table 1 reference baselines")
    print()
    print("Typical effort: ~100-200 lines, ~2-4 hours")
    print("See tabdpt_hooker.py (91 lines) as the simplest example.")
