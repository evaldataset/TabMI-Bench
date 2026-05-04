# pyright: reportMissingImports=false
"""Activation patching for TabPFN causal intervention experiments.

Implements the activation patching technique (also called causal tracing)
for TabPFN's 12-layer PerFeatureTransformer. This enables causal analysis
of which layers are critical for specific computations by replacing
activations from a corrupted run with cached clean activations.

Key idea:
    1. Run model on clean input → cache all 12 layer activations
    2. Run model on corrupted input with ONE layer's activation
       replaced by the clean version
    3. If replacing layer L restores the clean prediction,
       then layer L is causally important for the computation
       disrupted by the corruption.

Important: TabPFN v2 runs **multiple ensemble passes** per predict() call
(e.g. 8 passes with varying feature-block counts). The caching and patching
must handle ALL passes — not just one — to avoid shape mismatches and
silent no-ops.

Reference:
    - Meng et al. "Locating and Editing Factual Associations in GPT" (2022)
      arXiv:2202.05262 — Activation patching / causal tracing methodology
    - Gupta et al. "TabPFN Through The Looking Glass" (2026)
      arXiv:2601.08181 — TabPFN mechanistic interpretability
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TabPFNActivationPatcher:
    """Activation patching for TabPFN causal intervention experiments.

    Registers forward hooks on TabPFN's 12 PerFeatureEncoderLayer layers
    to cache activations during a clean run, then selectively replaces
    individual layer outputs during a corrupted run to measure causal
    importance.

    TabPFN v2 runs multiple ensemble passes per ``predict()`` call, with
    varying feature-block dimensions. This patcher correctly handles
    all passes by caching a list of tensors per layer and replaying
    them in the same order during patching.

    Usage::

        patcher = TabPFNActivationPatcher(model)

        # Step 1: Cache clean activations (all ensemble passes)
        preds_clean, clean_cache = patcher.run_with_cache(X_test)

        # Step 2: Run corrupted input with layer 6 patched from clean
        preds_patched = patcher.patched_run(X_test_corrupted, clean_cache, patch_layer=6)

        # Step 3: Measure patch effect
        effect = compute_patch_effect(preds_clean, preds_corrupted, preds_patched)
        print(f"Layer 6 patch effect: {effect['mean']:.4f}")

    Note:
        The model must be fitted (``model.fit(X_train, y_train)``) before
        creating a patcher. The same fitted model is used for both clean
        and corrupted runs — only ``X_test`` changes.
    """

    NUM_LAYERS = 12

    def __init__(self, model: Any) -> None:
        """Initialize with a fitted TabPFN model.

        Args:
            model: A fitted TabPFNRegressor (or TabPFNClassifier).
                Must have been ``.fit()`` already so that ``model.model_``
                exists.

        Raises:
            AttributeError: If model has not been fitted (no ``model_`` attribute).
        """
        if not hasattr(model, "model_"):
            raise AttributeError(
                "Model must be fitted before patching. Call model.fit(X, y) first."
            )
        self.model = model
        self._pytorch_model: nn.Module = model.model_
        self._n_layers: int = len(self._pytorch_model.transformer_encoder.layers)

    def _get_layer(self, idx: int) -> nn.Module:
        """Get transformer layer by index.

        Args:
            idx: Layer index (0-11).

        Returns:
            The PerFeatureEncoderLayer module at the given index.
        """
        return self._pytorch_model.transformer_encoder.layers[idx]

    def run_with_cache(
        self,
        X_test: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], dict[str, Any]]:
        """Run model and cache all layer activations across all ensemble passes.

        Registers forward hooks on every transformer layer, runs
        ``model.predict(X_test)``, caches the outputs, then removes
        all hooks. Each layer stores a **list** of tensors (one per
        ensemble pass) to handle TabPFN's multi-pass inference.

        Args:
            X_test: Test features, shape ``[N_test, n_features]``.

        Returns:
            Tuple of ``(predictions, cache)`` where:
                - predictions: ``np.ndarray`` of shape ``[N_test]``
                - cache: dict with keys:

                  - ``'layers'``: list of 12 lists, each containing
                    ``torch.Tensor`` objects (one per ensemble pass)
                  - ``'single_eval_pos'``: ``int`` (number of training samples)
                  - ``'n_passes'``: ``int`` (number of ensemble passes detected)
        """
        # Each layer gets a list to collect ALL ensemble passes
        layer_caches: list[list[torch.Tensor]] = [[] for _ in range(self._n_layers)]
        handles: list[torch.utils.hooks.RemovableHook] = []

        def _make_cache_hook(store: list[torch.Tensor], idx: int) -> Any:
            """Create a forward hook that appends detached output to the list."""

            def hook(
                module: nn.Module,
                input: Any,  # noqa: A002
                output: torch.Tensor,
            ) -> None:
                store.append(output.detach().clone())

            return hook

        try:
            # Register cache hooks on each transformer layer
            for i in range(self._n_layers):
                layer = self._get_layer(i)
                handles.append(
                    layer.register_forward_hook(_make_cache_hook(layer_caches[i], i))
                )

            # Run prediction — triggers the full forward pass (multiple ensemble passes)
            predictions = self.model.predict(X_test)

        finally:
            # Always remove hooks to prevent memory leaks
            for h in handles:
                h.remove()

        # Determine single_eval_pos from the first pass of layer 0
        N_test = X_test.shape[0]
        if layer_caches[0]:
            seq_len = layer_caches[0][0].shape[1]
            single_eval_pos = seq_len - N_test
        else:
            single_eval_pos = int(self.model.executor_.single_eval_pos)

        n_passes = len(layer_caches[0]) if layer_caches[0] else 0

        cache: dict[str, Any] = {
            "layers": layer_caches,
            "single_eval_pos": single_eval_pos,
            "n_passes": n_passes,
        }

        return predictions, cache

    def patched_run(
        self,
        X_test: NDArray[np.floating],
        clean_cache: dict[str, Any],
        patch_layer: int,
    ) -> NDArray[np.floating]:
        """Run model with activation patching at a specific layer.

        Runs ``model.predict(X_test)`` but replaces the output of
        ``patch_layer`` with the corresponding cached clean activation
        for EVERY ensemble pass. Uses a call counter to replay cached
        activations in the same order they were recorded.

        Args:
            X_test: Test features (typically corrupted), shape
                ``[N_test, n_features]``.
            clean_cache: Cache dict from a prior ``run_with_cache()``
                call on clean data. Must contain ``clean_cache['layers']``
                with cached tensor lists.
            patch_layer: Index of the layer to patch (0-11).

        Returns:
            Predictions as ``np.ndarray`` of shape ``[N_test]``, computed
            with the patched activation at ``patch_layer``.

        Raises:
            ValueError: If ``patch_layer`` is out of range.
            KeyError: If ``clean_cache`` does not contain ``'layers'``.
        """
        if not 0 <= patch_layer < self._n_layers:
            raise ValueError(
                f"patch_layer must be 0..{self._n_layers - 1}, got {patch_layer}"
            )
        if "layers" not in clean_cache:
            raise KeyError("clean_cache must contain 'layers' key")

        clean_activations: list[torch.Tensor] = clean_cache["layers"][patch_layer]
        call_counter = [0]  # Mutable counter for closure

        def _patch_hook(
            module: nn.Module,
            input: Any,  # noqa: A002
            output: torch.Tensor,
        ) -> torch.Tensor:
            """Replace the layer output with the cached clean activation."""
            idx = call_counter[0]
            call_counter[0] += 1
            if idx < len(clean_activations):
                return clean_activations[idx].detach().clone().to(output.device)
            # Fallback: if more passes than cached (shouldn't happen), return original
            return output

        handle = self._get_layer(patch_layer).register_forward_hook(_patch_hook)

        try:
            predictions = self.model.predict(X_test)
        finally:
            handle.remove()

        return predictions

    def sweep_all_layers(
        self,
        X_test_corrupted: NDArray[np.floating],
        clean_cache: dict[str, Any],
        preds_clean: NDArray[np.floating],
        preds_corrupted: NDArray[np.floating],
    ) -> dict[str, Any]:
        """Sweep activation patching across all 12 layers.

        Convenience method that patches each layer individually and
        computes the patch effect, producing a full causal importance
        profile across the network.

        Args:
            X_test_corrupted: Corrupted test features, shape
                ``[N_test, n_features]``.
            clean_cache: Cache from ``run_with_cache()`` on clean data.
            preds_clean: Clean predictions from ``run_with_cache()``.
            preds_corrupted: Predictions on corrupted data (no patching).

        Returns:
            Dict with keys:

            - ``'per_layer_effect'``: list of 12 floats (mean patch effect)
            - ``'per_layer_abs_effect'``: list of 12 floats (abs mean)
            - ``'per_layer_preds'``: list of 12 ``np.ndarray`` predictions
            - ``'most_important_layer'``: int (layer with highest abs effect)
        """
        per_layer_effect: list[float] = []
        per_layer_abs_effect: list[float] = []
        per_layer_preds: list[NDArray[np.floating]] = []

        for layer_idx in range(self._n_layers):
            preds_patched = self.patched_run(
                X_test_corrupted, clean_cache, patch_layer=layer_idx
            )
            effect = compute_patch_effect(preds_clean, preds_corrupted, preds_patched)
            per_layer_effect.append(effect["mean"])
            per_layer_abs_effect.append(effect["abs_mean"])
            per_layer_preds.append(preds_patched)

        most_important = int(np.argmax(per_layer_abs_effect))

        return {
            "per_layer_effect": per_layer_effect,
            "per_layer_abs_effect": per_layer_abs_effect,
            "per_layer_preds": per_layer_preds,
            "most_important_layer": most_important,
        }


def compute_patch_effect(
    preds_clean: NDArray[np.floating],
    preds_corrupted: NDArray[np.floating],
    preds_patched: NDArray[np.floating],
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Compute the activation patch effect metric.

    Measures how much patching a single layer recovers the clean output
    from the corrupted output:

    .. math::

        \\text{effect}_i = \\frac{\\hat{y}^{\\text{patched}}_i -
        \\hat{y}^{\\text{corrupted}}_i}{\\hat{y}^{\\text{clean}}_i -
        \\hat{y}^{\\text{corrupted}}_i + \\epsilon}

    - ``effect = 0``: patching had no effect (layer is not important)
    - ``effect = 1``: patching fully recovered the clean prediction
    - ``effect > 1`` or ``< 0``: overshooting or opposite effect

    Args:
        preds_clean: Predictions on clean data, shape ``[N_test]``.
        preds_corrupted: Predictions on corrupted data (no patching),
            shape ``[N_test]``.
        preds_patched: Predictions on corrupted data with one layer
            patched from clean cache, shape ``[N_test]``.
        eps: Small constant to avoid division by zero.

    Returns:
        Dict with keys:

        - ``'per_sample'``: ``np.ndarray`` of per-sample patch effects
        - ``'mean'``: ``float``, mean patch effect across samples
        - ``'abs_mean'``: ``float``, mean of absolute patch effects
    """
    clean = np.asarray(preds_clean, dtype=np.float64)
    corrupted = np.asarray(preds_corrupted, dtype=np.float64)
    patched = np.asarray(preds_patched, dtype=np.float64)

    per_sample = (patched - corrupted) / (clean - corrupted + eps)

    return {
        "per_sample": per_sample,
        "mean": float(np.mean(per_sample)),
        "abs_mean": float(np.mean(np.abs(per_sample))),
    }
