# pyright: reportMissingImports=false
"""Steering vectors for TabPFN model output manipulation.

Implements contrastive activation addition: extract a direction vector from
contrastive dataset pairs (e.g., high-α vs low-α), then add scaled versions
to the residual stream to steer model predictions.

Reference:
    - Turner et al. "Activation Addition" (2023)
    - Todd et al. "Function Vectors in Large Language Models" (ICLR 2024)
    - Hendel et al. "In-Context Learning Creates Task Vectors" (EMNLP 2023)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy.stats import pearsonr
from tabpfn import TabPFNRegressor
from torch import nn

from .tabpfn_hooker import TabPFNHookedModel

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _make_steering_hook(
    direction: torch.Tensor,
    lambda_val: float,
    token_idx: int = -1,
) -> Any:
    """Create forward hook that adds steering vector to layer output."""

    def hook(
        module: nn.Module,
        input: Any,  # noqa: A002
        output: torch.Tensor,
    ) -> torch.Tensor:
        steered = output.clone()
        direction_tensor = direction.to(output.device)
        steered[0, :, token_idx, :] = (
            steered[0, :, token_idx, :] + lambda_val * direction_tensor
        )
        return steered

    return hook


class TabPFNSteeringVector:
    """Extract and apply steering vectors to TabPFN.

    Usage:
        steerer = TabPFNSteeringVector(model)

        # Extract direction from contrastive pair
        v_hat = steerer.extract_direction(
            X_train_high, y_train_high,  # high-alpha dataset
            X_train_low, y_train_low,    # low-alpha dataset
            X_test,                      # shared test features
            layer=6,
        )

        # Steer predictions
        preds_steered = steerer.steer(X_test, layer=6, direction=v_hat, lambda_val=1.0)

        # Sweep lambda values
        results = steerer.sweep_lambda(
            X_test,
            layer=6,
            direction=v_hat,
            lambdas=[-2, -1, 0, 1, 2],
        )
    """

    def __init__(self, model: TabPFNRegressor) -> None:
        """Initialize steerer with a TabPFNRegressor.

        The model should generally be fitted before steering operations.

        Args:
            model: TabPFNRegressor instance.
        """
        self.model = model

    def _validate_fitted(self) -> None:
        """Validate that underlying TabPFN model has been fitted."""
        if not hasattr(self.model, "model_"):
            raise AttributeError(
                "Model must be fitted before steering. Call model.fit(X, y) first."
            )

    def _validate_layer(self, layer: int) -> None:
        """Validate transformer layer index."""
        self._validate_fitted()
        n_layers = len(self.model.model_.transformer_encoder.layers)
        if not 0 <= layer < n_layers:
            raise ValueError(f"layer must be 0..{n_layers - 1}, got {layer}")

    @staticmethod
    def _extract_token_mean(
        activation: torch.Tensor,
        token_idx: int,
    ) -> torch.Tensor:
        """Extract mean activation for one token position over sequence axis."""
        token_acts = activation[0, :, token_idx, :]
        return token_acts.mean(dim=0)

    def extract_direction(
        self,
        X_train_high: NDArray[np.floating],
        y_train_high: NDArray[np.floating],
        X_train_low: NDArray[np.floating],
        y_train_low: NDArray[np.floating],
        X_test: NDArray[np.floating],
        layer: int,
        token_idx: int = -1,
        *,
        X_val: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Extract steering direction from a contrastive dataset pair.

        1. Fit model on high dataset, run forward_with_cache -> get high activations.
        2. Fit model on low dataset, run forward_with_cache -> get low activations.
        3. Mean difference:
           v = mean(act_high[:, token_idx, :]) - mean(act_low[:, token_idx, :]).
        4. Normalize: v_hat = v / ||v||.

        Args:
            X_train_high: High-condition training features.
            y_train_high: High-condition training labels.
            X_train_low: Low-condition training features.
            y_train_low: Low-condition training labels.
            X_test: Shared test features (kept for backward compatibility).
            layer: Transformer layer index to extract from.
            token_idx: Feature-block token index (default: label token -1).
            X_val: Held-out validation features for direction extraction.
                If provided, activations are computed on X_val instead of
                X_test, avoiding test-set leakage. Recommended for all
                new experiments.

        Returns:
            Unit direction vector with shape [192].

        Raises:
            ValueError: If direction norm is near zero.
        """
        # Require explicit X_val to prevent silent test-set leakage in new code.
        # Backward-compat: callers that explicitly want X_test must pass X_val=X_test.
        if X_val is None:
            import warnings
            warnings.warn(
                "extract_direction called without X_val; falling back to X_test. "
                "This is a potential data-leakage path. Pass X_val explicitly "
                "(use a held-out split). To opt into the legacy behavior, pass "
                "X_val=X_test.",
                category=UserWarning,
                stacklevel=2,
            )
            X_probe = X_test
        else:
            X_probe = X_val

        self.model.fit(X_train_high, y_train_high)
        self._validate_layer(layer)
        hooker_high = TabPFNHookedModel(self.model)
        _, cache_high = hooker_high.forward_with_cache(X_probe)
        act_high = cache_high["layers"][layer]

        self.model.fit(X_train_low, y_train_low)
        self._validate_layer(layer)
        hooker_low = TabPFNHookedModel(self.model)
        _, cache_low = hooker_low.forward_with_cache(X_probe)
        act_low = cache_low["layers"][layer]

        mean_high = self._extract_token_mean(act_high, token_idx)
        mean_low = self._extract_token_mean(act_low, token_idx)
        direction = mean_high - mean_low

        norm = torch.linalg.norm(direction)
        if torch.isclose(norm.float(), torch.tensor(0.0, device=norm.device), atol=1e-12):
            raise ValueError(
                "Extracted direction has near-zero norm; cannot normalize."
            )

        direction_hat = direction / norm
        return direction_hat.detach().cpu().numpy().astype(np.float64)

    def steer(
        self,
        X_test: NDArray[np.floating],
        layer: int,
        direction: NDArray[np.floating],
        lambda_val: float,
        token_idx: int = -1,
    ) -> NDArray[np.floating]:
        """Apply steering direction to model predictions.

        1. Register forward hook on selected layer.
        2. Run ``model.predict(X_test)``.
        3. Remove hook.
        4. Return steered predictions.

        Args:
            X_test: Test features, shape [N_test, n_features].
            layer: Transformer layer index.
            direction: Steering direction vector of shape [192].
            lambda_val: Steering strength.
            token_idx: Feature-block token index (default: label token -1).

        Returns:
            Steered predictions as np.ndarray.
        """
        self._validate_layer(layer)

        direction_array = np.asarray(direction, dtype=np.float64).reshape(-1)
        direction_tensor = torch.from_numpy(direction_array).to(dtype=torch.float32)

        target_layer = self.model.model_.transformer_encoder.layers[layer]
        handle = target_layer.register_forward_hook(
            _make_steering_hook(direction_tensor, lambda_val, token_idx)
        )

        try:
            predictions = self.model.predict(X_test)
        finally:
            handle.remove()

        return np.asarray(predictions)

    def sweep_lambda(
        self,
        X_test: NDArray[np.floating],
        layer: int,
        direction: NDArray[np.floating],
        lambdas: list[float] | None = None,
        token_idx: int = -1,
    ) -> dict[str, Any]:
        """Sweep steering strengths and collect per-lambda predictions.

        Default lambda values: ``[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]``.

        Args:
            X_test: Test features.
            layer: Transformer layer index.
            direction: Steering direction vector.
            lambdas: Steering strengths to test.
            token_idx: Feature-block token index.

        Returns:
            {
                'lambdas': [...],
                'predictions': {lambda_val: preds_array, ...},
                'mean_preds': {lambda_val: float, ...},
            }
        """
        lambda_values = (
            [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] if lambdas is None else lambdas
        )

        predictions_by_lambda: dict[float, NDArray[np.floating]] = {}
        means_by_lambda: dict[float, float] = {}

        for lambda_val in lambda_values:
            preds = self.steer(
                X_test=X_test,
                layer=layer,
                direction=direction,
                lambda_val=float(lambda_val),
                token_idx=token_idx,
            )
            predictions_by_lambda[float(lambda_val)] = preds
            means_by_lambda[float(lambda_val)] = float(np.mean(preds))

        return {
            "lambdas": [float(v) for v in lambda_values],
            "predictions": predictions_by_lambda,
            "mean_preds": means_by_lambda,
        }


def compute_steering_effect(
    lambdas: list[float],
    mean_predictions: dict[float, float],
) -> dict[str, float]:
    """Compute steering effect metrics from a lambda sweep.

    Args:
        lambdas: Ordered lambda values used in steering sweep.
        mean_predictions: Mapping from lambda value to mean prediction.

    Returns:
        {
            'pearson_r': float,
            'pearson_p': float,
            'slope': float,
            'prediction_range': float,
        }

    Raises:
        ValueError: If fewer than 2 lambda values are provided.
        KeyError: If a lambda is missing from ``mean_predictions``.
    """
    if len(lambdas) < 2:
        raise ValueError("Need at least two lambda values to compute steering effect.")

    x = np.asarray(lambdas, dtype=np.float64)
    y = np.asarray([mean_predictions[float(l)] for l in lambdas], dtype=np.float64)

    pearson_r, pearson_p = pearsonr(x, y)
    slope, _ = np.polyfit(x, y, 1)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "slope": float(slope),
        "prediction_range": float(np.max(y) - np.min(y)),
    }
