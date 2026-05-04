# pyright: reportMissingImports=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy.stats import pearsonr
from tabicl import TabICLRegressor
from torch import nn

try:
    from .tabicl_hooker import TabICLHookedModel
except ImportError:
    from tabicl_hooker import TabICLHookedModel

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _make_steering_hook(direction: torch.Tensor, lambda_val: float) -> Any:
    def hook(
        module: nn.Module,
        input: Any,  # noqa: A002
        output: torch.Tensor,
    ) -> torch.Tensor:
        steered = output.clone()
        direction_tensor = direction.to(output.device)
        steered += lambda_val * direction_tensor.view(1, 1, -1)
        return steered

    return hook


class TabICLSteeringVector:
    def __init__(self, model: TabICLRegressor) -> None:
        self.model = model

    def _validate_fitted(self) -> None:
        if not hasattr(self.model, "model_"):
            raise AttributeError(
                "Model must be fitted before steering. Call model.fit(X, y) first."
            )

    def _validate_layer(self, layer: int) -> None:
        self._validate_fitted()
        n_layers = len(self.model.model_.icl_predictor.tf_icl.blocks)
        if not 0 <= layer < n_layers:
            raise ValueError(f"layer must be 0..{n_layers - 1}, got {layer}")

    def extract_direction(
        self,
        X_train_high: NDArray[np.floating],
        y_train_high: NDArray[np.floating],
        X_train_low: NDArray[np.floating],
        y_train_low: NDArray[np.floating],
        X_test: NDArray[np.floating],
        layer: int,
        *,
        X_val: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
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
        hooker_high = TabICLHookedModel(self.model)
        _, cache_high = hooker_high.forward_with_cache(X_probe)
        act_high = hooker_high.get_layer_activations(cache_high, layer)

        self.model.fit(X_train_low, y_train_low)
        self._validate_layer(layer)
        hooker_low = TabICLHookedModel(self.model)
        _, cache_low = hooker_low.forward_with_cache(X_probe)
        act_low = hooker_low.get_layer_activations(cache_low, layer)

        mean_high = np.asarray(act_high, dtype=np.float64).mean(axis=0)
        mean_low = np.asarray(act_low, dtype=np.float64).mean(axis=0)
        direction = mean_high - mean_low

        norm = float(np.linalg.norm(direction))
        if np.isclose(norm, 0.0, atol=1e-12):
            raise ValueError(
                "Extracted direction has near-zero norm; cannot normalize."
            )

        direction_hat = direction / norm
        return direction_hat.astype(np.float64)

    def steer(
        self,
        X_test: NDArray[np.floating],
        layer: int,
        direction: NDArray[np.floating],
        lambda_val: float,
    ) -> NDArray[np.floating]:
        self._validate_layer(layer)

        direction_array = np.asarray(direction, dtype=np.float64).reshape(-1)
        if direction_array.shape[0] != 512:
            raise ValueError(
                f"direction must have shape [512], got [{direction_array.shape[0]}]"
            )
        direction_tensor = torch.from_numpy(direction_array).to(dtype=torch.float32)

        X_test_encoded = self.model.X_encoder_.transform(X_test)
        data = self.model.ensemble_generator_.transform(X_test_encoded, mode="both")
        Xs, ys = next(iter(data.values()))

        X_tensor = torch.from_numpy(Xs[0:1]).float()
        y_tensor = torch.from_numpy(ys[0:1]).float()

        block = self.model.model_.icl_predictor.tf_icl.blocks[layer]
        handle = block.register_forward_hook(
            _make_steering_hook(direction_tensor, lambda_val)
        )

        try:
            device = self.model.device_
            self.model.model_.eval()
            with torch.no_grad():
                raw_q = self.model.model_._inference_forward(
                    X_tensor.to(device),
                    y_tensor.to(device),
                    inference_config=self.model.inference_config_,
                )
        finally:
            handle.remove()

        pred = raw_q.mean(dim=-1).detach().cpu().numpy()
        pred = self.model.y_scaler_.inverse_transform(pred.reshape(-1, 1)).reshape(-1)
        return np.asarray(pred)

    def sweep_lambda(
        self,
        X_test: NDArray[np.floating],
        layer: int,
        direction: NDArray[np.floating],
        lambdas: list[float] | None = None,
    ) -> dict[str, Any]:
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


if __name__ == "__main__":
    print("=" * 60)
    print("TabICLSteeringVector Verification")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n_train = 30
    n_test = 5

    def make_dataset(
        alpha: float, beta: float, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        x = rng.standard_normal((n, 3))
        noise = 0.05 * rng.standard_normal(n)
        y = alpha * x[:, 0] + beta * x[:, 1] + noise
        return x.astype(np.float64), y.astype(np.float64)

    print("\n[1] Building contrastive datasets...")
    X_train_high, y_train_high = make_dataset(alpha=5.0, beta=1.0, n=n_train)
    X_train_low, y_train_low = make_dataset(alpha=1.0, beta=1.0, n=n_train)
    X_test, _ = make_dataset(alpha=3.0, beta=1.0, n=n_test)
    print("    Done.")

    print("\n[2] Fitting base model and extracting direction (layer 5)...")
    model = TabICLRegressor(device="cpu", random_state=42)
    model.fit(X_train_low, y_train_low)

    steerer = TabICLSteeringVector(model)
    direction = steerer.extract_direction(
        X_train_high=X_train_high,
        y_train_high=y_train_high,
        X_train_low=X_train_low,
        y_train_low=y_train_low,
        X_test=X_test,
        layer=5,
    )
    print("    Done.")

    print("\n[3] Direction checks...")
    assert direction.shape == (512,), f"Expected (512,), got {direction.shape}"
    assert np.isclose(np.linalg.norm(direction), 1.0, atol=1e-6), (
        f"Direction norm should be 1.0, got {np.linalg.norm(direction):.8f}"
    )
    print("    Passed.")

    print("\n[4] Steering check at lambda=1.0...")
    baseline = steerer.steer(X_test, layer=5, direction=direction, lambda_val=0.0)
    steered = steerer.steer(X_test, layer=5, direction=direction, lambda_val=1.0)
    assert baseline.shape == (n_test,), f"Expected ({n_test},), got {baseline.shape}"
    assert steered.shape == (n_test,), f"Expected ({n_test},), got {steered.shape}"
    assert not np.allclose(steered, baseline), (
        "Steered predictions should differ from baseline predictions"
    )
    print("    Passed.")

    print("\n[5] Lambda sweep check...")
    sweep_lambdas = [-2.0, -1.0, 0.0, 1.0, 2.0]
    sweep = steerer.sweep_lambda(
        X_test=X_test,
        layer=5,
        direction=direction,
        lambdas=sweep_lambdas,
    )
    assert len(sweep["predictions"]) == 5, (
        f"Expected 5 prediction sets, got {len(sweep['predictions'])}"
    )
    print("    Passed.")

    print("\n[6] Steering effect metrics check...")
    effect = compute_steering_effect(
        lambdas=sweep["lambdas"],
        mean_predictions=sweep["mean_preds"],
    )
    assert "pearson_r" in effect, "pearson_r missing from steering effect metrics"
    print("    Passed.")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
