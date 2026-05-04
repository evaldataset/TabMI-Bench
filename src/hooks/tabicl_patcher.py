# pyright: reportMissingImports=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TabICLActivationPatcher:
    NUM_LAYERS = 12

    def __init__(self, model: Any) -> None:
        if not hasattr(model, "model_"):
            raise AttributeError(
                "Model must be fitted before patching. Call model.fit(X, y) first."
            )

        self.model = model
        self._pytorch_model: nn.Module = model.model_
        self._blocks: nn.ModuleList = self._pytorch_model.icl_predictor.tf_icl.blocks
        self._n_layers: int = len(self._blocks)

    def _get_layer(self, idx: int) -> nn.Module:
        return self._blocks[idx]

    def run_with_cache(
        self,
        X_test: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], dict[str, Any]]:
        cache: dict[str, Any] = {
            "layers": [None] * self._n_layers,
            "train_size": None,
        }
        handles: list[torch.utils.hooks.RemovableHandle] = []

        X_test_encoded = self.model.X_encoder_.transform(X_test)
        data = self.model.ensemble_generator_.transform(X_test_encoded, mode="both")
        Xs, ys = next(iter(data.values()))

        X_tensor = torch.from_numpy(Xs[0:1]).float()
        y_tensor = torch.from_numpy(ys[0:1]).float()

        def _make_cache_hook(store: list[torch.Tensor | None], idx: int) -> Any:
            def hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:  # noqa: A002
                store[idx] = output.detach().clone()

            return hook

        try:
            for i in range(self._n_layers):
                handles.append(
                    self._get_layer(i).register_forward_hook(
                        _make_cache_hook(cache["layers"], i)
                    )
                )

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

    def patched_run(
        self,
        X_test: NDArray[np.floating],
        clean_cache: dict[str, Any],
        patch_layer: int,
    ) -> NDArray[np.floating]:
        if not 0 <= patch_layer < self._n_layers:
            raise ValueError(
                f"patch_layer must be 0..{self._n_layers - 1}, got {patch_layer}"
            )
        if "layers" not in clean_cache:
            raise KeyError("clean_cache must contain 'layers' key")

        clean_activation = clean_cache["layers"][patch_layer]
        if clean_activation is None:
            raise ValueError(f"clean_cache['layers'][{patch_layer}] is None")

        X_test_encoded = self.model.X_encoder_.transform(X_test)
        data = self.model.ensemble_generator_.transform(X_test_encoded, mode="both")
        Xs, ys = next(iter(data.values()))

        X_tensor = torch.from_numpy(Xs[0:1]).float()
        y_tensor = torch.from_numpy(ys[0:1]).float()

        def _patch_hook(
            module: nn.Module,
            input: Any,  # noqa: A002
            output: torch.Tensor,
        ) -> torch.Tensor:
            return clean_activation.detach().clone().to(output.device)

        handle = self._get_layer(patch_layer).register_forward_hook(_patch_hook)

        try:
            device = self.model.device_
            self._pytorch_model.eval()
            with torch.no_grad():
                raw_quantiles = self._pytorch_model._inference_forward(
                    X_tensor.to(device),
                    y_tensor.to(device),
                    inference_config=self.model.inference_config_,
                )
        finally:
            handle.remove()

        pred = raw_quantiles.mean(dim=-1).detach().cpu().numpy()
        pred = self.model.y_scaler_.inverse_transform(pred.reshape(-1, 1)).reshape(-1)

        return pred

    def sweep_all_layers(
        self,
        X_test_corrupted: NDArray[np.floating],
        clean_cache: dict[str, Any],
        preds_clean: NDArray[np.floating],
        preds_corrupted: NDArray[np.floating],
    ) -> dict[str, Any]:
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
    clean = np.asarray(preds_clean, dtype=np.float64)
    corrupted = np.asarray(preds_corrupted, dtype=np.float64)
    patched = np.asarray(preds_patched, dtype=np.float64)

    per_sample = (patched - corrupted) / (clean - corrupted + eps)

    return {
        "per_sample": per_sample,
        "mean": float(np.mean(per_sample)),
        "abs_mean": float(np.mean(np.abs(per_sample))),
    }


if __name__ == "__main__":
    from tabicl import TabICLRegressor

    np.random.seed(42)

    n_train = 20
    n_test = 5

    X_train = np.random.randn(n_train, 3)
    noise = np.random.randn(n_train) * 0.1
    y_train = 2.0 * X_train[:, 0] + 3.0 * X_train[:, 1] + noise
    X_test = np.random.randn(n_test, 3)

    model = TabICLRegressor(device="cpu", random_state=42)
    model.fit(X_train, y_train)

    patcher = TabICLActivationPatcher(model)

    preds_clean, clean_cache = patcher.run_with_cache(X_test)
    assert preds_clean.shape == (5,), (
        f"Expected pred shape (5,), got {preds_clean.shape}"
    )
    assert len(clean_cache["layers"]) == 12, "Expected 12 cached layers"
    for i, layer in enumerate(clean_cache["layers"]):
        assert layer is not None, f"Layer {i} activation is None"
    assert clean_cache["train_size"] == 20, (
        f"Expected train_size 20, got {clean_cache['train_size']}"
    )

    X_test_corrupted = X_test * 2.0
    preds_corrupted, _ = patcher.run_with_cache(X_test_corrupted)

    preds_patched_l5 = patcher.patched_run(X_test_corrupted, clean_cache, patch_layer=5)
    assert preds_patched_l5.shape == (5,), (
        f"Expected patched pred shape (5,), got {preds_patched_l5.shape}"
    )
    assert not np.allclose(preds_patched_l5, preds_corrupted), (
        "Layer 5 patch did not change predictions from corrupted run"
    )

    sweep = patcher.sweep_all_layers(
        X_test_corrupted=X_test_corrupted,
        clean_cache=clean_cache,
        preds_clean=preds_clean,
        preds_corrupted=preds_corrupted,
    )

    most_important = sweep["most_important_layer"]
    assert isinstance(most_important, int), "most_important_layer must be int"
    assert 0 <= most_important < 12, (
        f"most_important_layer must be 0..11, got {most_important}"
    )

    print("ALL CHECKS PASSED")
