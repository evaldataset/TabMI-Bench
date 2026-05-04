# pyright: reportMissingImports=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportUnannotatedClassAttribute=false
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from tabdpt.utils import pad_x


class TabDPTHookedModel:
    def __init__(self, model: Any, device: str = "cpu") -> None:
        if not hasattr(model, "is_fitted_") or not bool(model.is_fitted_):
            raise AttributeError(
                "Model must be fitted before hooking. Call model.fit(X, y) first."
            )
        if not hasattr(model, "model"):
            raise AttributeError(
                "TabDPTRegressor is missing internal model attribute `model`."
            )

        self.model = model
        self.device = torch.device(device)
        self._pytorch_model: nn.Module = model.model.to(self.device)
        self._layers: nn.ModuleList = self._pytorch_model.transformer_encoder
        self._n_layers = len(self._layers)

    def get_activations(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> dict[int, np.ndarray]:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "X_train and y_train must have the same number of samples."
            )
        if X_train.shape[0] != int(self.model.n_instances):
            raise ValueError(
                "Provided training set size does not match fitted TabDPTRegressor context."
            )

        train_x, train_y, test_x = self.model._prepare_prediction(X_test)

        x_train = pad_x(train_x[None, :, :], self.model.max_features).to(self.device)
        x_test = pad_x(test_x[None, :, :], self.model.max_features).to(self.device)
        y_train_t = train_y[None, :].to(self.device).unsqueeze(-1)

        cache: dict[int, torch.Tensor | None] = {i: None for i in range(self._n_layers)}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def _make_hook(layer_idx: int):
            def hook(_module: nn.Module, _inp: Any, out: torch.Tensor) -> None:
                cache[layer_idx] = out.detach().cpu()

            return hook

        try:
            for i, layer in enumerate(self._layers):
                handles.append(layer.register_forward_hook(_make_hook(i)))

            self._pytorch_model.eval()
            with torch.no_grad():
                _ = self._pytorch_model(
                    x_src=torch.cat([x_train, x_test], dim=1),
                    y_src=y_train_t,
                    task=self.model.mode,
                )
        finally:
            for h in handles:
                h.remove()

        activations: dict[int, np.ndarray] = {}
        for i in range(self._n_layers):
            layer_out = cache[i]
            if layer_out is None:
                raise RuntimeError(f"Missing activation for layer {i}")
            activations[i] = layer_out.squeeze(1).numpy()

        return activations

    @property
    def num_layers(self) -> int:
        return self._n_layers

    @property
    def hidden_dim(self) -> int:
        return int(self._layers[0].embed_dim)
