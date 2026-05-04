# pyright: reportMissingImports=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from iltm.utils import transform_data_for_main_network

if TYPE_CHECKING:
    from numpy.typing import NDArray


class iLTMHookedModel:
    NUM_LAYERS = 4
    HIDDEN_DIM = 512

    def __init__(self, fitted_model: Any) -> None:
        if not hasattr(fitted_model, "predictors_"):
            raise AttributeError(
                "Model must be fitted before hooking. Call model.fit(X, y) first."
            )

        self.model = fitted_model
        self._pytorch_model = fitted_model._model
        self._predictor = fitted_model.predictors_[0]
        self._n_layers = 4

    def forward_with_cache(
        self,
        X_test: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], dict[str, Any]]:
        preprocessing_objects = self.model.preprocessors_[0]
        X_test_input = X_test
        if self.model.tree_embedding:
            if self.model.tree_for_each_predictor:
                tree_model = self.model.tr_[0]
                X_emb = tree_model.transform(X_test)
                if self.model.concat_tree_with_orig_features:
                    X_base = X_test
                    if tree_model.n_orig_features_to_keep_ is not None:
                        X_base = X_base[:, : tree_model.n_orig_features_to_keep_]
                    X_test_input = np.concatenate([X_base, X_emb], axis=1)
                else:
                    X_test_input = X_emb
            else:
                tree_model = self.model.tr_
                X_emb = tree_model.transform(X_test)
                if self.model.concat_tree_with_orig_features:
                    X_base = X_test
                    if tree_model.n_orig_features_to_keep_ is not None:
                        X_base = X_base[:, : tree_model.n_orig_features_to_keep_]
                    X_test_input = np.concatenate([X_base, X_emb], axis=1)
                else:
                    X_test_input = X_emb

        X_test_tensor = self.model._preprocess_test_data(
            X_test_input,
            preprocessing_objects,
        )

        if self._predictor["feature_bagging_idxs"] is not None:
            X_test_tensor = X_test_tensor[:, self._predictor["feature_bagging_idxs"]]

        device = self.model.device
        predictor = self.model._move_predictor_to_device(self._predictor, device=device)
        model_cfg = vars(self.model._model)

        try:
            X_transformed = transform_data_for_main_network(
                X=X_test_tensor,
                cfg=model_cfg,
                rf=predictor["rf"],
                pca=predictor["pca"],
                norm=predictor["norm"],
                device=device,
            )

            intermediates: list[NDArray[np.floating]] = [
                X_transformed.detach().cpu().numpy()
            ]

            x = X_transformed
            main_network = predictor["main_network"]
            residual = x

            with torch.no_grad():
                for n, layer in enumerate(main_network):
                    if n % 2 == 0:
                        residual = x

                    x = layer.to(device)(x)

                    if n % 2 == 1 and n != len(main_network) - 1:
                        x = x + residual

                    if n != len(main_network) - 1:
                        x = torch.relu(x)

                    intermediates.append(x.detach().cpu().numpy())

            predictions = intermediates[-1].squeeze(-1)

            cache: dict[str, Any] = {
                "layers": intermediates,
            }
        finally:
            self.model._move_predictor_to_cpu(predictor)

        return predictions, cache

    def get_layer_activations(
        self,
        cache: dict[str, Any],
        layer_idx: int,
    ) -> NDArray[np.floating]:
        return cache["layers"][layer_idx]

    @property
    def num_layers(self) -> int:
        return self._n_layers

    @property
    def hidden_dim(self) -> int:
        return self.HIDDEN_DIM


if __name__ == "__main__":
    from iltm import iLTMRegressor

    print("=" * 60)
    print("iLTMHookedModel Verification")
    print("=" * 60)

    np.random.seed(42)
    X_train = np.random.randn(50, 3)
    y_train = 2 * X_train[:, 0] + 3 * X_train[:, 1] + np.random.randn(50) * 0.1
    X_test = np.random.randn(5, 3)

    print("\n[1] Fitting iLTMRegressor...")
    model = iLTMRegressor(device="cpu")
    model.fit(X_train, y_train)
    print("    Done.")

    print("\n[2] Running forward_with_cache...")
    hooker = iLTMHookedModel(model)
    pred, cache = hooker.forward_with_cache(X_test)
    print("    Done.")

    print("\n[3] Shape verification:")
    print(f"    Predictions shape: {pred.shape}")
    print(f"    Number of cached layers: {len(cache['layers'])}")
    print(f"    Layer 0 shape: {cache['layers'][0].shape}")
    print(f"    Layer 1 shape: {cache['layers'][1].shape}")
    print(f"    Layer 2 shape: {cache['layers'][2].shape}")
    print(f"    Layer 3 shape: {cache['layers'][3].shape}")

    print("\n[4] Helper method verification:")
    for i in range(4):
        activation = hooker.get_layer_activations(cache, i)
        print(f"    get_layer_activations(cache, {i}) shape: {activation.shape}")

    assert pred.shape == (5,), f"Expected (5,), got {pred.shape}"
    assert len(cache["layers"]) == 4, f"Expected 4 layers, got {len(cache['layers'])}"
    assert cache["layers"][0].shape[1] == 512, (
        f"Expected layer 0 hidden dim 512, got {cache['layers'][0].shape[1]}"
    )
    assert cache["layers"][1].shape[1] == 512, (
        f"Expected layer 1 hidden dim 512, got {cache['layers'][1].shape[1]}"
    )
    assert cache["layers"][2].shape[1] == 512, (
        f"Expected layer 2 hidden dim 512, got {cache['layers'][2].shape[1]}"
    )
    assert cache["layers"][3].shape[1] == 1, (
        f"Expected output dim 1, got {cache['layers'][3].shape[1]}"
    )

    layer_0 = hooker.get_layer_activations(cache, 0)
    layer_1 = hooker.get_layer_activations(cache, 1)
    layer_2 = hooker.get_layer_activations(cache, 2)
    layer_3 = hooker.get_layer_activations(cache, 3)

    assert layer_0.shape == (5, 512), f"Expected (5, 512), got {layer_0.shape}"
    assert layer_1.shape == (5, 512), f"Expected (5, 512), got {layer_1.shape}"
    assert layer_2.shape == (5, 512), f"Expected (5, 512), got {layer_2.shape}"
    assert layer_3.shape == (5, 1), f"Expected (5, 1), got {layer_3.shape}"

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
