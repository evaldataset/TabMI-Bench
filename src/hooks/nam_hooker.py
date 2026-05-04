# pyright: reportMissingImports=false
"""NAM (Neural Additive Model) hook implementation for TabMI-Bench.

NAM [Agarwal et al., 2021, arXiv:2004.13912] is architecturally distinct
from transformer-based TFMs: it uses per-feature MLPs (``feature nets'')
whose scalar outputs are summed. This makes it a genuine out-of-family
holdout model for testing whether the TabMI-Bench computation profile
taxonomy transfers beyond transformer/tree-based architectures.

Implementation: simple PyTorch NAM trained per-task (in-context fit),
matching the TFM interface expected by TabMI-Bench experiments.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FeatureNet(nn.Module):
    """Per-feature MLP producing a scalar contribution."""

    def __init__(self, hidden_dims: tuple[int, ...] = (64, 64, 32)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 1
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N] -> [N, 1] -> [N, 1]
        return self.net(x.unsqueeze(-1)).squeeze(-1)


class NAMModel(nn.Module):
    """Neural Additive Model: output = bias + sum_i FeatureNet_i(x_i).

    Layers (for hook extraction):
        L0: concatenated hidden-1 activations of all FeatureNets (post-ReLU)
        L1: concatenated hidden-2 activations
        L2: concatenated hidden-3 activations (pre-output)
    """

    def __init__(self, n_features: int, hidden_dims: tuple[int, ...] = (64, 64, 32)) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.feature_nets = nn.ModuleList([FeatureNet(hidden_dims) for _ in range(n_features)])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, F]
        contributions = torch.stack(
            [net(x[:, i]) for i, net in enumerate(self.feature_nets)], dim=1
        )  # [N, F]
        return contributions.sum(dim=1) + self.bias

    def get_layer_activations(self, x: torch.Tensor, layer_idx: int) -> np.ndarray:
        """Extract activations at a specific layer of all feature nets (concatenated).

        layer_idx 0..len(hidden_dims)-1 corresponds to post-ReLU hidden activations.
        Returns [N, F*hidden_dims[layer_idx]] concatenated activations.
        """
        if not 0 <= layer_idx < len(self.hidden_dims):
            raise ValueError(f"layer_idx must be 0..{len(self.hidden_dims) - 1}, got {layer_idx}")

        feature_activations: list[torch.Tensor] = []
        for net in self.feature_nets:
            h = x.unsqueeze(-1)  # [N, 1] per feature (but we iterate features)
            # Actually we need to pass per-feature input
            pass

        # Re-do properly: iterate feature-by-feature
        collected: list[torch.Tensor] = []
        for i, net in enumerate(self.feature_nets):
            h = x[:, i].unsqueeze(-1)  # [N, 1]
            # Step through net.net children. layer_idx refers to hidden block (Linear+ReLU pair).
            linear_count = 0
            for child in net.net:
                h = child(h)
                if isinstance(child, nn.ReLU):
                    if linear_count == layer_idx + 1:
                        collected.append(h)
                        break
                    linear_count += 0  # counting happens on ReLU not Linear
                if isinstance(child, nn.Linear):
                    linear_count += 1
                    if linear_count == layer_idx + 1 and not isinstance(
                        net.net[min(len(net.net) - 1, list(net.net).index(child) + 1)], nn.ReLU
                    ):
                        # Final Linear (output) — rare, skip
                        pass
        if not collected:
            raise RuntimeError(f"Could not extract layer {layer_idx}")
        # Each collected[i]: [N, hidden_dims[layer_idx]]. Concatenate along feature axis.
        return torch.cat(collected, dim=1).detach().cpu().numpy()


class NAMRegressor:
    """TabPFN/TabICL-style regressor wrapper for NAM.

    Fits per-task (in-context): given (X_train, y_train), trains the NAM
    from scratch for a few epochs, then predicts on X_test.
    """

    def __init__(
        self,
        device: str = "cpu",
        hidden_dims: tuple[int, ...] = (64, 64, 32),
        n_epochs: int = 200,
        lr: float = 1e-2,
        batch_size: int = 64,
        random_state: int = 42,
    ) -> None:
        self.device = device
        self.hidden_dims = hidden_dims
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.model_: NAMModel | None = None
        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NAMRegressor":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0) + 1e-8
        X_n = (X - self.x_mean_) / self.x_std_
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std() + 1e-8)
        y_n = (y - self.y_mean_) / self.y_std_

        self.model_ = NAMModel(n_features=X.shape[1], hidden_dims=self.hidden_dims).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        X_tensor = torch.from_numpy(X_n).to(self.device)
        y_tensor = torch.from_numpy(y_n).to(self.device)
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=min(self.batch_size, len(y)),
            shuffle=True,
        )

        self.model_.train()
        for _ in range(self.n_epochs):
            for xb, yb in loader:
                opt.zero_grad()
                pred = self.model_(xb)
                loss = ((pred - yb) ** 2).mean()
                loss.backward()
                opt.step()

        self.model_.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model_ is not None and self.x_mean_ is not None
        X_n = (np.asarray(X, dtype=np.float32) - self.x_mean_) / self.x_std_
        with torch.no_grad():
            pred = self.model_(torch.from_numpy(X_n).to(self.device)).cpu().numpy()
        return pred * self.y_std_ + self.y_mean_


class NAMHookedModel:
    """Hook-based interface for NAM, compatible with TabMI-Bench experiments.

    Layer convention:
        - num_layers = len(hidden_dims)  (e.g., 3 for default config)
        - layer i = post-ReLU hidden activation at hidden block i of all feature nets
    """

    def __init__(self, fitted_model: NAMRegressor) -> None:
        if fitted_model.model_ is None:
            raise AttributeError("Pass a fitted NAMRegressor.")
        self.model = fitted_model
        self.num_layers = len(fitted_model.hidden_dims)

    def forward_with_cache(self, X_test: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Run inference and cache activations from each hidden layer."""
        assert self.model.model_ is not None and self.model.x_mean_ is not None
        X_n = (np.asarray(X_test, dtype=np.float32) - self.model.x_mean_) / self.model.x_std_
        X_tensor = torch.from_numpy(X_n).to(self.model.device)

        # Hook each feature net's hidden layers to capture activations
        layer_activations: dict[int, list[torch.Tensor]] = {i: [] for i in range(self.num_layers)}

        # Manually forward and collect: for each feature net, step through.
        with torch.no_grad():
            all_contribs = []
            for net in self.model.model_.feature_nets:
                # Process per feature
                pass
            # Use direct approach: run feature-wise forward collecting after each ReLU
            per_feature_per_layer: list[list[torch.Tensor]] = []
            for i, net in enumerate(self.model.model_.feature_nets):
                h = X_tensor[:, i].unsqueeze(-1)
                layers_this_feature: list[torch.Tensor] = []
                linear_hit = 0
                for child in net.net:
                    h = child(h)
                    if isinstance(child, nn.ReLU):
                        layers_this_feature.append(h.clone())
                per_feature_per_layer.append(layers_this_feature)

            # Concatenate per-feature activations at each layer
            cache_layers = []
            for layer_i in range(self.num_layers):
                stacked = torch.cat([pf[layer_i] for pf in per_feature_per_layer], dim=1)
                cache_layers.append(stacked.cpu().numpy())

            # Final prediction
            pred_n = self.model.model_(X_tensor).cpu().numpy()
            pred = pred_n * self.model.y_std_ + self.model.y_mean_

        cache = {"layers": cache_layers}
        return pred, cache

    def get_layer_activations(self, cache: dict[str, Any], layer_idx: int) -> np.ndarray:
        return cache["layers"][layer_idx]


if __name__ == "__main__":
    # Self-test
    print("=" * 60)
    print("NAM Self-Test")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n_train, n_test, d = 100, 50, 3
    X = rng.standard_normal((n_train + n_test, d)).astype(np.float32)
    y = (X[:, 0] * 2.0 + X[:, 1] * X[:, 2] + rng.standard_normal(n_train + n_test) * 0.1).astype(np.float32)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:], y[n_train:]

    print("\n[1] Fitting NAM...")
    model = NAMRegressor(device="cpu", n_epochs=50, random_state=42)
    model.fit(X_tr, y_tr)

    print("\n[2] Predicting...")
    pred = model.predict(X_te)
    mse = float(np.mean((pred - y_te) ** 2))
    print(f"    Test MSE: {mse:.4f}")

    print("\n[3] Hook test...")
    hooker = NAMHookedModel(model)
    pred2, cache = hooker.forward_with_cache(X_te)
    assert pred2.shape == (n_test,), f"Expected ({n_test},), got {pred2.shape}"
    assert len(cache["layers"]) == 3, f"Expected 3 layers, got {len(cache['layers'])}"
    for i, act in enumerate(cache["layers"]):
        print(f"    L{i}: shape={act.shape}")

    print("\n[PASS] NAM hook tests complete.")
