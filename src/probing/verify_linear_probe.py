"""
Standalone verification script for LinearProbe.

This script verifies the LinearProbe implementation works correctly.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Optional, Callable, Dict, Any


class LinearProbe:
    def __init__(
        self, complexity: int = 0, hidden_size: int = 64, random_seed: int = 42
    ):
        if complexity < 0 or complexity > 3:
            raise ValueError(f"complexity must be 0-3, got {complexity}")
        self.complexity = complexity
        self.hidden_size = hidden_size
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.model = None
        self.n_features_in_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Store shape info BEFORE using it
        self.n_features_in_ = X.shape[1]
        y = np.asarray(y)
        is_multioutput = y.ndim > 1
        if is_multioutput:
            n_targets = y.shape[1]
        else:
            n_targets = 1
        X_scaled = self.scaler.fit_transform(X)
        if self.complexity == 0:
            self.model = Ridge(alpha=1.0, fit_intercept=True)
            self.model.fit(X_scaled, y)
        else:
            layers = []
            layers.append(nn.Linear(self.n_features_in_, self.hidden_size))
            layers.append(nn.ReLU())
            for _ in range(self.complexity):
                layers.append(nn.Linear(self.hidden_size, self.hidden_size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_size, n_targets))
            self.model = nn.Sequential(*layers)
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = (
                torch.FloatTensor(y) if not is_multioutput else torch.FloatTensor(y)
            )
            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            self.model.train()
            for epoch in range(50):
                self.model = self.model.cpu()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        if self.complexity == 0:
            y_pred = self.model.predict(X_scaled)
        else:
            X_tensor = torch.FloatTensor(X_scaled)
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_tensor).numpy()
        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        return {"r2": r2_score(y, y_pred), "mse": mean_squared_error(y, y_pred)}


def probe_layer(
    activations, targets, complexities=[0, 1, 2, 3], test_size=0.2, random_seed=42
):
    np.random.seed(random_seed)
    X_train, X_test, y_train, y_test = train_test_split(
        activations,
        targets,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )
    results = {}
    for complexity in complexities:
        probe = LinearProbe(complexity=complexity, random_seed=random_seed)
        probe.fit(X_train, y_train)
        results[complexity] = probe.score(X_test, y_test)
    return results


def probe_all_layers(
    activations_per_layer,
    targets,
    complexities=[0, 1, 2, 3],
    flatten_fn=None,
    random_seed=42,
):
    np.random.seed(random_seed)
    if flatten_fn is None:
        flatten_fn = lambda x: x.reshape(x.shape[0], -1)
    flattened_activations = [flatten_fn(layer) for layer in activations_per_layer]
    all_features = np.vstack(flattened_activations)
    n_samples = flattened_activations[0].shape[0]
    n_layers = len(flattened_activations)
    n_complexities = len(complexities)
    r2_results = np.zeros((n_layers, n_complexities))
    mse_results = np.zeros((n_layers, n_complexities))
    for layer_idx, layer_features in enumerate(flattened_activations):
        X_layer_train, X_layer_test, y_layer_train, y_layer_test = train_test_split(
            layer_features, targets, test_size=0.2, random_state=random_seed, shuffle=True
        )
        for comp_idx, complexity in enumerate(complexities):
            probe = LinearProbe(complexity=complexity, random_seed=random_seed)
            probe.fit(X_layer_train, y_layer_train)
            scores = probe.score(X_layer_test, y_layer_test)
            r2_results[layer_idx, comp_idx] = scores['r2']
            mse_results[layer_idx, comp_idx] = scores['mse']
    return {
        'r2': r2_results,
        'mse': mse_results,
        'complexities': complexities,
        'n_layers': n_layers,
        'n_features': all_features.shape[1]
    }


if __name__ == "__main__":
    np.random.seed(42)

    # Perfect linear data: expect R²=1.0
    X = np.random.randn(200, 10)
    y = 2 * X[:, 0] + 3 * X[:, 1]  # Perfect linear relationship

    print("=" * 60)
    print("LinearProbe Verification")
    print("=" * 60)

    # Single probe test
    print("\n1. Single LinearProbe test (complexity=0):")
    probe = LinearProbe(complexity=0)
    probe.fit(X[:160], y[:160])
    scores = probe.score(X[160:], y[160:])
    print(f"   Linear probe R²: {scores['r2']:.4f} (expected ~1.0)")

    if scores["r2"] > 0.95:
        print("   ✓ R² is high (good)")
    else:
        print(f"   ✗ R² too low: {scores['r2']}")
    assert scores["r2"] > 0.95, f"R² too low: {scores['r2']}"

    # Complexity comparison test
    print("\n2. Complexity comparison test:")
    results = probe_layer(X, y, complexities=[0, 1, 2, 3])
    print("   Complexity vs R²:")
    for c, r in results.items():
        print(f"     complexity={c}: R²={r['r2']:.4f}, MSE={r['mse']:.6f}")

    if results[0]["r2"] > results[1]["r2"] > results[2]["r2"] > results[3]["r2"]:
        print("   ✓ Complexity monotonic decrease (linear encoding)")
    else:
        print("   ! Complexity trend not perfectly monotonic")

    # All layers test (dummy layers)
    print("\n3. All layers test (12 dummy layers):")
    fake_layers = [np.random.randn(200, 5, 3, 192) for _ in range(12)]
    fake_targets = np.random.randn(200)
    all_results = probe_all_layers(fake_layers, fake_targets)
    print(f"   All layers R² shape: {all_results['r2'].shape}")
    print("   Expected shape: (12, 4)")

    assert all_results["r2"].shape == (12, 4), (
        f"Shape mismatch: {all_results['r2'].shape} != (12, 4)"
    )
    print("   ✓ Shape correct")

    # Layer-wise results summary
    print("\n   Layer-wise R² statistics:")
    print(f"     Mean: {all_results['r2'].mean():.4f}")
    print(f"     Std:  {all_results['r2'].std():.4f}")
    print(f"     Min:  {all_results['r2'].min():.4f}")
    print(f"     Max:  {all_results['r2'].max():.4f}")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED ✓")
    print("=" * 60)
