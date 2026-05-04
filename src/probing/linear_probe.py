"""
LinearProbe Framework for TabPFN Mechanistic Interpretability.

Provides simple linear probes to decode target information from hidden states.
Core principle: If complexity=0 (pure linear) achieves best R², then information
is linearly encoded in the hidden state.

Author: TFMI Research (2026)
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
    """
    Simple linear probe to decode target values from hidden states.

    complexity=0: Pure linear (Ridge regression) — fastest and most interpretable
    complexity=1: 1-hidden-layer MLP (input → hidden → output)
    complexity=2: 2-hidden-layer MLP (input → hidden → hidden → output)
    complexity=3: 3-hidden-layer MLP (input → hidden → hidden → hidden → output)

    Core Principle: If complexity=0 achieves highest R², information is linearly
    encoded in the hidden state.
    """

    def __init__(
        self, complexity: int = 0, hidden_size: int = 64, random_seed: int = 42
    ):
        """
        Args:
            complexity: 0=linear, 1/2/3=MLP hidden layer count
            hidden_size: MLP hidden layer size (for complexity>=1)
            random_seed: Reproducibility
        """
        if complexity < 0 or complexity > 3:
            raise ValueError(f"complexity must be 0-3, got {complexity}")

        self.complexity = complexity
        self.hidden_size = hidden_size
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names_ = None
        self.n_features_in_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        """
        Fit linear probe on activations and targets.

        Args:
            X: [n_samples, n_features] activations
            y: [n_samples] or [n_samples, n_targets] target values

        Returns:
            self
        """
        # Store shape info
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = None

        # Handle 1D y (scalar) and 2D y (multiple targets)
        y = np.asarray(y)
        is_multioutput = y.ndim > 1

        if is_multioutput:
            n_targets = y.shape[1]
        else:
            n_targets = 1

        X_scaled = self.scaler.fit_transform(X)

        if self.complexity == 0:
            # Ridge regression (fast and stable)
            self.model = Ridge(alpha=1.0, fit_intercept=True)
            self.model.fit(X_scaled, y)

        else:
            # PyTorch MLP — set seeds for reproducibility
            torch.manual_seed(self.random_seed)
            self.model = self._build_mlp(n_targets)
            self.model = self.model.float()

            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y)
            if not is_multioutput:
                y_tensor = y_tensor.unsqueeze(-1)  # [n] → [n, 1] for MSELoss

            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            self.model.train()
            for epoch in range(100):  # 100 epochs: fast + sufficient
                optimizer.zero_grad()
                outputs = self.model(X_tensor)  # [n, 1] or [n, n_targets]
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()

            # Move to CPU
            self.model = self.model.cpu()

        return self

    def _build_mlp(self, n_targets: int) -> nn.Module:
        """Build MLP model based on complexity.

        complexity=1: input → hidden(+ReLU) → output  (1 hidden layer)
        complexity=2: input → hidden(+ReLU) → hidden(+ReLU) → output
        complexity=3: input → hidden(+ReLU) → hidden(+ReLU) → hidden(+ReLU) → output
        """
        layers: list[nn.Module] = []

        # First hidden layer (always present for complexity >= 1)
        layers.append(nn.Linear(self.n_features_in_, self.hidden_size))
        layers.append(nn.ReLU())

        # Additional hidden layers (complexity - 1 more)
        for _ in range(self.complexity - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(self.hidden_size, n_targets))

        return nn.Sequential(*layers)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict targets from activations.

        Args:
            X: [n_samples, n_features] activations

        Returns:
            y_pred: [n_samples] or [n_samples, n_targets] predictions
        """
        X_scaled = self.scaler.transform(X)

        if self.complexity == 0:
            y_pred = np.asarray(self.model.predict(X_scaled))
            if y_pred.ndim == 2 and y_pred.shape[1] == 1:
                y_pred = y_pred.ravel()
        else:
            X_tensor = torch.FloatTensor(X_scaled)
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_tensor).squeeze(-1).numpy()

        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate probe on test data.

        Args:
            X: [n_samples, n_features] activations
            y: [n_samples] or [n_samples, n_targets] target values

        Returns:
            dict with 'r2' and 'mse'
        """
        y_pred = self.predict(X)

        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        return {"r2": r2, "mse": mse}


def probe_layer(
    activations: np.ndarray,
    targets: np.ndarray,
    complexities: list[int] = [0, 1, 2, 3],
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Dict[int, Dict[str, float]]:
    """
    Probe a single layer's activations for target values.

    Args:
        activations: [n_samples, n_features] activations (already flattened)
        targets: [n_samples] target values
        complexities: List of probe complexities to try
        test_size: Test split ratio
        random_seed: Reproducibility

    Returns:
        {complexity: {'r2': float, 'mse': float}} dictionary
    """
    # Train/test split (random_state ensures reproducibility without global seed mutation)
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
        scores = probe.score(X_test, y_test)
        results[complexity] = scores

    return results


def probe_all_layers(
    activations_per_layer: list[np.ndarray],
    targets: np.ndarray,
    complexities: list[int] = [0, 1, 2, 3],
    flatten_fn: Optional[Callable] = None,
    random_seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Probe all layers' activations for target values.

    Args:
        activations_per_layer: List of layer activations
            Each element: [batch, seq_len, feature_blocks+1, emsize]
            or already flattened [n_samples, n_features]
        targets: [n_samples] target values
        complexities: List of probe complexities to try
        flatten_fn: Function to convert activations to [n_samples, n_features]
            None = use default flatten
        random_seed: Reproducibility

    Returns:
        {
            'r2': np.ndarray of shape [n_layers, n_complexities],
            'mse': np.ndarray of shape [n_layers, n_complexities],
            'complexities': list[int],
            'n_layers': int
        }
    """
    # Default flatten function
    if flatten_fn is None:
        flatten_fn = lambda x: x.reshape(x.shape[0], -1)

    # Flatten all layer activations
    flattened_activations = [flatten_fn(layer) for layer in activations_per_layer]

    n_samples = flattened_activations[0].shape[0]
    n_layers = len(flattened_activations)
    n_features = flattened_activations[0].shape[1]
    n_complexities = len(complexities)

    # Pre-allocate results
    r2_results = np.zeros((n_layers, n_complexities))
    mse_results = np.zeros((n_layers, n_complexities))

    # Compute R² for each layer independently (same random split each time)
    for layer_idx, layer_features in enumerate(flattened_activations):
        X_layer_train, X_layer_test, y_layer_train, y_layer_test = train_test_split(
            layer_features,
            targets,
            test_size=0.2,
            random_state=random_seed,
            shuffle=True,
        )

        for comp_idx, complexity in enumerate(complexities):
            probe = LinearProbe(complexity=complexity, random_seed=random_seed)
            probe.fit(X_layer_train, y_layer_train)
            scores = probe.score(X_layer_test, y_layer_test)

            r2_results[layer_idx, comp_idx] = scores["r2"]
            mse_results[layer_idx, comp_idx] = scores["mse"]

    return {
        "r2": r2_results,
        "mse": mse_results,
        "complexities": complexities,
        "n_layers": n_layers,
        "n_features": n_features,
    }


# Verification script
if __name__ == "__main__":
    import numpy as np
    import time

    rng = np.random.default_rng(42)

    # Perfect linear data: expect R²=1.0
    X = np.random.randn(200, 10)
    y = 2 * X[:, 0] + 3 * X[:, 1]  # Perfect linear relationship

    print("=" * 60)
    print("LinearProbe Verification")
    print("=" * 60)

    # Single probe test
    print("\n1. Single LinearProbe test (complexity=0):")
    start = time.time()
    probe = LinearProbe(complexity=0)
    probe.fit(X[:160], y[:160])
    scores = probe.score(X[160:], y[160:])
    elapsed = time.time() - start
    print(f"   Linear probe R²: {scores['r2']:.4f} (expected ~1.0)")
    print(f"   Time: {elapsed:.4f}s")

    if scores['r2'] > 0.95:
        print("   ✓ R² is high (good)")
    else:
        print(f"   ✗ R² too low: {scores['r2']}")
    assert scores['r2'] > 0.95, f"R² too low: {scores['r2']}"

    # Complexity comparison test
    print("\n2. Complexity comparison test:")
    start = time.time()
    results = probe_layer(X, y, complexities=[0, 1, 2, 3])
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.4f}s")
    print("   Complexity vs R²:")
    for c, r in results.items():
        print(f"     complexity={c}: R²={r['r2']:.4f}, MSE={r['mse']:.6f}")

    if results[0]['r2'] > results[1]['r2'] > results[2]['r2'] > results[3]['r2']:
        print("   ✓ Complexity monotonic decrease (linear encoding)")
    else:
        print("   ! Complexity trend not perfectly monotonic")

    # All layers test (dummy layers)
    print("\n3. All layers test (12 dummy layers):")
    start = time.time()
    fake_layers = [np.random.randn(200, 5, 3, 192) for _ in range(12)]
    fake_targets = np.random.randn(200)
    all_results = probe_all_layers(fake_layers, fake_targets)
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.4f}s")
    print(f"   All layers R² shape: {all_results['r2'].shape}")
    print("   Expected shape: (12, 4)")

    assert all_results['r2'].shape == (12, 4), f"Shape mismatch: {all_results['r2'].shape} != (12, 4)"
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
