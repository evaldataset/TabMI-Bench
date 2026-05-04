# pyright: reportMissingImports=false
"""Synthetic classification data generators for RD-7 experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClassificationDataset:
    """Dataset for classification experiments."""

    X_train: np.ndarray  # [N_train, n_features]
    y_train: np.ndarray  # [N_train] int labels {0,1} or {0,1,...,K-1}
    X_test: np.ndarray  # [N_test, n_features]
    y_test: np.ndarray  # [N_test] int labels
    n_classes: int
    description: str


def generate_linear_classification(
    alpha: float = 2.0,
    beta: float = 1.0,
    n_train: int = 100,
    n_test: int = 20,
    noise_sigma: float = 0.1,
    random_seed: int = 42,
) -> ClassificationDataset:
    """
    Generate binary classification with a linear decision boundary.

    Decision rule: y = int(alpha*x1 + beta*x2 > 0)
    Features x1, x2 ~ N(0, 1) with optional Gaussian noise on features.

    Args:
        alpha: Coefficient for x1 in decision boundary
        beta: Coefficient for x2 in decision boundary
        n_train: Number of training samples
        n_test: Number of test samples
        noise_sigma: Gaussian noise std added to feature values
        random_seed: Reproducibility

    Returns:
        ClassificationDataset with n_classes=2
    """
    rng = np.random.default_rng(random_seed)

    n_total = n_train + n_test

    # Generate features: x1, x2 ~ N(0, 1) + optional noise
    features = rng.standard_normal(size=(n_total, 2))
    if noise_sigma > 0:
        features = features + rng.normal(0.0, noise_sigma, size=(n_total, 2))

    # Decision boundary: y = int(alpha*x1 + beta*x2 > 0)
    y = (alpha * features[:, 0] + beta * features[:, 1] > 0).astype(np.int64)

    # Split into train/test
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Shuffle train set
    train_perm = rng.permutation(n_train)
    X_train = X_train[train_perm]
    y_train = y_train[train_perm]

    return ClassificationDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_classes=2,
        description=f"Linear boundary: {alpha}*x1 + {beta}*x2 > 0",
    )


def generate_xor_data(
    n_train: int = 100,
    n_test: int = 20,
    noise_sigma: float = 0.1,
    random_seed: int = 42,
) -> ClassificationDataset:
    """
    Generate XOR classification pattern.

    Four quadrants with alternating labels:
    y = int((x1 > 0) XOR (x2 > 0))
    Features x1, x2 ~ Uniform[-1, 1] with optional Gaussian noise.

    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        noise_sigma: Gaussian noise std added to feature values
        random_seed: Reproducibility

    Returns:
        ClassificationDataset with n_classes=2
    """
    rng = np.random.default_rng(random_seed)

    n_total = n_train + n_test

    # Generate features: x1, x2 ~ Uniform[-1, 1] + optional noise
    features = rng.uniform(-1.0, 1.0, size=(n_total, 2))
    if noise_sigma > 0:
        features = features + rng.normal(0.0, noise_sigma, size=(n_total, 2))

    # XOR pattern: y = int((x1 > 0) XOR (x2 > 0))
    y = ((features[:, 0] > 0) ^ (features[:, 1] > 0)).astype(np.int64)

    # Split into train/test
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Shuffle train set
    train_perm = rng.permutation(n_train)
    X_train = X_train[train_perm]
    y_train = y_train[train_perm]

    return ClassificationDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_classes=2,
        description="XOR pattern: (x1>0) XOR (x2>0)",
    )


def generate_circle_data(
    radius: float = 1.0,
    n_train: int = 100,
    n_test: int = 20,
    noise_sigma: float = 0.1,
    random_seed: int = 42,
) -> ClassificationDataset:
    """
    Generate binary classification with a circular decision boundary.

    Decision rule: y = int(x1² + x2² < radius²)
    Features x1, x2 ~ Uniform[-2, 2] with optional Gaussian noise.

    Args:
        radius: Radius of the circular decision boundary
        n_train: Number of training samples
        n_test: Number of test samples
        noise_sigma: Gaussian noise std added to feature values
        random_seed: Reproducibility

    Returns:
        ClassificationDataset with n_classes=2
    """
    rng = np.random.default_rng(random_seed)

    n_total = n_train + n_test

    # Generate features: x1, x2 ~ Uniform[-2, 2] + optional noise
    features = rng.uniform(-2.0, 2.0, size=(n_total, 2))
    if noise_sigma > 0:
        features = features + rng.normal(0.0, noise_sigma, size=(n_total, 2))

    # Circular boundary: y = int(x1² + x2² < radius²)
    y = (features[:, 0] ** 2 + features[:, 1] ** 2 < radius**2).astype(np.int64)

    # Split into train/test
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Shuffle train set
    train_perm = rng.permutation(n_train)
    X_train = X_train[train_perm]
    y_train = y_train[train_perm]

    return ClassificationDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_classes=2,
        description=f"Circle boundary: x1² + x2² < {radius}²",
    )


def generate_multiclass_gaussian(
    n_classes: int = 3,
    n_features: int = 2,
    n_train_per_class: int = 30,
    n_test_per_class: int = 10,
    random_seed: int = 42,
) -> ClassificationDataset:
    """
    Generate multiclass classification with Gaussian clusters.

    K Gaussian clusters, each with a random center (scaled from N(0,1)*2)
    and identity covariance. y = class index {0, ..., K-1}.

    Args:
        n_classes: Number of classes (K)
        n_features: Number of features per sample
        n_train_per_class: Training samples per class
        n_test_per_class: Test samples per class
        random_seed: Reproducibility

    Returns:
        ClassificationDataset with n_classes=n_classes
    """
    rng = np.random.default_rng(random_seed)

    # Generate random cluster centers: N(0, 1) * 2
    centers = rng.standard_normal(size=(n_classes, n_features)) * 2.0

    # Generate samples for each class
    X_train_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []
    X_test_parts: list[np.ndarray] = []
    y_test_parts: list[np.ndarray] = []

    for k in range(n_classes):
        # Train samples: center + N(0, I)
        X_tr = centers[k] + rng.standard_normal(size=(n_train_per_class, n_features))
        y_tr = np.full(n_train_per_class, k, dtype=np.int64)
        X_train_parts.append(X_tr)
        y_train_parts.append(y_tr)

        # Test samples: center + N(0, I)
        X_te = centers[k] + rng.standard_normal(size=(n_test_per_class, n_features))
        y_te = np.full(n_test_per_class, k, dtype=np.int64)
        X_test_parts.append(X_te)
        y_test_parts.append(y_te)

    # Concatenate
    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    X_test = np.concatenate(X_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)

    # Shuffle train set
    n_train = n_classes * n_train_per_class
    train_perm = rng.permutation(n_train)
    X_train = X_train[train_perm]
    y_train = y_train[train_perm]

    return ClassificationDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_classes=n_classes,
        description=f"{n_classes}-class Gaussian clusters in {n_features}D",
    )


if __name__ == "__main__":
    # Test generate_linear_classification
    ds = generate_linear_classification(alpha=2.0, beta=1.0, n_train=100, n_test=20)
    assert ds.X_train.shape == (100, 2)
    assert ds.y_train.shape == (100,)
    assert ds.X_test.shape == (20, 2)
    assert ds.n_classes == 2
    assert set(np.unique(ds.y_train)).issubset({0, 1})
    assert ds.y_train.dtype == np.int64
    print("generate_linear_classification: OK")

    # Test generate_xor_data
    ds2 = generate_xor_data(n_train=100, n_test=20)
    assert ds2.X_train.shape == (100, 2)
    assert ds2.n_classes == 2
    assert set(np.unique(ds2.y_train)).issubset({0, 1})
    assert ds2.y_train.dtype == np.int64
    print("generate_xor_data: OK")

    # Test generate_circle_data
    ds3 = generate_circle_data(radius=1.0, n_train=100, n_test=20)
    assert ds3.X_train.shape == (100, 2)
    assert ds3.n_classes == 2
    assert set(np.unique(ds3.y_train)).issubset({0, 1})
    assert ds3.y_train.dtype == np.int64
    print("generate_circle_data: OK")

    # Test generate_multiclass_gaussian
    ds4 = generate_multiclass_gaussian(
        n_classes=3, n_features=2, n_train_per_class=30, n_test_per_class=10
    )
    assert ds4.X_train.shape == (90, 2)  # 3 * 30
    assert ds4.y_train.shape == (90,)
    assert ds4.X_test.shape == (30, 2)  # 3 * 10
    assert ds4.n_classes == 3
    assert set(np.unique(ds4.y_train)) == {0, 1, 2}
    assert ds4.y_train.dtype == np.int64
    print("generate_multiclass_gaussian: OK")

    print("ALL CHECKS PASSED")
