"""Synthetic data generators for TabPFN mechanistic interpretability experiments."""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class LinearDataset:
    """Dataset for z = alpha*x + beta*y experiments."""

    X_train: np.ndarray  # [n_train, 2]
    y_train: np.ndarray  # [n_train]
    X_test: np.ndarray  # [n_test, 2]
    y_test: np.ndarray  # [n_test]
    alpha: float
    beta: float


@dataclass
class QuadraticDataset:
    """Dataset for z = a*b + c experiments."""

    X_train: np.ndarray  # [n_train, 3] columns: a, b, c
    y_train: np.ndarray  # [n_train] = a*b + c
    X_test: np.ndarray  # [n_test, 3]
    y_test: np.ndarray  # [n_test]
    # Intermediate value: a*b (for probing)
    intermediary_train: np.ndarray  # [n_train] = a*b
    intermediary_test: np.ndarray  # [n_test] = a*b


def generate_linear_data(
    alpha: float,
    beta: float,
    n_train: int = 50,
    n_test: int = 10,
    noise_sigma: float = 0.0,
    random_seed: int = 42,
) -> LinearDataset:
    """
    Generate z = alpha*x + beta*y dataset.

    Args:
        alpha: Coefficient for x
        beta: Coefficient for y
        n_train: Number of training samples
        n_test: Number of test samples
        noise_sigma: Gaussian noise std (0 = noiseless)
        random_seed: Reproducibility

    Returns:
        LinearDataset with X=[x,y], y=alpha*x+beta*y
    """
    rng = np.random.default_rng(random_seed)

    # Generate features: x ~ Uniform(-1, 1), y ~ Uniform(-1, 1)
    X_train = rng.uniform(-1.0, 1.0, size=(n_train, 2))
    X_test = rng.uniform(-1.0, 1.0, size=(n_test, 2))

    # Generate targets: z = alpha*x + beta*y + noise
    y_train = alpha * X_train[:, 0] + beta * X_train[:, 1]
    y_test = alpha * X_test[:, 0] + beta * X_test[:, 1]

    # Add noise if requested
    if noise_sigma > 0:
        y_train += rng.normal(0.0, noise_sigma, size=n_train)
        y_test += rng.normal(0.0, noise_sigma, size=n_test)

    return LinearDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        alpha=alpha,
        beta=beta,
    )


def generate_quadratic_data(
    a_range: tuple[float, float] = (0.5, 3.0),
    b_range: tuple[float, float] = (0.5, 3.0),
    c_range: tuple[float, float] = (0.5, 3.0),
    n_train: int = 50,
    n_test: int = 10,
    noise_sigma: float = 0.0,
    random_seed: int = 42,
) -> QuadraticDataset:
    """
    Generate z = a*b + c dataset.

    Args:
        a_range, b_range, c_range: Uniform sampling ranges for a, b, c
        n_train: Number of training samples
        n_test: Number of test samples
        noise_sigma: Gaussian noise std
        random_seed: Reproducibility

    Returns:
        QuadraticDataset with X=[a,b,c], y=a*b+c, intermediary=a*b
    """
    rng = np.random.default_rng(random_seed)

    # Generate features
    a_train = rng.uniform(a_range[0], a_range[1], size=n_train)
    b_train = rng.uniform(b_range[0], b_range[1], size=n_train)
    c_train = rng.uniform(c_range[0], c_range[1], size=n_train)
    X_train = np.column_stack([a_train, b_train, c_train])

    a_test = rng.uniform(a_range[0], a_range[1], size=n_test)
    b_test = rng.uniform(b_range[0], b_range[1], size=n_test)
    c_test = rng.uniform(c_range[0], c_range[1], size=n_test)
    X_test = np.column_stack([a_test, b_test, c_test])

    # Compute intermediary: a*b
    intermediary_train = a_train * b_train
    intermediary_test = a_test * b_test

    # Compute target: a*b + c
    y_train = intermediary_train + c_train
    y_test = intermediary_test + c_test

    # Add noise if requested
    if noise_sigma > 0:
        y_train += rng.normal(0.0, noise_sigma, size=n_train)
        y_test += rng.normal(0.0, noise_sigma, size=n_test)

    return QuadraticDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        intermediary_train=intermediary_train,
        intermediary_test=intermediary_test,
    )


def generate_multiple_linear_fits(
    n_fits: int = 100,
    alpha_range: tuple[float, float] = (0.5, 5.0),
    beta_range: tuple[float, float] = (0.5, 5.0),
    n_train: int = 50,
    n_test: int = 10,
    noise_sigma: float = 0.0,
    random_seed: int = 42,
) -> list[LinearDataset]:
    """
    Generate multiple linear datasets with different (alpha, beta) pairs.
    Used for Configuration 1 (Multiple Fits, Pooled) probing.

    Args:
        n_fits: Number of different (alpha, beta) pairs
        alpha_range: Uniform range for alpha sampling
        beta_range: Uniform range for beta sampling
        n_train: Number of training samples per fit
        n_test: Number of test samples per fit
        noise_sigma: Gaussian noise std
        random_seed: Reproducibility

    Returns:
        List of n_fits LinearDataset objects
    """
    rng = np.random.default_rng(random_seed)
    datasets = []

    for fit_id in range(n_fits):
        alpha = rng.uniform(alpha_range[0], alpha_range[1])
        beta = rng.uniform(beta_range[0], beta_range[1])

        ds = generate_linear_data(
            alpha=alpha,
            beta=beta,
            n_train=n_train,
            n_test=n_test,
            noise_sigma=noise_sigma,
            random_seed=random_seed + fit_id,  # Ensure different seeds
        )
        datasets.append(ds)

    return datasets


def generate_switch_variable_data(
    n_coefficient_pairs: int = 20,
    alpha_range: tuple[float, float] = (0.5, 5.0),
    beta_range: tuple[float, float] = (0.5, 5.0),
    n_samples_per_pair: int = 50,
    noise_sigma: float = 0.0,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate switch variable dataset for Configuration 2 probing.

    Creates a single dataset where a switch variable u identifies each
    (alpha, beta) pair. Used for single-fit probing experiments.

    Formula: z = alpha(u) * x + beta(u) * y

    Returns:
        X: [n_total, 3] — columns: [x, y, u] where u is integer switch variable
        y: [n_total] — target values
        alphas: [n_total] — true alpha for each sample
        betas: [n_total] — true beta for each sample
    """
    rng = np.random.default_rng(random_seed)

    # Generate coefficients for each pair
    alphas = np.zeros(n_coefficient_pairs)
    betas = np.zeros(n_coefficient_pairs)
    for i in range(n_coefficient_pairs):
        alphas[i] = rng.uniform(alpha_range[0], alpha_range[1])
        betas[i] = rng.uniform(beta_range[0], beta_range[1])

    # Generate samples for each pair
    n_total = n_coefficient_pairs * n_samples_per_pair
    X = np.zeros((n_total, 3))  # [x, y, u]
    y = np.zeros(n_total)
    true_alphas = np.zeros(n_total)
    true_betas = np.zeros(n_total)

    for u_idx in range(n_coefficient_pairs):
        n_samples = n_samples_per_pair

        # Generate x, y for this pair
        x = rng.uniform(-1.0, 1.0, size=n_samples)
        y_feat = rng.uniform(-1.0, 1.0, size=n_samples)

        # Compute targets
        z = alphas[u_idx] * x + betas[u_idx] * y_feat

        # Add noise if requested
        if noise_sigma > 0:
            z += rng.normal(0.0, noise_sigma, size=n_samples)

        # Fill arrays
        start_idx = u_idx * n_samples
        end_idx = start_idx + n_samples
        X[start_idx:end_idx, 0] = x
        X[start_idx:end_idx, 1] = y_feat
        X[start_idx:end_idx, 2] = u_idx  # Switch variable
        y[start_idx:end_idx] = z
        true_alphas[start_idx:end_idx] = alphas[u_idx]
        true_betas[start_idx:end_idx] = betas[u_idx]

    return X, y, true_alphas, true_betas


@dataclass
class NonLinearDataset:
    """Dataset for non-linear function experiments with intermediary tracking."""

    X_train: np.ndarray  # [n_train, 3] columns: a, b, c
    y_train: np.ndarray  # [n_train]
    X_test: np.ndarray  # [n_test, 3]
    y_test: np.ndarray  # [n_test]
    intermediary_train: np.ndarray  # [n_train] — non-linear intermediary
    intermediary_test: np.ndarray  # [n_test]
    func_type: str  # "sinusoidal", "polynomial", or "mixed"


def generate_nonlinear_data(
    func_type: str = "sinusoidal",
    a_range: tuple[float, float] = (0.5, 3.0),
    b_range: tuple[float, float] = (0.5, 3.0),
    c_range: tuple[float, float] = (0.5, 3.0),
    n_train: int = 50,
    n_test: int = 10,
    noise_sigma: float = 0.0,
    random_seed: int = 42,
) -> NonLinearDataset:
    """Generate non-linear function datasets with tracked intermediaries.

    func_type options:
        "sinusoidal": z = sin(a*b) + c,       intermediary = sin(a*b)
        "polynomial": z = a^2 + b*c,           intermediary = a^2
        "mixed":      z = sin(a*b) + a*b*c,    intermediary = sin(a*b)
    """
    rng = np.random.default_rng(random_seed)

    a_train = rng.uniform(a_range[0], a_range[1], size=n_train)
    b_train = rng.uniform(b_range[0], b_range[1], size=n_train)
    c_train = rng.uniform(c_range[0], c_range[1], size=n_train)
    X_train = np.column_stack([a_train, b_train, c_train])

    a_test = rng.uniform(a_range[0], a_range[1], size=n_test)
    b_test = rng.uniform(b_range[0], b_range[1], size=n_test)
    c_test = rng.uniform(c_range[0], c_range[1], size=n_test)
    X_test = np.column_stack([a_test, b_test, c_test])

    if func_type == "sinusoidal":
        intermediary_train = np.sin(a_train * b_train)
        intermediary_test = np.sin(a_test * b_test)
        y_train = intermediary_train + c_train
        y_test = intermediary_test + c_test
    elif func_type == "polynomial":
        intermediary_train = a_train**2
        intermediary_test = a_test**2
        y_train = intermediary_train + b_train * c_train
        y_test = intermediary_test + b_test * c_test
    elif func_type == "mixed":
        intermediary_train = np.sin(a_train * b_train)
        intermediary_test = np.sin(a_test * b_test)
        y_train = intermediary_train + a_train * b_train * c_train
        y_test = intermediary_test + a_test * b_test * c_test
    else:
        raise ValueError(f"Unknown func_type: {func_type!r}")

    if noise_sigma > 0:
        y_train += rng.normal(0.0, noise_sigma, size=n_train)
        y_test += rng.normal(0.0, noise_sigma, size=n_test)

    return NonLinearDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        intermediary_train=intermediary_train,
        intermediary_test=intermediary_test,
        func_type=func_type,
    )


def generate_multifeature_data(
    n_features: int = 4,
    n_train: int = 50,
    n_test: int = 10,
    noise_sigma: float = 0.0,
    random_seed: int = 42,
) -> NonLinearDataset:
    """Generate multi-feature data: z = sum(w_i * x_i) + sin(x_0 * x_1).

    The intermediary is sin(x_0 * x_1) — a non-linear interaction term.
    Linear weights w_i are drawn from Uniform(0.5, 3.0).
    """
    rng = np.random.default_rng(random_seed)
    weights = rng.uniform(0.5, 3.0, size=n_features)

    X_train = rng.uniform(0.1, 3.0, size=(n_train, n_features))
    X_test = rng.uniform(0.1, 3.0, size=(n_test, n_features))

    intermediary_train = np.sin(X_train[:, 0] * X_train[:, 1])
    intermediary_test = np.sin(X_test[:, 0] * X_test[:, 1])

    y_train = X_train @ weights + intermediary_train
    y_test = X_test @ weights + intermediary_test

    if noise_sigma > 0:
        y_train += rng.normal(0.0, noise_sigma, size=n_train)
        y_test += rng.normal(0.0, noise_sigma, size=n_test)

    return NonLinearDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        intermediary_train=intermediary_train,
        intermediary_test=intermediary_test,
        func_type=f"multifeature_d{n_features}",
    )


def generate_semi_synthetic_data(
    func_type: str = "nonlinear",
    noise_sigma: float = 0.5,
    missing_rate: float = 0.0,
    n_features: int = 5,
    n_train: int = 100,
    n_test: int = 20,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate semi-synthetic data for RD-4 Phase 4A experiments.

    func_type options:
        "nonlinear": z = sin(a*b) + c^2
        "noisy_linear": z = a*b + c + noise
        "polynomial": z = a^2 + b*c
        "mixed": combination of above

    Args:
        func_type: Type of function to generate
        noise_sigma: Gaussian noise std
        missing_rate: Fraction of values to set as NaN (0.0 = no missing)
        n_features: Number of features (>= 3 for the formulas)
        n_train: Training samples
        n_test: Test samples
        random_seed: Reproducibility

    Returns:
        X_train, y_train, X_test, y_test
        (NaN values present if missing_rate > 0)
    """
    rng = np.random.default_rng(random_seed)

    if func_type == "nonlinear":

        def target_func(X):
            return np.sin(X[:, 0] * X[:, 1]) + X[:, 2] ** 2

    elif func_type == "noisy_linear":

        def target_func(X):
            return X[:, 0] * X[:, 1] + X[:, 2] + rng.normal(0, noise_sigma, size=len(X))

    elif func_type == "polynomial":

        def target_func(X):
            return X[:, 0] ** 2 + X[:, 1] * X[:, 2]

    elif func_type == "mixed":

        def target_func(X):
            return (
                np.sin(X[:, 0] * X[:, 1]) + X[:, 2] ** 2 + X[:, 0] * X[:, 1] * X[:, 2]
            )

    else:
        raise ValueError(f"Unknown func_type: {func_type}")

    # Generate features
    X_train = rng.uniform(0.1, 3.0, size=(n_train, n_features))
    X_test = rng.uniform(0.1, 3.0, size=(n_test, n_features))

    # Generate targets
    y_train = target_func(X_train)
    y_test = target_func(X_test)

    # Add missing values if requested
    if missing_rate > 0.0:
        # Create copy to avoid modifying originals
        X_train_nan = X_train.copy()
        X_test_nan = X_test.copy()

        # Set random values to NaN
        train_nan_count = int(n_train * n_features * missing_rate)
        train_nan_indices = rng.choice(
            n_train * n_features, size=train_nan_count, replace=False
        )
        train_nan_rows = train_nan_indices // n_features
        train_nan_cols = train_nan_indices % n_features
        X_train_nan[train_nan_rows, train_nan_cols] = np.nan

        test_nan_count = int(n_test * n_features * missing_rate)
        test_nan_indices = rng.choice(
            n_test * n_features, size=test_nan_count, replace=False
        )
        test_nan_rows = test_nan_indices // n_features
        test_nan_cols = test_nan_indices % n_features
        X_test_nan[test_nan_rows, test_nan_cols] = np.nan

        return X_train_nan, y_train, X_test_nan, y_test

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Test generate_linear_data
    ds = generate_linear_data(alpha=2.0, beta=3.0, n_train=50, n_test=10)
    assert ds.X_train.shape == (50, 2)
    assert ds.y_train.shape == (50,)
    assert abs(ds.alpha - 2.0) < 1e-9
    # Verify formula: y ≈ alpha*x + beta*y_col
    y_check = 2.0 * ds.X_train[:, 0] + 3.0 * ds.X_train[:, 1]
    assert np.allclose(ds.y_train, y_check), "Linear formula mismatch"
    print("generate_linear_data: OK")

    # Test generate_quadratic_data
    ds2 = generate_quadratic_data(n_train=50, n_test=10)
    assert ds2.X_train.shape == (50, 3)
    assert ds2.intermediary_train.shape == (50,)
    # Verify: intermediary = a*b
    a, b, c = ds2.X_train[:, 0], ds2.X_train[:, 1], ds2.X_train[:, 2]
    assert np.allclose(ds2.intermediary_train, a * b), "Intermediary mismatch"
    assert np.allclose(ds2.y_train, a * b + c), "Quadratic formula mismatch"
    print("generate_quadratic_data: OK")

    # Test generate_multiple_linear_fits
    fits = generate_multiple_linear_fits(n_fits=10, n_train=30, n_test=5)
    assert len(fits) == 10
    assert fits[0].X_train.shape == (30, 2)
    # Each fit has different alpha/beta
    alphas = [f.alpha for f in fits]
    assert len(set(alphas)) > 1, "All alphas identical - bad RNG"
    print("generate_multiple_linear_fits: OK")

    # Test generate_switch_variable_data
    X, y, alphas, betas = generate_switch_variable_data(
        n_coefficient_pairs=5, n_samples_per_pair=20
    )
    assert X.shape == (100, 3)  # 5 pairs * 20 samples
    assert y.shape == (100,)
    assert alphas.shape == (100,)
    print("generate_switch_variable_data: OK")

    # Test generate_semi_synthetic_data
    X_tr, y_tr, X_te, y_te = generate_semi_synthetic_data(
        func_type="nonlinear", noise_sigma=0.1, missing_rate=0.1, n_features=5
    )
    assert X_tr.shape[1] == 5
    nan_count = np.isnan(X_tr).sum()
    assert nan_count > 0, "Expected some NaN values"
    print(f"generate_semi_synthetic_data: OK (NaN count: {nan_count})")

    print("ALL CHECKS PASSED")
