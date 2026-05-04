"""Real-world dataset loaders for TabPFN interpretability experiments."""

# pyright: reportMissingImports=false

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_breast_cancer as sklearn_load_breast_cancer,
    load_diabetes,
    load_iris as sklearn_load_iris,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class RealWorldDataset:
    """Container for a preprocessed real-world dataset split."""

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    task_type: str  # 'regression' or 'classification'
    n_features: int
    n_train: int
    n_test: int
    feature_names: list[str]


def _prepare_regression_dataset(
    name: str,
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
    n_train: int,
    n_test: int,
    random_seed: int,
    task_type: str = "regression",
) -> RealWorldDataset:
    """Create train/test split and apply feature scaling for regression."""
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive")

    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    total_requested = n_train + n_test
    if features.shape[0] < total_requested:
        raise ValueError(
            "Not enough samples for "
            f"{name}: requested {total_requested}, got {features.shape[0]}"
        )

    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(features.shape[0])[:total_requested]
    x_subset = features[indices]
    y_subset = targets[indices]

    x_train: np.ndarray = np.asarray(x_subset[:n_train], dtype=np.float64)
    x_test: np.ndarray = np.asarray(x_subset[n_train:], dtype=np.float64)
    y_train: np.ndarray = np.asarray(y_subset[:n_train], dtype=np.float64)
    y_test: np.ndarray = np.asarray(y_subset[n_train:], dtype=np.float64)

    scaler = StandardScaler()
    x_train = np.asarray(scaler.fit_transform(x_train), dtype=np.float64)
    x_test = np.asarray(scaler.transform(x_test), dtype=np.float64)

    return RealWorldDataset(
        name=name,
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        task_type=task_type,
        n_features=x_train.shape[1],
        n_train=x_train.shape[0],
        n_test=x_test.shape[0],
        feature_names=feature_names,
    )


def _coerce_features_to_float(
    features: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    if isinstance(features, pd.DataFrame):
        df = features.copy()
    else:
        arr = np.asarray(features)
        if arr.ndim != 2:
            raise ValueError(f"features must be 2D, got shape={arr.shape}")
        df = pd.DataFrame(arr)

    for col in df.columns:
        col_series = df[col]
        if pd.api.types.is_numeric_dtype(col_series):
            df[col] = pd.to_numeric(col_series, errors="coerce")
        else:
            df[col] = col_series.astype("category").cat.codes.astype(np.float64)

    x = df.to_numpy(dtype=np.float64)
    feature_names = [str(col) for col in df.columns]
    return x, feature_names


def _coerce_targets_to_float(targets: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(targets, pd.Series):
        series = targets.copy()
    else:
        series = pd.Series(np.asarray(targets).reshape(-1))

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    return pd.to_numeric(series.astype(str), errors="coerce").to_numpy(dtype=np.float64)


def _filter_valid_rows(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have same number of rows")

    valid_mask = np.isfinite(targets) & np.isfinite(features).all(axis=1)
    x = features[valid_mask]
    y = targets[valid_mask]
    return x, y, feature_names


def _split_openml_xy(
    raw_x: pd.DataFrame | np.ndarray,
    raw_y: pd.Series | np.ndarray | None,
) -> tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray]:
    if raw_y is not None:
        return raw_x, raw_y

    if isinstance(raw_x, pd.DataFrame) and raw_x.shape[1] >= 2:
        return raw_x.iloc[:, :-1], raw_x.iloc[:, -1]

    raise ValueError("OpenML target is missing and cannot be inferred")


def _generate_wine_quality_fallback(
    n_samples: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate offline synthetic wine-like regression data."""
    rng = np.random.default_rng(random_seed)
    feature_names = [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]

    features = np.zeros((n_samples, len(feature_names)), dtype=np.float64)
    features[:, 0] = rng.normal(7.5, 1.2, size=n_samples)
    features[:, 1] = rng.normal(0.5, 0.18, size=n_samples)
    features[:, 2] = rng.normal(0.3, 0.2, size=n_samples)
    features[:, 3] = rng.normal(6.0, 4.0, size=n_samples)
    features[:, 4] = rng.normal(0.06, 0.02, size=n_samples)
    features[:, 5] = rng.normal(30.0, 12.0, size=n_samples)
    features[:, 6] = rng.normal(115.0, 40.0, size=n_samples)
    features[:, 7] = rng.normal(0.996, 0.0025, size=n_samples)
    features[:, 8] = rng.normal(3.2, 0.2, size=n_samples)
    features[:, 9] = rng.normal(0.65, 0.2, size=n_samples)
    features[:, 10] = rng.normal(10.5, 1.1, size=n_samples)
    features = np.clip(features, a_min=0.0, a_max=None)

    y = (
        0.55 * features[:, 10]
        - 1.6 * features[:, 1]
        + 0.9 * features[:, 2]
        - 18.0 * features[:, 4]
        + 0.25 * features[:, 9]
        + 3.0 * (features[:, 8] - 3.0)
    )
    y += rng.normal(0.0, 0.8, size=n_samples)

    return features, y, feature_names


def load_california_housing(
    n_train: int = 1000,
    n_test: int = 200,
    random_seed: int = 42,
) -> RealWorldDataset:
    """Load California Housing dataset (regression, 8 features, 20640 samples)."""
    features, targets = fetch_california_housing(return_X_y=True)
    feature_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    return _prepare_regression_dataset(
        name="california_housing",
        features=features,
        targets=targets,
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
    )


def load_wine_quality(
    n_train: int = 1000,
    n_test: int = 200,
    random_seed: int = 42,
) -> RealWorldDataset:
    """Load Wine Quality dataset (regression, 11 features, 6497 samples)."""
    total_requested = n_train + n_test

    local_candidates = [
        Path("data/winequality-red.csv"),
        Path("data/winequality-white.csv"),
        Path("data/winequality.csv"),
        Path("datasets/winequality-red.csv"),
        Path("datasets/winequality-white.csv"),
        Path("datasets/winequality.csv"),
    ]

    features: np.ndarray = np.empty((0, 0), dtype=np.float64)
    targets: np.ndarray = np.empty(0, dtype=np.float64)
    feature_names: list[str] = []
    feature_names: list[str]

    loaded = False
    for path in local_candidates:
        if not path.exists():
            continue

        df = pd.read_csv(path, sep=None, engine="python")
        if "quality" not in df.columns:
            continue

        targets = df["quality"].to_numpy(dtype=np.float64)
        features = df.drop(columns=["quality"]).to_numpy(dtype=np.float64)
        feature_names = [str(col) for col in df.columns if col != "quality"]
        loaded = True
        break

    if not loaded:
        raise FileNotFoundError(
            "Wine Quality dataset not found locally. Download from UCI ML "
            "Repository and place CSV in data/ or datasets/ directory."
        )

    return _prepare_regression_dataset(
        name="wine_quality",
        features=features,
        targets=targets,
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
    )


def load_diabetes_sklearn(
    n_train: int = 300,
    n_test: int = 100,
    random_seed: int = 42,
) -> RealWorldDataset:
    """Load sklearn Diabetes dataset (regression, 10 features, 442 samples)."""
    features, targets = load_diabetes(return_X_y=True)
    feature_names = [
        "age",
        "sex",
        "bmi",
        "bp",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
    ]
    return _prepare_regression_dataset(
        name="diabetes_sklearn",
        features=np.asarray(features, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float64),
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
    )


def load_boston(
    n_train: int = 500,
    n_test: int = 100,
    random_seed: int = 42,
) -> RealWorldDataset:
    raw_x, raw_y = fetch_openml(data_id=531, as_frame=True, return_X_y=True)
    raw_x, raw_y = _split_openml_xy(raw_x, raw_y)
    features, feature_names = _coerce_features_to_float(raw_x)
    targets = _coerce_targets_to_float(raw_y)
    features, targets, feature_names = _filter_valid_rows(
        features, targets, feature_names
    )

    return _prepare_regression_dataset(
        name="boston",
        features=features,
        targets=targets,
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
    )


def load_abalone(
    n_train: int = 500,
    n_test: int = 100,
    random_seed: int = 42,
) -> RealWorldDataset:
    raw_x, raw_y = fetch_openml(data_id=183, as_frame=True, return_X_y=True)
    raw_x, raw_y = _split_openml_xy(raw_x, raw_y)
    features, feature_names = _coerce_features_to_float(raw_x)
    targets = _coerce_targets_to_float(raw_y)
    features, targets, feature_names = _filter_valid_rows(
        features, targets, feature_names
    )

    return _prepare_regression_dataset(
        name="abalone",
        features=features,
        targets=targets,
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
    )


def load_bike_sharing(
    n_train: int = 500,
    n_test: int = 100,
    random_seed: int = 42,
) -> RealWorldDataset:
    raw_x, raw_y = fetch_openml(data_id=44063, as_frame=True, return_X_y=True)
    raw_x, raw_y = _split_openml_xy(raw_x, raw_y)
    features, feature_names = _coerce_features_to_float(raw_x)
    targets = _coerce_targets_to_float(raw_y)
    features, targets, feature_names = _filter_valid_rows(
        features, targets, feature_names
    )

    return _prepare_regression_dataset(
        name="bike_sharing",
        features=features,
        targets=targets,
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
    )


def load_energy_efficiency(
    n_train: int = 500,
    n_test: int = 100,
    random_seed: int = 42,
) -> RealWorldDataset:
    raw_x, raw_y = fetch_openml(data_id=242, as_frame=True, return_X_y=True)
    raw_x, raw_y = _split_openml_xy(raw_x, raw_y)
    features, feature_names = _coerce_features_to_float(raw_x)
    targets = _coerce_targets_to_float(raw_y)
    features, targets, feature_names = _filter_valid_rows(
        features, targets, feature_names
    )

    return _prepare_regression_dataset(
        name="energy_efficiency",
        features=features,
        targets=targets,
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
    )


def load_concrete(
    n_train: int = 500,
    n_test: int = 100,
    random_seed: int = 42,
) -> RealWorldDataset:
    raw_x, raw_y = fetch_openml(data_id=4353, as_frame=True, return_X_y=True)
    raw_x, raw_y = _split_openml_xy(raw_x, raw_y)
    features, feature_names = _coerce_features_to_float(raw_x)
    targets = _coerce_targets_to_float(raw_y)
    features, targets, feature_names = _filter_valid_rows(
        features, targets, feature_names
    )

    return _prepare_regression_dataset(
        name="concrete",
        features=features,
        targets=targets,
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
    )


def load_breast_cancer(
    n_train: int = 500,
    n_test: int = 100,
    random_seed: int = 42,
) -> RealWorldDataset:
    raw = sklearn_load_breast_cancer()
    features = np.asarray(raw.data, dtype=np.float64)
    targets = np.asarray(raw.target, dtype=np.float64)
    feature_names = [str(name) for name in raw.feature_names]

    return _prepare_regression_dataset(
        name="breast_cancer",
        features=features,
        targets=targets,
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
        task_type="classification",
    )


def load_iris_binary(
    n_train: int = 500,
    n_test: int = 100,
    random_seed: int = 42,
) -> RealWorldDataset:
    raw = sklearn_load_iris()
    features = np.asarray(raw.data, dtype=np.float64)
    targets = np.asarray(raw.target, dtype=np.int64)

    binary_mask = np.isin(targets, [0, 1])
    features = features[binary_mask]
    targets_binary = targets[binary_mask].astype(np.float64)
    feature_names = [str(name) for name in raw.feature_names]

    return _prepare_regression_dataset(
        name="iris_binary",
        features=features,
        targets=targets_binary,
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
        task_type="classification",
    )


def load_adult_income(
    n_train: int = 500,
    n_test: int = 100,
    random_seed: int = 42,
) -> RealWorldDataset:
    raw_x, raw_y = fetch_openml(data_id=1590, as_frame=True, return_X_y=True)
    raw_x, raw_y = _split_openml_xy(raw_x, raw_y)
    features, feature_names = _coerce_features_to_float(raw_x)

    target_series = (
        pd.Series(raw_y).astype(str).str.strip().str.replace(".", "", regex=False)
    )
    unique_values = sorted(target_series.unique().tolist())
    if len(unique_values) != 2:
        raise ValueError(
            f"adult_income expected binary labels, got {len(unique_values)} values"
        )
    label_map = {unique_values[0]: 0.0, unique_values[1]: 1.0}
    targets = target_series.map(label_map).to_numpy(dtype=np.float64)
    features, targets, feature_names = _filter_valid_rows(
        features, targets, feature_names
    )

    return _prepare_regression_dataset(
        name="adult_income",
        features=features,
        targets=targets,
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
        task_type="classification",
    )


def load_satellite(
    n_train: int = 500,
    n_test: int = 100,
    random_seed: int = 42,
) -> RealWorldDataset:
    raw_x, raw_y = fetch_openml(data_id=40, as_frame=True, return_X_y=True)
    raw_x, raw_y = _split_openml_xy(raw_x, raw_y)
    features, feature_names = _coerce_features_to_float(raw_x)
    targets = _coerce_targets_to_float(raw_y)
    if not np.isfinite(targets).any():
        targets = (
            pd.Series(raw_y).astype("category").cat.codes.to_numpy(dtype=np.float64)
        )
    features, targets, feature_names = _filter_valid_rows(
        features, targets, feature_names
    )

    return _prepare_regression_dataset(
        name="satellite",
        features=features,
        targets=targets,
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
    )


def load_bank_marketing(
    n_train: int = 500,
    n_test: int = 100,
    random_seed: int = 42,
) -> RealWorldDataset:
    raw_x, raw_y = fetch_openml(data_id=1461, as_frame=True, return_X_y=True)
    raw_x, raw_y = _split_openml_xy(raw_x, raw_y)
    features, feature_names = _coerce_features_to_float(raw_x)

    target_series = pd.Series(raw_y).astype(str).str.strip()
    unique_values = sorted(target_series.unique().tolist())
    if len(unique_values) != 2:
        raise ValueError(
            f"bank_marketing expected binary labels, got {len(unique_values)} values"
        )
    label_map = {unique_values[0]: 0.0, unique_values[1]: 1.0}
    targets = target_series.map(label_map).to_numpy(dtype=np.float64)
    features, targets, feature_names = _filter_valid_rows(
        features, targets, feature_names
    )

    return _prepare_regression_dataset(
        name="bank_marketing",
        features=features,
        targets=targets,
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
        task_type="classification",
    )


def load_credit_g(
    n_train: int = 500,
    n_test: int = 100,
    random_seed: int = 42,
) -> RealWorldDataset:
    raw_x, raw_y = fetch_openml(data_id=31, as_frame=True, return_X_y=True)
    raw_x, raw_y = _split_openml_xy(raw_x, raw_y)
    features, feature_names = _coerce_features_to_float(raw_x)

    target_series = pd.Series(raw_y).astype(str).str.strip()
    unique_values = sorted(target_series.unique().tolist())
    if len(unique_values) != 2:
        raise ValueError(
            f"credit_g expected binary labels, got {len(unique_values)} values"
        )
    label_map = {unique_values[0]: 0.0, unique_values[1]: 1.0}
    targets = target_series.map(label_map).to_numpy(dtype=np.float64)
    features, targets, feature_names = _filter_valid_rows(
        features, targets, feature_names
    )

    return _prepare_regression_dataset(
        name="credit_g",
        features=features,
        targets=targets,
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
        task_type="classification",
    )


def load_segment(
    n_train: int = 500,
    n_test: int = 100,
    random_seed: int = 42,
) -> RealWorldDataset:
    raw_x, raw_y = fetch_openml(data_id=36, as_frame=True, return_X_y=True)
    raw_x, raw_y = _split_openml_xy(raw_x, raw_y)
    features, feature_names = _coerce_features_to_float(raw_x)
    targets = _coerce_targets_to_float(raw_y)
    if not np.isfinite(targets).any():
        targets = (
            pd.Series(raw_y).astype("category").cat.codes.to_numpy(dtype=np.float64)
        )
    features, targets, feature_names = _filter_valid_rows(
        features, targets, feature_names
    )

    return _prepare_regression_dataset(
        name="segment",
        features=features,
        targets=targets,
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
    )


def load_vehicle(
    n_train: int = 500,
    n_test: int = 100,
    random_seed: int = 42,
) -> RealWorldDataset:
    raw_x, raw_y = fetch_openml(data_id=54, as_frame=True, return_X_y=True)
    raw_x, raw_y = _split_openml_xy(raw_x, raw_y)
    features, feature_names = _coerce_features_to_float(raw_x)
    targets = _coerce_targets_to_float(raw_y)
    if not np.isfinite(targets).any():
        targets = (
            pd.Series(raw_y).astype("category").cat.codes.to_numpy(dtype=np.float64)
        )
    features, targets, feature_names = _filter_valid_rows(
        features, targets, feature_names
    )

    return _prepare_regression_dataset(
        name="vehicle",
        features=features,
        targets=targets,
        feature_names=feature_names,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
    )


def get_available_datasets(
    n_train: int = 500,
    n_test: int = 100,
    random_seed: int = 42,
) -> tuple[list[RealWorldDataset], list[dict[str, str]]]:
    """Load all available datasets, tracking any that are skipped.

    Returns:
        (datasets, skipped) where skipped is a list of
        {"name": str, "reason": str} dicts for transparency.
    """
    import warnings

    datasets: list[RealWorldDataset] = []
    skipped: list[dict[str, str]] = []

    datasets.append(
        load_california_housing(
            n_train=n_train,
            n_test=n_test,
            random_seed=random_seed,
        )
    )
    datasets.append(
        load_diabetes_sklearn(
            n_train=min(n_train, 300),
            n_test=min(n_test, 100),
            random_seed=random_seed,
        )
    )

    try:
        datasets.append(
            load_wine_quality(
                n_train=n_train,
                n_test=n_test,
                random_seed=random_seed,
            )
        )
    except Exception as e:
        skipped.append({"name": "wine_quality", "reason": str(e)})
        warnings.warn(f"Skipped wine_quality: {e}", stacklevel=2)

    optional_loaders = [
        load_boston,
        load_abalone,
        load_bike_sharing,
        load_energy_efficiency,
        load_concrete,
        load_breast_cancer,
        load_iris_binary,
        load_adult_income,
        load_satellite,
        load_bank_marketing,
        load_credit_g,
        load_segment,
        load_vehicle,
    ]

    for loader in optional_loaders:
        max_train = n_train
        max_test = n_test
        if loader is load_iris_binary:
            max_train = min(max_train, 80)
            max_test = min(max_test, 20)

        try:
            datasets.append(
                loader(
                    n_train=max_train,
                    n_test=max_test,
                    random_seed=random_seed,
                )
            )
        except Exception as e:
            loader_name = getattr(loader, "__name__", str(loader))
            skipped.append({"name": loader_name, "reason": str(e)})
            warnings.warn(f"Skipped {loader_name}: {e}", stacklevel=2)

    return datasets, skipped


if __name__ == "__main__":
    california = load_california_housing()
    assert california.X_train.shape == (1000, 8)
    assert california.y_train.shape == (1000,)
    assert california.X_test.shape == (200, 8)

    diabetes = load_diabetes_sklearn()
    assert diabetes.X_train.shape == (300, 10)
    assert diabetes.y_test.shape == (100,)

    wine = load_wine_quality()
    assert wine.X_train.shape[1] == 11
    assert wine.y_train.shape == (1000,)

    breast_cancer = load_breast_cancer()
    assert breast_cancer.X_train.shape[1] == 30
    assert set(np.unique(breast_cancer.y_train)).issubset({0.0, 1.0})

    print("All real-world dataset loaders: OK")
