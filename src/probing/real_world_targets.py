"""Probing target definitions for real-world tabular datasets."""

# pyright: reportMissingImports=false

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbingTarget:
    """Single probing target vector with metadata."""

    name: str
    values: np.ndarray
    description: str


def compute_prediction_targets(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> list[ProbingTarget]:
    """Compute target vectors from predictions and ground truth."""
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"Mismatched lengths: y_pred={y_pred.shape[0]}, y_true={y_true.shape[0]}"
        )

    error = y_pred - y_true
    abs_error = np.abs(error)

    return [
        ProbingTarget(
            name="prediction_error",
            values=error,
            description="Signed prediction error: y_pred - y_true",
        ),
        ProbingTarget(
            name="absolute_error",
            values=abs_error,
            description="Absolute prediction error: |y_pred - y_true|",
        ),
        ProbingTarget(
            name="y_true",
            values=y_true,
            description="Ground-truth target values",
        ),
        ProbingTarget(
            name="y_pred",
            values=y_pred,
            description="Model prediction values",
        ),
    ]


def compute_feature_targets(
    features: np.ndarray,
    feature_names: list[str],
) -> list[ProbingTarget]:
    """Create one probing target per input feature column."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")

    if len(feature_names) != features.shape[1]:
        raise ValueError(
            "feature_names length "
            f"({len(feature_names)}) must match feature columns ({features.shape[1]})"
        )

    targets: list[ProbingTarget] = []
    for idx, feature_name in enumerate(feature_names):
        targets.append(
            ProbingTarget(
                name=f"feature_{feature_name}",
                values=features[:, idx].copy(),
                description=f"Raw values for feature '{feature_name}'",
            )
        )

    return targets


def compute_distribution_targets(y: np.ndarray) -> list[ProbingTarget]:
    """Compute distribution-derived probing targets from y."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.shape[0] == 0:
        raise ValueError("y must contain at least one value")

    scaler = StandardScaler()
    y_zscore = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)

    order = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort")
    rank_percentile = order.astype(np.float64)
    if y.shape[0] > 1:
        rank_percentile /= y.shape[0] - 1

    median = np.median(y)
    above_median = (y >= median).astype(np.float64)

    return [
        ProbingTarget(
            name="y_zscore",
            values=y_zscore,
            description="Standardized target z-score",
        ),
        ProbingTarget(
            name="y_rank_percentile",
            values=rank_percentile,
            description="Rank percentile in [0, 1]",
        ),
        ProbingTarget(
            name="y_above_median",
            values=above_median,
            description="Binary indicator for y >= median",
        ),
    ]


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    features = rng.normal(size=(20, 3))
    y_true = rng.normal(size=20)
    y_pred = y_true + rng.normal(scale=0.1, size=20)

    prediction_targets = compute_prediction_targets(y_pred=y_pred, y_true=y_true)
    assert len(prediction_targets) == 4
    assert prediction_targets[0].values.shape == (20,)

    feature_targets = compute_feature_targets(
        features=features,
        feature_names=["f1", "f2", "f3"],
    )
    assert len(feature_targets) == 3
    assert feature_targets[2].name == "feature_f3"

    distribution_targets = compute_distribution_targets(y_true)
    assert len(distribution_targets) == 3
    assert distribution_targets[1].values.min() >= 0.0
    assert distribution_targets[1].values.max() <= 1.0

    print("All real-world probing target utilities: OK")
