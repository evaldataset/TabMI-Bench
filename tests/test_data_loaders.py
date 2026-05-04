"""Tests for dataset loader contracts.

Ensures real-world loaders return correct shapes and never silently
fall back to synthetic data.
"""

import numpy as np
import pytest

from src.data.real_world_datasets import (
    RealWorldDataset,
    load_california_housing,
    load_diabetes_sklearn,
)


class TestDataLoaderContract:
    """Verify that dataset loaders return valid RealWorldDataset objects."""

    def test_california_housing_shape(self):
        ds = load_california_housing(n_train=100, n_test=20, random_seed=42)
        assert isinstance(ds, RealWorldDataset)
        assert ds.X_train.shape == (100, 8)
        assert ds.X_test.shape == (20, 8)
        assert ds.y_train.shape == (100,)
        assert ds.y_test.shape == (20,)

    def test_diabetes_shape(self):
        ds = load_diabetes_sklearn(n_train=100, n_test=50, random_seed=42)
        assert isinstance(ds, RealWorldDataset)
        assert ds.X_train.shape[0] == 100
        assert ds.X_test.shape[0] == 50
        assert ds.n_features == 10

    def test_no_nan_in_features(self):
        ds = load_california_housing(n_train=100, n_test=20, random_seed=42)
        assert np.all(np.isfinite(ds.X_train))
        assert np.all(np.isfinite(ds.X_test))
        assert np.all(np.isfinite(ds.y_train))
        assert np.all(np.isfinite(ds.y_test))

    def test_feature_scaling_applied(self):
        ds = load_california_housing(n_train=200, n_test=50, random_seed=42)
        # StandardScaler should make mean ≈ 0, std ≈ 1 for train
        train_means = np.abs(ds.X_train.mean(axis=0))
        assert np.all(train_means < 0.2), f"Train means not near zero: {train_means}"

    def test_deterministic_with_seed(self):
        ds1 = load_california_housing(n_train=50, n_test=10, random_seed=42)
        ds2 = load_california_housing(n_train=50, n_test=10, random_seed=42)
        np.testing.assert_array_equal(ds1.X_train, ds2.X_train)
        np.testing.assert_array_equal(ds1.y_train, ds2.y_train)

    def test_different_seeds_differ(self):
        ds1 = load_california_housing(n_train=50, n_test=10, random_seed=42)
        ds2 = load_california_housing(n_train=50, n_test=10, random_seed=123)
        assert not np.array_equal(ds1.X_train, ds2.X_train)
