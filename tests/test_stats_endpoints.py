"""Tests for statistical analysis endpoint alignment with paper.

Ensures the primary endpoint (profile variance) is computed and
included in the statistical comparisons, matching the paper's
pre-specified analysis plan.
"""

import pytest


class TestStatsEndpoints:
    """Verify statistical_analysis.py matches paper's endpoint structure."""

    def test_profile_variance_is_primary_endpoint(self):
        """The primary endpoint must be profile variance, not peak R²."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from experiments.statistical_analysis import load_cross_model_values

        try:
            data = load_cross_model_values()
        except Exception:
            pytest.skip("Result files not available")

        assert "profile_variance" in data, (
            "profile_variance must be a key in cross-model comparisons "
            "(paper primary endpoint)"
        )
        title = data["profile_variance"][4]
        assert "PRIMARY" in title.upper(), (
            f"Profile variance should be labeled as primary, got: {title}"
        )

    def test_intermediary_is_sensitivity_analysis(self):
        """Peak-layer intermediary R² should be labeled as sensitivity, not primary."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from experiments.statistical_analysis import load_cross_model_values

        try:
            data = load_cross_model_values()
        except Exception:
            pytest.skip("Result files not available")

        assert "intermediary" in data
        title = data["intermediary"][4]
        assert "sensitivity" in title.lower(), (
            f"Peak-layer R² should be labeled as sensitivity analysis, got: {title}"
        )
