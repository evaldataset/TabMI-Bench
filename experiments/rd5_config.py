"""Shared configuration for RD-5 cross-model experiments.

Controls experiment scale and reproducibility via environment variables:
    QUICK_RUN=0|1  (default: 1)
    SEED=<int>     (default: 42)
    N_TRAIN=<int>  (default: 100 for full, 50 for quick)
    N_TEST=<int>   (default: 50 for full, 10 for quick)
    DEVICE=cpu|cuda|cuda:0|...  (default: cpu)

Usage in experiment scripts:
    from rd5_config import cfg
    # Access: cfg.QUICK_RUN, cfg.SEED, cfg.N_TRAIN, cfg.N_TEST, cfg.DEVICE
    # Also: cfg.dataset_count("tabpfn"), cfg.layer_indices("iltm")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RD5Config:
    """Immutable experiment configuration."""

    QUICK_RUN: bool
    SEED: int
    N_TRAIN: int
    N_TEST: int
    DEVICE: str

    # Full-scale parameters
    FULL_DATASET_COUNTS: dict[str, int] = field(
        default_factory=lambda: {"tabpfn": 20, "tabicl": 20, "iltm": 10}
    )
    QUICK_DATASET_COUNTS: dict[str, int] = field(
        default_factory=lambda: {"tabpfn": 5, "tabicl": 5, "iltm": 3}
    )
    FULL_COEF_VALUES: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    QUICK_COEF_VALUES: list[int] = field(default_factory=lambda: [1, 3, 5])

    # SAE parameters
    SAE_EPOCHS_FULL: int = 200
    SAE_EPOCHS_QUICK: int = 50
    SAE_BATCH_SIZE: int = 64
    SAE_LR: float = 1e-3
    EXPANSION_FACTOR: int = 4

    # Multi-seed list
    ALL_SEEDS: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    def dataset_count(self, model_name: str) -> int:
        """Number of datasets to use for a given model."""
        counts = (
            self.QUICK_DATASET_COUNTS if self.QUICK_RUN else self.FULL_DATASET_COUNTS
        )
        return counts.get(model_name, 5)

    def coef_values(self) -> list[int]:
        """Coefficient values for probing experiments."""
        return self.QUICK_COEF_VALUES if self.QUICK_RUN else self.FULL_COEF_VALUES

    def coef_pairs(self, model_name: str) -> list[tuple[int, int]]:
        """Generate (alpha, beta) pairs for coefficient probing."""
        vals = self.coef_values()
        if self.QUICK_RUN and model_name == "iltm":
            return [(a, 3) for a in [1, 3, 5]]
        return [(a, b) for a in vals for b in vals]

    def layer_indices(self, model_name: str) -> list[int]:
        """Layer indices for a given model."""
        if model_name == "iltm":
            return [0, 1, 2]
        if model_name == "tabdpt":
            return list(range(16))
        return list(range(12))

    @property
    def sae_epochs(self) -> int:
        return self.SAE_EPOCHS_QUICK if self.QUICK_RUN else self.SAE_EPOCHS_FULL

    @property
    def results_base(self) -> str:
        """Results subdirectory suffix based on mode."""
        return "quick" if self.QUICK_RUN else "full"


def _load_config() -> RD5Config:
    """Load config from environment variables with sensible defaults."""
    quick_run = os.environ.get("QUICK_RUN", "1").strip() in ("1", "true", "True", "yes")
    seed = int(os.environ.get("SEED", "42"))
    device = os.environ.get("DEVICE", "cpu").strip()

    if quick_run:
        n_train = int(os.environ.get("N_TRAIN", "50"))
        n_test = int(os.environ.get("N_TEST", "10"))
    else:
        n_train = int(os.environ.get("N_TRAIN", "100"))
        n_test = int(os.environ.get("N_TEST", "50"))

    return RD5Config(
        QUICK_RUN=quick_run,
        SEED=seed,
        N_TRAIN=n_train,
        N_TEST=n_test,
        DEVICE=device,
    )


# Module-level singleton — import as `from rd5_config import cfg`
cfg = _load_config()
