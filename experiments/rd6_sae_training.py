# pyright: reportMissingImports=false
"""RD-6 M10-T2: Sparse Autoencoder Training + Basic Analysis.

Trains sparse autoencoders on TabPFN Layer 6 activations collected from
diverse linear regression datasets (varying α, β). Compares expansion
factors (4× and 8×) and evaluates reconstruction quality and sparsity.

Methodology:
    1. Generate 50 diverse datasets with α, β ∈ [0.5, 5.0].
    2. Collect Layer 6 label-token activations from all datasets.
    3. Train SAE with expansion_factor=4 (768 hidden) and 8 (1536 hidden).
    4. Evaluate: reconstruction R², sparsity, training convergence.

Reference:
    - Bricken et al. "Towards Monosemanticity" (Anthropic, 2023)
    - Chanin et al. "A is for Absorption" arXiv:2409.14507
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rd5_config import cfg

from src.sae.sparse_autoencoder import (  # noqa: E402
    SAETrainer,
    TabPFNSparseAutoencoder,
    generate_diverse_datasets,
)

QUICK_RUN = True
RANDOM_SEED = 42

# Activation collection
COLLECTION_LAYER = 6
TOKEN_IDX = -1  # label token
N_DATASETS = 50 if not QUICK_RUN else 20

# SAE configurations to compare
SAE_CONFIGS = [
    {"expansion_factor": 4, "label": "4× (768 hidden)"},
    {"expansion_factor": 8, "label": "8× (1536 hidden)"},
]

# Training parameters
EPOCHS = 100 if not QUICK_RUN else 30
BATCH_SIZE = 64
LR = 1e-3
L1_COEFF = 1e-3


def _set_global_seed(seed: int) -> None:
    """Set reproducibility seed for NumPy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_model() -> TabPFNRegressor:
    """Construct CPU-only TabPFN regressor."""
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _collect_activations(datasets: list[dict[str, Any]]) -> np.ndarray:
    """Collect layer activations using SAETrainer helper API."""
    model = _build_model()
    temp_sae = TabPFNSparseAutoencoder(input_dim=192, expansion_factor=4)
    temp_trainer = SAETrainer(temp_sae, lr=LR, l1_coeff=L1_COEFF)
    activations = temp_trainer.collect_activations(
        model,
        datasets,
        layer=COLLECTION_LAYER,
        token_idx=TOKEN_IDX,
    )
    print(f"  Collected activations: {activations.shape}")
    return activations


def _activation_stats(activations: np.ndarray) -> dict[str, float | int | list[int]]:
    """Compute summary statistics for collected activation matrix."""
    return {
        "shape": [int(activations.shape[0]), int(activations.shape[1])],
        "n_samples": int(activations.shape[0]),
        "input_dim": int(activations.shape[1]),
        "mean": float(np.mean(activations)),
        "std": float(np.std(activations)),
        "min": float(np.min(activations)),
        "max": float(np.max(activations)),
        "l2_mean": float(np.linalg.norm(activations, axis=1).mean()),
    }


def _train_sae_config(
    activations: np.ndarray,
    config: dict[str, Any],
    results_dir: Path,
) -> dict[str, Any]:
    """Train one SAE configuration and save its state dict."""
    expansion_factor = int(config["expansion_factor"])
    label = str(config["label"])

    torch.manual_seed(RANDOM_SEED)
    sae = TabPFNSparseAutoencoder(input_dim=192, expansion_factor=expansion_factor)
    trainer = SAETrainer(sae, lr=LR, l1_coeff=L1_COEFF)

    print(
        f"\n  Training config: {label} | "
        f"epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, l1={L1_COEFF}"
    )
    history = trainer.train(activations, epochs=EPOCHS, batch_size=BATCH_SIZE)

    ckpt_path = results_dir / f"sae_{expansion_factor}x.pt"
    torch.save(sae.state_dict(), ckpt_path)

    sparsity_zero_frac = [float(v) for v in history["sparsity"]]
    sparsity_active_frac = [float(1.0 - v) for v in sparsity_zero_frac]

    final_mse = float(history["mse_loss"][-1])
    final_r2 = float(history["reconstruction_r2"][-1])
    final_sparsity = float(sparsity_active_frac[-1])

    return {
        "expansion_factor": expansion_factor,
        "label": label,
        "hidden_dim": int(192 * expansion_factor),
        "model_path": str(ckpt_path),
        "history": {
            "total_loss": [float(v) for v in history["total_loss"]],
            "mse_loss": [float(v) for v in history["mse_loss"]],
            "l1_loss": [float(v) for v in history["l1_loss"]],
            "sparsity_zero_fraction": sparsity_zero_frac,
            "sparsity_active_fraction": sparsity_active_frac,
            "reconstruction_r2": [float(v) for v in history["reconstruction_r2"]],
        },
        "final_metrics": {
            "reconstruction_r2": final_r2,
            "sparsity": final_sparsity,
            "sparsity_zero_fraction": float(sparsity_zero_frac[-1]),
            "mse_loss": final_mse,
        },
        "success_criteria": {
            "r2_gt_0_9": bool(final_r2 > 0.9),
            "sparsity_lt_0_2": bool(final_sparsity < 0.2),
        },
    }


def _plot_training_curves(results: list[dict[str, Any]], save_path: Path) -> None:
    """Plot MSE, reconstruction R², and sparsity curves for all configs."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for item in results:
        label = str(item["label"])
        history = item["history"]
        epochs = np.arange(1, len(history["mse_loss"]) + 1)

        axes[0].plot(epochs, history["mse_loss"], linewidth=2.0, label=label)
        axes[1].plot(epochs, history["reconstruction_r2"], linewidth=2.0, label=label)
        axes[2].plot(
            epochs,
            history["sparsity_active_fraction"],
            linewidth=2.0,
            label=label,
        )

    axes[0].set_title("(a) MSE Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("(b) Reconstruction R² vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Reconstruction R²")
    axes[1].axhline(0.9, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_title("(c) Sparsity vs Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Sparsity (fraction active features)")
    axes[2].axhline(0.2, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_summary_table(results: list[dict[str, Any]]) -> None:
    """Print compact summary table for final metrics."""
    print("\nSummary Table")
    print("Config       | Final R² | Final Sparsity | Final MSE")
    print("-------------|----------|----------------|----------")

    for item in results:
        ef = int(item["expansion_factor"])
        hidden = int(item["hidden_dim"])
        metrics = item["final_metrics"]
        print(
            f"{ef}× ({hidden})".ljust(13)
            + "| "
            + f"{metrics['reconstruction_r2']:.4f}".ljust(9)
            + "| "
            + f"{metrics['sparsity']:.3f}".ljust(15)
            + "| "
            + f"{metrics['mse_loss']:.6f}"
        )

    print("\nSuccess Criteria")
    print("  - Reconstruction R² > 0.9")
    print("  - Sparsity < 20% (fraction of active features)")
    for item in results:
        metrics = item["final_metrics"]
        criteria = item["success_criteria"]
        label = item["label"]
        r2_status = "PASS" if criteria["r2_gt_0_9"] else "FAIL"
        sparse_status = "PASS" if criteria["sparsity_lt_0_2"] else "FAIL"
        print(
            f"  - {label}: R²={metrics['reconstruction_r2']:.4f} [{r2_status}], "
            f"sparsity={metrics['sparsity']:.3f} [{sparse_status}]"
        )


def _save_results_json(
    results: list[dict[str, Any]],
    activation_stats: dict[str, float | int | list[int]],
    output_path: Path,
) -> None:
    """Write experiment outputs and metadata to JSON file."""
    payload: dict[str, Any] = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "collection": {
            "layer": COLLECTION_LAYER,
            "token_idx": TOKEN_IDX,
            "n_datasets": N_DATASETS,
            "activation_stats": activation_stats,
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "l1_coeff": L1_COEFF,
        },
        "configs": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    """Run RD-6 SAE training and analysis experiment."""
    _set_global_seed(RANDOM_SEED)

    results_dir = ROOT / "results" / "rd6" / "training"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RD-6 M10-T2: Sparse Autoencoder Training + Basic Analysis")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}")
    print(
        f"Collection: layer={COLLECTION_LAYER}, token_idx={TOKEN_IDX}, "
        f"n_datasets={N_DATASETS}"
    )
    print(
        f"Training: epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
        f"lr={LR}, l1_coeff={L1_COEFF}"
    )

    print("\n[1/7] Generating diverse datasets ...")
    datasets = generate_diverse_datasets(
        n_datasets=N_DATASETS,
        random_seed=RANDOM_SEED,
    )
    print(f"  Generated datasets: {len(datasets)}")

    print("\n[2/7] Building TabPFN and collecting layer activations ...")
    activations = _collect_activations(datasets)
    act_stats = _activation_stats(activations)
    print(
        "  Activation stats: "
        f"mean={act_stats['mean']:.4f}, std={act_stats['std']:.4f}, "
        f"min={act_stats['min']:.4f}, max={act_stats['max']:.4f}"
    )

    print("\n[3/7] Training SAE configurations ...")
    config_results: list[dict[str, Any]] = []
    for config in SAE_CONFIGS:
        result = _train_sae_config(activations, config, results_dir)
        config_results.append(result)
        print(
            f"  Final ({result['label']}): "
            f"R²={result['final_metrics']['reconstruction_r2']:.4f}, "
            f"sparsity={result['final_metrics']['sparsity']:.3f}, "
            f"MSE={result['final_metrics']['mse_loss']:.6f}"
        )

    print("\n[4/7] SAE checkpoints saved during training loop.")
    print("  - results/rd6/training/sae_4x.pt")
    print("  - results/rd6/training/sae_8x.pt")

    print("\n[5/7] Plotting training curves ...")
    plot_path = results_dir / "training_curves.png"
    _plot_training_curves(config_results, plot_path)
    print(f"  Saved plot: {plot_path}")

    print("\n[6/7] Printing summary table ...")
    _print_summary_table(config_results)

    print("\n[7/7] Saving results JSON ...")
    json_path = results_dir / "results.json"
    _save_results_json(config_results, act_stats, json_path)
    print(f"  Saved results: {json_path}")

    print(f"\nDone. All outputs in: {results_dir}")


if __name__ == "__main__":
    main()
