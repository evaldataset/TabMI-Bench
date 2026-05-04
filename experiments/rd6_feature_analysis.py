# pyright: reportMissingImports=false
"""RD-6 M10-T3: SAE Feature Correlation Analysis.

Analyzes which SAE features correlate with known data-generating parameters
(α, β) and intermediate computations (a·b = α·x₁·x₂). Tests whether SAE
decomposes polysemantic neurons into interpretable monosemantic features.

Methodology:
    1. Train an SAE on Layer 6 activations (reuse training pipeline).
    2. For each dataset (with known α, β), encode its activations.
    3. Compute Pearson correlation between each SAE feature's mean
       activation and the dataset's α, β, and a·b values.
    4. Identify top-K features most correlated with each parameter.

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
from scipy.stats import pearsonr
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rd5_config import cfg

from src.hooks.tabpfn_hooker import TabPFNHookedModel  # noqa: E402
from src.sae.sparse_autoencoder import (  # noqa: E402
    SAETrainer,
    TabPFNSparseAutoencoder,
    generate_diverse_datasets,
)

# ── Global constants ───────────────────────────────────────────────────
QUICK_RUN = True
RANDOM_SEED = 42

COLLECTION_LAYER = 6
TOKEN_IDX = -1
N_DATASETS = 50 if not QUICK_RUN else 20
EXPANSION_FACTOR = 4  # Use 4× for analysis (768 features)

EPOCHS = 100 if not QUICK_RUN else 30
BATCH_SIZE = 64
LR = 1e-3
L1_COEFF = 1e-3

TOP_K = 10  # Top K features to highlight


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson r safely, returning 0.0 for degenerate cases."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if x_arr.shape != y_arr.shape:
        raise ValueError(f"Shape mismatch for Pearson: {x_arr.shape} vs {y_arr.shape}")
    if x_arr.size < 2:
        return 0.0
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return 0.0

    r, _ = pearsonr(x_arr, y_arr)
    if not np.isfinite(r):
        return 0.0
    return float(r)


def _dataset_mean_ab(dataset: dict[str, Any]) -> float:
    x_train = np.asarray(dataset["X_train"], dtype=np.float32)
    alpha = float(dataset["alpha"])
    beta = float(dataset["beta"])
    ab_values = x_train[:, 0] * x_train[:, 1] * alpha * beta
    return float(np.mean(ab_values))


def _extract_per_dataset_feature_means(
    model: TabPFNRegressor,
    sae: TabPFNSparseAutoencoder,
    datasets: list[dict[str, Any]],
    layer: int,
    token_idx: int,
) -> np.ndarray:
    per_dataset_means: list[np.ndarray] = []

    sae.eval()
    with torch.no_grad():
        for ds_idx, ds in enumerate(datasets):
            model.fit(ds["X_train"], ds["y_train"])
            hooker = TabPFNHookedModel(model)
            _preds, cache = hooker.forward_with_cache(ds["X_test"])

            single_eval_pos = int(cache["single_eval_pos"])
            layer_act = cache["layers"][layer]
            train_label_act = (
                layer_act[0, :single_eval_pos, token_idx, :].detach().cpu().numpy()
            )

            train_label_act = np.asarray(train_label_act, dtype=np.float32)
            encoded = sae.encode(torch.from_numpy(train_label_act).float())
            feat_mean = encoded.mean(dim=0).detach().cpu().numpy()
            per_dataset_means.append(feat_mean.astype(np.float32, copy=False))

            if (ds_idx + 1) % 5 == 0 or ds_idx == len(datasets) - 1:
                print(f"    Encoded {ds_idx + 1:>2d}/{len(datasets)} datasets")

    return np.stack(per_dataset_means, axis=0)


def _compute_feature_correlations(
    feature_means: np.ndarray,
    alpha_values: np.ndarray,
    beta_values: np.ndarray,
    ab_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-feature Pearson correlation with α, β, and mean(a·b)."""
    hidden_dim = int(feature_means.shape[1])
    corr_alpha = np.zeros(hidden_dim, dtype=np.float32)
    corr_beta = np.zeros(hidden_dim, dtype=np.float32)
    corr_ab = np.zeros(hidden_dim, dtype=np.float32)

    for feature_idx in range(hidden_dim):
        feat_signal = feature_means[:, feature_idx]
        corr_alpha[feature_idx] = _safe_pearson(feat_signal, alpha_values)
        corr_beta[feature_idx] = _safe_pearson(feat_signal, beta_values)
        corr_ab[feature_idx] = _safe_pearson(feat_signal, ab_values)

    return corr_alpha, corr_beta, corr_ab


def _build_top_feature_table(corr: np.ndarray, top_k: int) -> list[dict[str, float]]:
    """Build top-k table sorted by absolute correlation magnitude."""
    ranked_idx = np.argsort(np.abs(corr))[::-1][:top_k]
    table: list[dict[str, float]] = []
    for idx in ranked_idx:
        table.append(
            {
                "feature_idx": int(idx),
                "r": float(corr[idx]),
                "abs_r": float(abs(corr[idx])),
            }
        )
    return table


def _collect_strong_features(
    corr_alpha: np.ndarray,
    corr_beta: np.ndarray,
    threshold: float = 0.7,
) -> list[dict[str, float]]:
    """Return features whose |r| exceeds threshold for α or β."""
    strong_mask = (np.abs(corr_alpha) > threshold) | (np.abs(corr_beta) > threshold)
    strong_idx = np.where(strong_mask)[0]

    rows: list[dict[str, float]] = []
    for idx in strong_idx:
        rows.append(
            {
                "feature_idx": int(idx),
                "r_alpha": float(corr_alpha[idx]),
                "r_beta": float(corr_beta[idx]),
                "max_abs_r": float(max(abs(corr_alpha[idx]), abs(corr_beta[idx]))),
            }
        )

    rows.sort(key=lambda row: row["max_abs_r"], reverse=True)
    return rows


def _plot_results(
    corr_alpha: np.ndarray,
    corr_beta: np.ndarray,
    feature_means: np.ndarray,
    alpha_values: np.ndarray,
    beta_values: np.ndarray,
    save_path: Path,
) -> None:
    """Create three-panel figure: heatmap + two scatter plots."""
    corr_matrix = np.stack([corr_alpha, corr_beta], axis=1)
    top20_idx = np.argsort(np.max(np.abs(corr_matrix), axis=1))[::-1][:20]
    heatmap_data = corr_matrix[top20_idx]

    top_alpha_feature = int(np.argmax(np.abs(corr_alpha)))
    top_beta_feature = int(np.argmax(np.abs(corr_beta)))

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 5))

    im = ax0.imshow(
        heatmap_data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(["alpha", "beta"])
    ax0.set_yticks(np.arange(len(top20_idx), dtype=np.int32))
    ax0.set_yticklabels([f"f{idx}" for idx in top20_idx])
    ax0.set_title("Top-20 SAE Features: Correlation Heatmap", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", rotation=90)

    feat_alpha = feature_means[:, top_alpha_feature]
    ax1.scatter(alpha_values, feat_alpha, alpha=0.75, color="#1f77b4", s=35)
    alpha_line = np.linspace(float(alpha_values.min()), float(alpha_values.max()), 100)
    alpha_fit = np.polyfit(alpha_values, feat_alpha, deg=1)
    ax1.plot(alpha_line, alpha_fit[0] * alpha_line + alpha_fit[1], color="#d62728")
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("Mean feature activation")
    ax1.set_title(
        f"Top alpha feature f{top_alpha_feature} (r={corr_alpha[top_alpha_feature]:+.3f})",
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.25)

    feat_beta = feature_means[:, top_beta_feature]
    ax2.scatter(beta_values, feat_beta, alpha=0.75, color="#2ca02c", s=35)
    beta_line = np.linspace(float(beta_values.min()), float(beta_values.max()), 100)
    beta_fit = np.polyfit(beta_values, feat_beta, deg=1)
    ax2.plot(beta_line, beta_fit[0] * beta_line + beta_fit[1], color="#d62728")
    ax2.set_xlabel("beta")
    ax2.set_ylabel("Mean feature activation")
    ax2.set_title(
        f"Top beta feature f{top_beta_feature} (r={corr_beta[top_beta_feature]:+.3f})",
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.25)

    fig.suptitle("RD-6: SAE Feature Correlation Analysis", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run RD-6 SAE feature correlation analysis experiment."""
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    results_dir = ROOT / "results" / "rd6" / "features"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RD-6 M10-T3: SAE Feature Correlation Analysis")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}")
    print(f"N_DATASETS={N_DATASETS}, layer={COLLECTION_LAYER}, token_idx={TOKEN_IDX}")
    print(
        f"SAE expansion={EXPANSION_FACTOR}x, epochs={EPOCHS}, batch_size={BATCH_SIZE}"
    )

    print("\n[1/7] Generating diverse datasets and initializing TabPFN ...")
    datasets = generate_diverse_datasets(n_datasets=N_DATASETS, random_seed=RANDOM_SEED)
    model = TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")
    print(f"    Generated {len(datasets)} datasets")

    print("\n[2/7] Collecting activations and training SAE ...")
    sae = TabPFNSparseAutoencoder(input_dim=192, expansion_factor=EXPANSION_FACTOR)
    trainer = SAETrainer(sae, lr=LR, l1_coeff=L1_COEFF)
    activations = trainer.collect_activations(
        model,
        datasets,
        layer=COLLECTION_LAYER,
        token_idx=TOKEN_IDX,
    )
    print(f"    Collected activations shape: {activations.shape}")

    history = trainer.train(activations, epochs=EPOCHS, batch_size=BATCH_SIZE)
    final_mse = float(history["mse_loss"][-1])
    final_r2 = float(history["reconstruction_r2"][-1])
    final_sparsity = float(history["sparsity"][-1])
    print(
        f"    SAE train done: MSE={final_mse:.6f}, R2={final_r2:.4f}, sparsity={final_sparsity:.3f}"
    )

    print("\n[3/7] Encoding each dataset and aggregating mean feature activations ...")
    feature_means = _extract_per_dataset_feature_means(
        model=model,
        sae=sae,
        datasets=datasets,
        layer=COLLECTION_LAYER,
        token_idx=TOKEN_IDX,
    )
    print(f"    feature_means shape: {feature_means.shape}")

    print("\n[4/7] Computing feature correlations with alpha, beta, and mean(a·b) ...")
    alpha_values = np.array([float(ds["alpha"]) for ds in datasets], dtype=np.float32)
    beta_values = np.array([float(ds["beta"]) for ds in datasets], dtype=np.float32)
    ab_values = np.array([_dataset_mean_ab(ds) for ds in datasets], dtype=np.float32)

    corr_alpha, corr_beta, corr_ab = _compute_feature_correlations(
        feature_means=feature_means,
        alpha_values=alpha_values,
        beta_values=beta_values,
        ab_values=ab_values,
    )

    print("\n[5/7] Ranking top features and checking success criterion ...")
    top_alpha_features = _build_top_feature_table(corr_alpha, top_k=TOP_K)
    top_beta_features = _build_top_feature_table(corr_beta, top_k=TOP_K)
    top_ab_features = _build_top_feature_table(corr_ab, top_k=TOP_K)
    strong_features = _collect_strong_features(corr_alpha, corr_beta, threshold=0.7)
    success_criteria_met = len(strong_features) >= 2

    print(f"    Strong features (|r|>0.7 with alpha or beta): {len(strong_features)}")
    print(f"    Success criterion (>=2 strong features): {success_criteria_met}")

    print("\n[6/7] Plotting feature correlation panels ...")
    fig_path = results_dir / "feature_correlations.png"
    _plot_results(
        corr_alpha=corr_alpha,
        corr_beta=corr_beta,
        feature_means=feature_means,
        alpha_values=alpha_values,
        beta_values=beta_values,
        save_path=fig_path,
    )
    print(f"    Saved plot: {fig_path}")

    print("\n[7/7] Saving results JSON ...")
    payload: dict[str, Any] = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "collection_layer": COLLECTION_LAYER,
        "token_idx": TOKEN_IDX,
        "n_datasets": N_DATASETS,
        "sae_config": {
            "input_dim": sae.input_dim,
            "hidden_dim": sae.hidden_dim,
            "expansion_factor": EXPANSION_FACTOR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "l1_coeff": L1_COEFF,
        },
        "training": {
            "history": history,
            "final_mse_loss": final_mse,
            "final_reconstruction_r2": final_r2,
            "final_sparsity": final_sparsity,
        },
        "correlations": {
            "alpha": corr_alpha.tolist(),
            "beta": corr_beta.tolist(),
            "mean_ab": corr_ab.tolist(),
        },
        "top_alpha_features": top_alpha_features,
        "top_beta_features": top_beta_features,
        "top_mean_ab_features": top_ab_features,
        "strong_features_abs_r_gt_0_7": strong_features,
        "success_criteria": {
            "description": "At least 2 SAE features with |r| > 0.7 to alpha or beta",
            "num_strong_features": len(strong_features),
            "met": success_criteria_met,
        },
    }

    json_path = results_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"    Saved results: {json_path}")
    print(f"\nDone. Outputs written to: {results_dir}")


if __name__ == "__main__":
    main()
