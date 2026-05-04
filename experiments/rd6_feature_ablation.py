# pyright: reportMissingImports=false
"""RD-6 M10-T4: SAE Feature Ablation vs Probe Direction Ablation (RD-3).

Compares two approaches to information removal:
1. SAE ablation: Zero out specific SAE features, decode back to activation space.
2. Probe ablation (RD-3): Project out the linear probe direction.

Tests whether SAE features provide more surgical removal with less
collateral damage than a single probe direction.

Methodology:
    1. Train SAE and identify top-correlated features.
    2. Zero out top features, decode, measure prediction MSE change.
    3. Extract probe direction (as in RD-3), ablate, measure MSE change.
    4. Compare collateral damage: which method preserves more other information?

Reference:
    - Bricken et al. "Towards Monosemanticity" (Anthropic, 2023)
    - Elazar et al. "Amnesic Probing" arXiv:2006.00995
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
from src.probing.linear_probe import LinearProbe, probe_layer  # noqa: E402
from src.sae.sparse_autoencoder import (  # noqa: E402
    SAETrainer,
    TabPFNSparseAutoencoder,
    generate_diverse_datasets,
)

QUICK_RUN = True
RANDOM_SEED = 42

ALPHA = 2.0
BETA = 3.0
N_TRAIN = 50
N_TEST = 10

COLLECTION_LAYER = 6
TOKEN_IDX = -1
N_DATASETS = 50 if not QUICK_RUN else 20
EXPANSION_FACTOR = 4
EPOCHS = 100 if not QUICK_RUN else 30
BATCH_SIZE = 64
LR = 1e-3
L1_COEFF = 1e-3

TOP_K_ABLATE = 5


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _generate_primary_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    n_total = N_TRAIN + N_TEST
    X_all = rng.standard_normal((n_total, 2), dtype=np.float32)

    X_train = X_all[:N_TRAIN]
    X_test = X_all[N_TRAIN:]
    y_train = (ALPHA * X_train[:, 0] + BETA * X_train[:, 1]).astype(np.float32)
    y_test = (ALPHA * X_test[:, 0] + BETA * X_test[:, 1]).astype(np.float32)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def _measure_probe_r2(activations: np.ndarray, targets: np.ndarray) -> float:
    results = probe_layer(
        activations,
        targets,
        complexities=[0],
        test_size=0.2,
        random_seed=RANDOM_SEED,
    )
    return float(results[0]["r2"])


def _extract_probe_direction(
    activations: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    probe = LinearProbe(complexity=0, random_seed=RANDOM_SEED)
    probe.fit(activations, targets)

    v_scaled = probe.model.coef_.flatten()
    v_original = v_scaled / probe.scaler.scale_

    norm = np.linalg.norm(v_original)
    if norm <= 0:
        return v_original
    return v_original / norm


def _ablate_direction(activations: np.ndarray, v_hat: np.ndarray) -> np.ndarray:
    return activations - (activations @ v_hat[:, None]) * v_hat[None, :]


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return 0.0
    corr, _ = pearsonr(x, y)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _extract_primary_activations(
    model: TabPFNRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.fit(X_train, y_train)
    hooker = TabPFNHookedModel(model)
    preds_original, cache = hooker.forward_with_cache(X_test)

    act_layer6 = cache["layers"][COLLECTION_LAYER]
    single_eval_pos = int(cache["single_eval_pos"])
    act_train = act_layer6[0, :single_eval_pos, TOKEN_IDX, :].detach().cpu().numpy()

    return (
        act_train.astype(np.float32),
        preds_original.astype(np.float32),
        float(
            np.mean(
                (preds_original - (ALPHA * X_test[:, 0] + BETA * X_test[:, 1])) ** 2
            )
        ),
    )


def _rank_sae_features(
    encoded: np.ndarray,
    y_train: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    feature_activity = encoded.mean(axis=0)
    feature_corr = np.array(
        [_safe_corr(encoded[:, i], y_train) for i in range(encoded.shape[1])]
    )

    candidate_pool = min(encoded.shape[1], max(top_k * 10, top_k))
    active_idx = np.argsort(-feature_activity)[:candidate_pool]
    active_corr_scores = np.abs(feature_corr[active_idx])
    picked_local = np.argsort(-active_corr_scores)[:top_k]
    top_features = active_idx[picked_local]

    return {
        "top_features": top_features.astype(int).tolist(),
        "feature_activity": feature_activity.astype(np.float64).tolist(),
        "feature_corr": feature_corr.astype(np.float64).tolist(),
    }


def _ablate_sae_features(
    sae: TabPFNSparseAutoencoder,
    act_train: np.ndarray,
    feature_indices: list[int],
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray, np.ndarray]:
    sae.eval()
    with torch.no_grad():
        act_tensor = torch.from_numpy(act_train).float()
        encoded = sae.encode(act_tensor)
        decoded_original = sae.decode(encoded).cpu().numpy()

        per_feature_decoded: dict[int, np.ndarray] = {}
        for idx in feature_indices:
            encoded_ablated = encoded.clone()
            encoded_ablated[:, idx] = 0.0
            per_feature_decoded[idx] = sae.decode(encoded_ablated).cpu().numpy()

        encoded_joint_ablated = encoded.clone()
        encoded_joint_ablated[:, feature_indices] = 0.0
        decoded_joint = sae.decode(encoded_joint_ablated).cpu().numpy()

    return decoded_original, per_feature_decoded, decoded_joint, encoded.cpu().numpy()


def _compute_per_feature_metrics(
    act_train: np.ndarray,
    decoded_original: np.ndarray,
    per_feature_decoded: dict[int, np.ndarray],
    feature_indices: list[int],
    y_train: np.ndarray,
    orth_target: np.ndarray,
    baseline_y_r2: float,
    baseline_orth_r2: float,
) -> list[dict[str, float | int]]:
    base_recon_mse = float(np.mean((decoded_original - act_train) ** 2))
    rows: list[dict[str, float | int]] = []

    for idx in feature_indices:
        reconstructed = per_feature_decoded[idx]
        recon_mse = float(np.mean((reconstructed - act_train) ** 2))
        y_r2_after = _measure_probe_r2(reconstructed, y_train)
        orth_r2_after = _measure_probe_r2(reconstructed, orth_target)
        rows.append(
            {
                "feature_idx": int(idx),
                "reconstruction_mse": recon_mse,
                "mse_increase": recon_mse - base_recon_mse,
                "y_r2_after": y_r2_after,
                "y_r2_drop": baseline_y_r2 - y_r2_after,
                "orth_r2_after": orth_r2_after,
                "orth_r2_change": orth_r2_after - baseline_orth_r2,
            }
        )
    return rows


def _compare_methods(
    act_train: np.ndarray,
    sae_joint_decoded: np.ndarray,
    y_train: np.ndarray,
    orth_target: np.ndarray,
    baseline_y_r2: float,
    baseline_orth_r2: float,
) -> dict[str, dict[str, float]]:
    y_sae = _measure_probe_r2(sae_joint_decoded, y_train)
    orth_sae = _measure_probe_r2(sae_joint_decoded, orth_target)

    v_hat = _extract_probe_direction(act_train, y_train)
    probe_ablated = _ablate_direction(act_train, v_hat)
    y_probe = _measure_probe_r2(probe_ablated, y_train)
    orth_probe = _measure_probe_r2(probe_ablated, orth_target)

    return {
        "sae_joint_ablation": {
            "y_r2_after": y_sae,
            "y_r2_drop": baseline_y_r2 - y_sae,
            "orth_r2_after": orth_sae,
            "orth_r2_change": orth_sae - baseline_orth_r2,
            "reconstruction_mse": float(np.mean((sae_joint_decoded - act_train) ** 2)),
        },
        "probe_direction_ablation": {
            "y_r2_after": y_probe,
            "y_r2_drop": baseline_y_r2 - y_probe,
            "orth_r2_after": orth_probe,
            "orth_r2_change": orth_probe - baseline_orth_r2,
            "reconstruction_mse": float(np.mean((probe_ablated - act_train) ** 2)),
        },
    }


def _plot_results(
    per_feature_rows: list[dict[str, float | int]],
    method_comparison: dict[str, dict[str, float]],
    save_path: Path,
) -> None:
    feature_labels = [f"f{int(row['feature_idx'])}" for row in per_feature_rows]
    mse_increase = [float(row["mse_increase"]) for row in per_feature_rows]

    methods = ["SAE (top-K)", "Probe dir"]
    y_drop = [
        method_comparison["sae_joint_ablation"]["y_r2_drop"],
        method_comparison["probe_direction_ablation"]["y_r2_drop"],
    ]
    orth_change = [
        method_comparison["sae_joint_ablation"]["orth_r2_change"],
        method_comparison["probe_direction_ablation"]["orth_r2_change"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    axes[0].bar(feature_labels, mse_increase, color="#4C78A8", alpha=0.9)
    axes[0].set_title(
        "(a) SAE Feature Ablation\nMSE Increase per Feature", fontweight="bold"
    )
    axes[0].set_xlabel("Ablated SAE feature")
    axes[0].set_ylabel("Activation reconstruction MSE increase")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(methods, y_drop, color=["#59A14F", "#E15759"], alpha=0.9)
    axes[1].set_title(
        "(b) Target Information Removal\ny-probe R2 drop", fontweight="bold"
    )
    axes[1].set_ylabel("R2 drop (higher = more removal)")
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].bar(methods, orth_change, color=["#59A14F", "#E15759"], alpha=0.9)
    axes[2].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[2].set_title(
        "(c) Collateral Damage\nOrthogonal probe R2 change", fontweight="bold"
    )
    axes[2].set_ylabel("R2 change (closer to 0 is better)")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "RD-6 SAE Feature Ablation vs RD-3 Probe Direction Ablation",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _set_seeds(RANDOM_SEED)

    out_dir = ROOT / "results" / "rd6" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RD-6 M10-T4: SAE Feature Ablation vs RD-3 Probe Direction Ablation")
    print("=" * 80)
    print(
        f"QUICK_RUN={QUICK_RUN} | N_DATASETS={N_DATASETS} | EPOCHS={EPOCHS} | TOP_K={TOP_K_ABLATE}"
    )

    print("\n[1/8] Generating primary dataset and fitting base model ...")
    data = _generate_primary_data()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    model = TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")

    print("[2/8] Training SAE on diverse datasets (layer-6 activations) ...")
    diverse_datasets = generate_diverse_datasets(
        n_datasets=N_DATASETS,
        random_seed=RANDOM_SEED,
    )
    torch.manual_seed(RANDOM_SEED)
    sae = TabPFNSparseAutoencoder(input_dim=192, expansion_factor=EXPANSION_FACTOR)
    trainer = SAETrainer(sae, lr=LR, l1_coeff=L1_COEFF)
    activations = trainer.collect_activations(
        model, diverse_datasets, layer=COLLECTION_LAYER
    )
    history = trainer.train(activations, epochs=EPOCHS, batch_size=BATCH_SIZE)

    print("[3/8] Collecting primary activations via forward_with_cache ...")
    act_train, preds_original, pred_mse = _extract_primary_activations(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
    )
    print(f"  Primary act shape: {act_train.shape} | Test pred MSE: {pred_mse:.6f}")

    print("[4/8] Ranking SAE features and running feature-wise ablation ...")
    orth_target = X_train.mean(axis=1).astype(np.float32)
    baseline_y_r2 = _measure_probe_r2(act_train, y_train)
    baseline_orth_r2 = _measure_probe_r2(act_train, orth_target)

    with torch.no_grad():
        encoded_primary = sae.encode(torch.from_numpy(act_train).float()).cpu().numpy()
    ranking = _rank_sae_features(encoded_primary, y_train, TOP_K_ABLATE)
    top_features = ranking["top_features"]

    decoded_original, per_feature_decoded, decoded_joint, _ = _ablate_sae_features(
        sae,
        act_train,
        top_features,
    )

    per_feature_rows = _compute_per_feature_metrics(
        act_train=act_train,
        decoded_original=decoded_original,
        per_feature_decoded=per_feature_decoded,
        feature_indices=top_features,
        y_train=y_train,
        orth_target=orth_target,
        baseline_y_r2=baseline_y_r2,
        baseline_orth_r2=baseline_orth_r2,
    )

    print("[5/8] Running RD-3 probe-direction ablation for direct comparison ...")
    method_comparison = _compare_methods(
        act_train=act_train,
        sae_joint_decoded=decoded_joint,
        y_train=y_train,
        orth_target=orth_target,
        baseline_y_r2=baseline_y_r2,
        baseline_orth_r2=baseline_orth_r2,
    )

    print("[6/8] Plotting three-panel comparison figure ...")
    plot_path = out_dir / "feature_ablation.png"
    _plot_results(per_feature_rows, method_comparison, plot_path)

    print("[7/8] Saving results JSON ...")
    results_payload: dict[str, Any] = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "data": {
            "alpha": ALPHA,
            "beta": BETA,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
        },
        "sae_config": {
            "collection_layer": COLLECTION_LAYER,
            "token_idx": TOKEN_IDX,
            "n_datasets": N_DATASETS,
            "expansion_factor": EXPANSION_FACTOR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "l1_coeff": L1_COEFF,
        },
        "baseline": {
            "y_probe_r2": baseline_y_r2,
            "orth_probe_r2": baseline_orth_r2,
            "test_pred_mse": pred_mse,
            "pred_mean": float(np.mean(preds_original)),
            "y_test_mean": float(np.mean(y_test)),
        },
        "feature_ranking": {
            "top_features": top_features,
            "feature_corr_to_y": [ranking["feature_corr"][i] for i in top_features],
            "feature_activity": [ranking["feature_activity"][i] for i in top_features],
        },
        "sae_feature_ablation": per_feature_rows,
        "method_comparison": method_comparison,
        "sae_training_history": {
            "final_total_loss": float(history["total_loss"][-1]),
            "final_mse_loss": float(history["mse_loss"][-1]),
            "final_l1_loss": float(history["l1_loss"][-1]),
            "final_sparsity": float(history["sparsity"][-1]),
            "final_reconstruction_r2": float(history["reconstruction_r2"][-1]),
        },
    }

    json_path = out_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("[8/8] Final summary")
    sae_drop = method_comparison["sae_joint_ablation"]["y_r2_drop"]
    probe_drop = method_comparison["probe_direction_ablation"]["y_r2_drop"]
    sae_collateral = abs(method_comparison["sae_joint_ablation"]["orth_r2_change"])
    probe_collateral = abs(
        method_comparison["probe_direction_ablation"]["orth_r2_change"]
    )

    both_reduce = sae_drop > 0 and probe_drop > 0
    sae_better_collateral = sae_collateral < probe_collateral

    print(f"  - Outputs: {plot_path} | {json_path}")
    print(
        "  - Success criterion (both reduce y-probe R2): "
        f"{'PASS' if both_reduce else 'CHECK'}"
    )
    print(
        "  - Success criterion (SAE lower collateral damage): "
        f"{'PASS' if sae_better_collateral else 'CHECK'}"
    )
    print(
        f"  - y-probe R2 drop: SAE={sae_drop:+.4f}, Probe={probe_drop:+.4f} | "
        f"|orth change|: SAE={sae_collateral:.4f}, Probe={probe_collateral:.4f}"
    )


if __name__ == "__main__":
    main()
