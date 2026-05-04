# pyright: reportMissingImports=false
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
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rd5_config import cfg
from src.data.synthetic_generator import generate_linear_data
from src.hooks.tabicl_hooker import TabICLHookedModel
from src.hooks.tabpfn_hooker import TabPFNHookedModel
from src.sae.sparse_autoencoder import SAETrainer, TabPFNSparseAutoencoder


RESULTS_DIR = ROOT / "results" / "neurips"
JSON_PATH = RESULTS_DIR / "m3_sae_expansion_ablation.json"
PLOT_PATH = RESULTS_DIR / "m3_sae_expansion.png"

EXPANSION_FACTORS = [4, 16, 32, 64]
L1_COEFF = 1e-3
# Override for tractable full-scale run: 10 datasets, 100 epochs
_OVERRIDE_N_DATASETS = 10 if not cfg.QUICK_RUN else None
_OVERRIDE_EPOCHS = 100 if not cfg.QUICK_RUN else None


def _safe_abs_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_arr.shape != y_arr.shape or x_arr.size < 2:
        return 0.0
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return 0.0
    corr = float(np.corrcoef(x_arr, y_arr)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(abs(corr))


def _build_dataset_specs(model_name: str) -> list[tuple[float, float]]:
    rng = np.random.default_rng(cfg.SEED)
    count = (
        _OVERRIDE_N_DATASETS if _OVERRIDE_N_DATASETS else cfg.dataset_count(model_name)
    )
    specs: list[tuple[float, float]] = []
    for _ in range(count):
        alpha = float(rng.uniform(0.5, 5.0))
        beta = float(rng.uniform(0.5, 5.0))
        specs.append((alpha, beta))
    return specs


def _collect_activations(
    model_name: str,
    layer_idx: int,
    hidden_dim: int,
    dataset_specs: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    all_acts: list[np.ndarray] = []
    all_alpha: list[np.ndarray] = []

    if model_name == "tabpfn":
        model = TabPFNRegressor(
            device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt"
        )
    elif model_name == "tabicl":
        model = TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    for ds_idx, (alpha, beta) in enumerate(dataset_specs):
        ds = generate_linear_data(
            alpha=alpha,
            beta=beta,
            n_train=cfg.N_TRAIN,
            n_test=cfg.N_TEST,
            random_seed=cfg.SEED + ds_idx,
        )

        model.fit(ds.X_train, ds.y_train)
        if model_name == "tabpfn":
            hooker = TabPFNHookedModel(model)
            _, cache = hooker.forward_with_cache(ds.X_test)
            acts = np.asarray(
                hooker.get_test_label_token(cache, layer_idx), dtype=np.float32
            )
        else:
            hooker = TabICLHookedModel(model)
            _, cache = hooker.forward_with_cache(ds.X_test)
            acts = np.asarray(
                hooker.get_layer_activations(cache, layer_idx), dtype=np.float32
            )

        if acts.ndim != 2 or acts.shape[1] != hidden_dim:
            raise ValueError(
                f"Unexpected activation shape for {model_name}: {acts.shape}; "
                f"expected [n_samples, {hidden_dim}]"
            )

        all_acts.append(acts)
        all_alpha.append(np.full(acts.shape[0], alpha, dtype=np.float32))

    return np.concatenate(all_acts, axis=0), np.concatenate(all_alpha, axis=0)


def _train_sae(
    activations: np.ndarray,
    alpha_targets: np.ndarray,
    input_dim: int,
    expansion: int,
) -> dict[str, float]:
    torch.manual_seed(cfg.SEED)
    sae = TabPFNSparseAutoencoder(
        input_dim=input_dim,
        expansion_factor=expansion,
        activation="topk",
    )
    trainer = SAETrainer(sae, lr=cfg.SAE_LR, l1_coeff=L1_COEFF)
    history = trainer.train(
        activations=activations,
        epochs=_OVERRIDE_EPOCHS if _OVERRIDE_EPOCHS else cfg.sae_epochs,
        batch_size=cfg.SAE_BATCH_SIZE,
        verbose=False,
    )

    with torch.no_grad():
        act_tensor = torch.from_numpy(np.asarray(activations, dtype=np.float32))
        recon, feats = sae(act_tensor)

    recon_np = recon.detach().cpu().numpy()
    feats_np = feats.detach().cpu().numpy()

    max_r_alpha = 0.0
    for feat_idx in range(feats_np.shape[1]):
        max_r_alpha = max(
            max_r_alpha,
            _safe_abs_pearson(feats_np[:, feat_idx], alpha_targets),
        )

    return {
        "reconstruction_r2": float(history["reconstruction_r2"][-1]),
        "sparsity": float((np.abs(feats_np) <= 1e-8).mean()),
        "max_abs_r_alpha": float(max_r_alpha),
        "mse": float(np.mean((recon_np - activations) ** 2)),
    }


def _plot(results: dict[str, Any], save_path: Path) -> None:
    expansions = np.asarray(EXPANSION_FACTORS, dtype=np.int32)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)

    metric_specs = [
        ("reconstruction_r2", "Reconstruction R2"),
        ("sparsity", "Sparsity"),
        ("max_abs_r_alpha", "Max |r_alpha|"),
    ]

    for ax_idx, (metric_key, title) in enumerate(metric_specs):
        ax = axes[0, ax_idx]
        for model_name, color in (("tabpfn", "#1f77b4"), ("tabicl", "#ff7f0e")):
            y = [
                float(results[model_name][str(expansion)][metric_key])
                for expansion in EXPANSION_FACTORS
            ]
            ax.plot(
                expansions, y, marker="o", linewidth=2.0, color=color, label=model_name
            )

        ax.set_title(title)
        ax.set_xlabel("Expansion factor")
        ax.set_xticks(EXPANSION_FACTORS)
        ax.grid(alpha=0.25)
        if metric_key == "reconstruction_r2":
            ax.legend(loc="best")

    fig.suptitle("NeurIPS M3 SAE Expansion Ablation (TopK)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_specs: dict[str, dict[str, int]] = {
        "tabpfn": {"layer": 8, "dim": 192},
        "tabicl": {"layer": 1, "dim": 512},
    }

    results: dict[str, Any] = {}
    for model_name, spec in model_specs.items():
        dataset_specs = _build_dataset_specs(model_name)
        print(
            f"[{model_name}] collecting activations "
            f"(layer={spec['layer']}, dim={spec['dim']}, n_datasets={len(dataset_specs)})"
        )
        activations, alpha_targets = _collect_activations(
            model_name=model_name,
            layer_idx=spec["layer"],
            hidden_dim=spec["dim"],
            dataset_specs=dataset_specs,
        )
        print(f"[{model_name}] pooled activations: {activations.shape}")

        model_results: dict[str, Any] = {}
        for expansion in EXPANSION_FACTORS:
            print(f"[{model_name}] training SAE topk x{expansion}...")
            metrics = _train_sae(
                activations=activations,
                alpha_targets=alpha_targets,
                input_dim=spec["dim"],
                expansion=expansion,
            )
            model_results[str(expansion)] = metrics
            print(
                f"  r2={metrics['reconstruction_r2']:.4f}, sparsity={metrics['sparsity']:.3f}, "
                f"max|r_alpha|={metrics['max_abs_r_alpha']:.4f}"
            )

        results[model_name] = model_results

    payload = {
        "quick_run": bool(cfg.QUICK_RUN),
        "seed": int(cfg.SEED),
        "device": cfg.DEVICE,
        "expansion_factors": EXPANSION_FACTORS,
        "epochs": int(_OVERRIDE_EPOCHS if _OVERRIDE_EPOCHS else cfg.sae_epochs),
        "batch_size": int(cfg.SAE_BATCH_SIZE),
        "learning_rate": float(cfg.SAE_LR),
        "l1_coeff": float(L1_COEFF),
        "dataset_counts": {
            "tabpfn": int(
                _OVERRIDE_N_DATASETS
                if _OVERRIDE_N_DATASETS
                else cfg.dataset_count("tabpfn")
            ),
            "tabicl": int(
                _OVERRIDE_N_DATASETS
                if _OVERRIDE_N_DATASETS
                else cfg.dataset_count("tabicl")
            ),
        },
        "model_specs": model_specs,
        "results": results,
    }

    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _plot(results=results, save_path=PLOT_PATH)
    print(f"Saved: {JSON_PATH}")
    print(f"Saved: {PLOT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
