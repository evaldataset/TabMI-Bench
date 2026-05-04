# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportImplicitStringConcatenation=false
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from iltm import iLTMRegressor
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rd5_config import cfg  # noqa: E402
from src.data.synthetic_generator import generate_linear_data  # noqa: E402
from src.hooks.iltm_hooker import iLTMHookedModel  # noqa: E402
from src.hooks.tabicl_hooker import TabICLHookedModel  # noqa: E402
from src.hooks.tabpfn_hooker import TabPFNHookedModel  # noqa: E402
from src.sae.sparse_autoencoder import SAETrainer, TabPFNSparseAutoencoder  # noqa: E402


RESULTS_DIR = ROOT / "results" / "rd6" / "improved_sae"
RNG = np.random.default_rng(cfg.SEED)
N_DATASETS = 10 if cfg.QUICK_RUN else 20
EPOCHS = 50 if cfg.QUICK_RUN else 200
BATCH_SIZE = cfg.SAE_BATCH_SIZE
LR = cfg.SAE_LR
L1_COEFF = 1e-3


class ModelSpec(TypedDict):
    layer: int
    input_dim: int


class VariantSpec(TypedDict):
    name: str
    activation: str
    expansion_factor: int


def _safe_abs_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"Shape mismatch for Pearson: {x_arr.shape} vs {y_arr.shape}")
    if x_arr.size < 2:
        return 0.0
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return 0.0
    corr_f = float(np.corrcoef(x_arr, y_arr)[0, 1])
    if not np.isfinite(corr_f):
        return 0.0
    return float(abs(corr_f))


def _build_tabpfn() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _build_tabicl() -> TabICLRegressor:
    return TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)


def _build_iltm() -> iLTMRegressor:
    return iLTMRegressor(device="cpu", n_ensemble=1, seed=cfg.SEED)


def _build_dataset_specs(n_datasets: int) -> list[tuple[float, float]]:
    specs: list[tuple[float, float]] = []
    for _ in range(n_datasets):
        alpha = float(RNG.uniform(0.5, 5.0))
        beta = float(RNG.uniform(0.5, 5.0))
        specs.append((alpha, beta))
    return specs


def _extract_model_activations(
    model_name: str,
    layer_idx: int,
    hidden_dim: int,
    dataset_specs: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    acts_all: list[np.ndarray] = []
    alpha_all: list[np.ndarray] = []
    beta_all: list[np.ndarray] = []

    if model_name == "tabpfn":
        model = _build_tabpfn()
    elif model_name == "tabicl":
        model = _build_tabicl()
    elif model_name == "iltm":
        model = _build_iltm()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    for ds_idx, (alpha, beta) in enumerate(dataset_specs):
        ds = generate_linear_data(
            alpha=alpha,
            beta=beta,
            n_train=cfg.N_TRAIN,
            n_test=cfg.N_TEST,
            random_seed=cfg.SEED + ds_idx,
        )

        if model_name == "tabpfn":
            model.fit(ds.X_train, ds.y_train)
            hooker = TabPFNHookedModel(model)
            _, cache = hooker.forward_with_cache(ds.X_test)
            acts = hooker.get_test_label_token(cache, layer_idx)
        elif model_name == "tabicl":
            model.fit(ds.X_train, ds.y_train)
            hooker = TabICLHookedModel(model)
            _, cache = hooker.forward_with_cache(ds.X_test)
            acts = hooker.get_layer_activations(cache, layer_idx)
        else:
            model.fit(ds.X_train, ds.y_train)
            hooker = iLTMHookedModel(model)
            _, cache = hooker.forward_with_cache(ds.X_test)
            acts = hooker.get_layer_activations(cache, layer_idx)

        acts_np = np.asarray(acts, dtype=np.float32)
        if acts_np.ndim != 2 or acts_np.shape[1] != hidden_dim:
            raise ValueError(
                f"Unexpected activation shape {acts_np.shape} for {model_name}; "
                f"expected [n_samples, {hidden_dim}]"
            )

        n_samples = acts_np.shape[0]
        acts_all.append(acts_np)
        alpha_all.append(np.full(n_samples, alpha, dtype=np.float32))
        beta_all.append(np.full(n_samples, beta, dtype=np.float32))

    return (
        np.concatenate(acts_all, axis=0),
        np.concatenate(alpha_all, axis=0),
        np.concatenate(beta_all, axis=0),
    )


def _train_variant(
    activations: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    input_dim: int,
    expansion_factor: int,
    activation_name: str,
) -> dict[str, float]:
    torch.manual_seed(cfg.SEED)
    sae = TabPFNSparseAutoencoder(
        input_dim=input_dim,
        expansion_factor=expansion_factor,
        activation=activation_name,
    )
    trainer = SAETrainer(sae, lr=LR, l1_coeff=L1_COEFF)
    history = trainer.train(
        activations=activations,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=False,
    )

    with torch.no_grad():
        act_tensor = torch.from_numpy(np.asarray(activations, dtype=np.float32))
        recon, feats = sae(act_tensor)

    recon_np = recon.detach().cpu().numpy()
    feats_np = feats.detach().cpu().numpy()

    max_alpha_corr = 0.0
    max_beta_corr = 0.0
    for feat_idx in range(feats_np.shape[1]):
        feat_col = feats_np[:, feat_idx]
        max_alpha_corr = max(max_alpha_corr, _safe_abs_pearson(feat_col, alphas))
        max_beta_corr = max(max_beta_corr, _safe_abs_pearson(feat_col, betas))

    sparsity = float((np.abs(feats_np) <= 1e-4).mean())
    mse = float(np.mean((recon_np - activations) ** 2))

    return {
        "reconstruction_r2": float(history["reconstruction_r2"][-1]),
        "sparsity": sparsity,
        "mse": mse,
        "max_alpha_corr": max_alpha_corr,
        "max_beta_corr": max_beta_corr,
    }


def _plot_variant_comparison(
    all_results: dict[str, dict[str, dict[str, float]]],
    save_path: Path,
) -> None:
    variants = list(all_results.keys())
    models = ["tabpfn", "tabicl", "iltm"]
    x = np.arange(len(models), dtype=np.float32)
    width = 0.8 / max(len(variants), 1)

    fig, (ax_alpha, ax_beta) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for idx, variant in enumerate(variants):
        offsets = x - 0.4 + (idx + 0.5) * width
        alpha_vals = [all_results[variant][model]["max_alpha_corr"] for model in models]
        beta_vals = [all_results[variant][model]["max_beta_corr"] for model in models]

        ax_alpha.bar(offsets, alpha_vals, width=width, label=variant)
        ax_beta.bar(offsets, beta_vals, width=width, label=variant)

    ax_alpha.set_title("Max |r_alpha| by Variant and Model")
    ax_alpha.set_xticks(x)
    ax_alpha.set_xticklabels(models)
    ax_alpha.set_ylim(0.0, 1.0)
    ax_alpha.grid(alpha=0.25, axis="y")

    ax_beta.set_title("Max |r_beta| by Variant and Model")
    ax_beta.set_xticks(x)
    ax_beta.set_xticklabels(models)
    ax_beta.set_ylim(0.0, 1.0)
    ax_beta.grid(alpha=0.25, axis="y")

    ax_alpha.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_specs: dict[str, ModelSpec] = {
        "tabpfn": {"layer": 6, "input_dim": 192},
        "tabicl": {"layer": 5, "input_dim": 512},
        "iltm": {"layer": 1, "input_dim": 512},
    }

    variants: list[VariantSpec] = [
        {"name": "relu_4x", "activation": "relu", "expansion_factor": 4},
        {"name": "relu_16x", "activation": "relu", "expansion_factor": 16},
        {"name": "jumprelu_16x", "activation": "jumprelu", "expansion_factor": 16},
    ]
    if cfg.QUICK_RUN:
        variants = [v for v in variants if v["name"] != "relu_4x"]

    dataset_specs = _build_dataset_specs(N_DATASETS)
    model_dataset_specs: dict[str, list[tuple[float, float]]] = {}
    if cfg.QUICK_RUN:
        quick_counts = {"tabpfn": 4, "tabicl": 3, "iltm": 3}
        start = 0
        for model_name in ("tabpfn", "tabicl", "iltm"):
            count = quick_counts[model_name]
            model_dataset_specs[model_name] = dataset_specs[start : start + count]
            start += count
    else:
        for model_name in ("tabpfn", "tabicl", "iltm"):
            model_dataset_specs[model_name] = dataset_specs

    results_by_variant: dict[str, dict[str, dict[str, float]]] = {
        variant["name"]: {} for variant in variants
    }

    for model_name, model_spec in model_specs.items():
        print(
            f"[{model_name}] extracting activations: layer={model_spec['layer']}, "
            f"dim={model_spec['input_dim']}"
        )
        activations, alphas, betas = _extract_model_activations(
            model_name=model_name,
            layer_idx=int(model_spec["layer"]),
            hidden_dim=int(model_spec["input_dim"]),
            dataset_specs=model_dataset_specs[model_name],
        )
        print(f"[{model_name}] pooled activations: {activations.shape}")

        for variant in variants:
            variant_name = str(variant["name"])
            print(
                f"  -> training {variant_name} "
                f"(activation={variant['activation']}, x{variant['expansion_factor']})"
            )
            metrics = _train_variant(
                activations=activations,
                alphas=alphas,
                betas=betas,
                input_dim=int(model_spec["input_dim"]),
                expansion_factor=int(variant["expansion_factor"]),
                activation_name=str(variant["activation"]),
            )
            results_by_variant[variant_name][model_name] = metrics
            print(
                f"     r2={metrics['reconstruction_r2']:.4f}, sparsity={metrics['sparsity']:.3f}, "
                f"max|r_alpha|={metrics['max_alpha_corr']:.4f}, max|r_beta|={metrics['max_beta_corr']:.4f}"
            )

    plot_path = RESULTS_DIR / "comparison.png"
    _plot_variant_comparison(results_by_variant, plot_path)

    payload: dict[str, object] = {
        "quick_run": cfg.QUICK_RUN,
        "seed": cfg.SEED,
        "n_datasets_total": N_DATASETS,
        "n_datasets_per_model": {
            model_name: len(specs) for model_name, specs in model_dataset_specs.items()
        },
        "n_train": cfg.N_TRAIN,
        "n_test": cfg.N_TEST,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "l1_coeff": L1_COEFF,
        "variants": variants,
        "models": model_specs,
        "results": results_by_variant,
        "plot_path": str(plot_path),
    }
    json_path = RESULTS_DIR / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
