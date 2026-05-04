# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportImplicitStringConcatenation=false
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
from typing import Any

import numpy as np
import torch
from iltm import iLTMRegressor
from scipy.stats import pearsonr
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

from src.data.synthetic_generator import generate_linear_data
from src.hooks.iltm_hooker import iLTMHookedModel
from src.hooks.tabicl_hooker import TabICLHookedModel
from src.hooks.tabpfn_hooker import TabPFNHookedModel
from src.sae.sparse_autoencoder import SAETrainer, TabPFNSparseAutoencoder
from rd5_config import cfg

QUICK_RUN = cfg.QUICK_RUN
RANDOM_SEED = cfg.SEED

N_TRAIN = cfg.N_TRAIN
N_TEST = cfg.N_TEST
SAE_EPOCHS = cfg.sae_epochs
SAE_BATCH_SIZE = cfg.SAE_BATCH_SIZE
SAE_LR = cfg.SAE_LR
EXPANSION_FACTOR = cfg.EXPANSION_FACTOR

RESULTS_DIR = ROOT / "results" / "rd5" / "sae"


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"Pearson shape mismatch: {x_arr.shape} vs {y_arr.shape}")
    if x_arr.size < 2:
        return 0.0
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return 0.0
    r, _ = pearsonr(x_arr, y_arr)
    return float(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0))


def _build_pairs(model_name: str) -> list[tuple[int, int]]:
    if QUICK_RUN:
        base_pairs = [(a, b) for a in (1, 3, 5) for b in (1, 3, 5)]
        if model_name == "iltm":
            return [(1, 1), (1, 5), (5, 1)]
        return base_pairs
    return [(a, b) for a in range(1, 6) for b in range(1, 6)]


def _build_tabpfn() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _build_tabicl() -> TabICLRegressor:
    return TabICLRegressor(device=cfg.DEVICE, random_state=RANDOM_SEED)


def _build_iltm() -> iLTMRegressor:
    return iLTMRegressor(device="cpu", n_ensemble=1, seed=RANDOM_SEED)


def _extract_model_activations(
    model_name: str,
    core_layer: int,
    hidden_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = _build_pairs(model_name)

    all_acts: list[np.ndarray] = []
    all_alphas: list[np.ndarray] = []
    all_betas: list[np.ndarray] = []

    for ds_idx, (alpha, beta) in enumerate(pairs):
        ds = generate_linear_data(
            alpha=float(alpha),
            beta=float(beta),
            n_train=N_TRAIN,
            n_test=N_TEST,
            random_seed=RANDOM_SEED + ds_idx,
        )

        if model_name == "tabpfn":
            model = _build_tabpfn()
            model.fit(ds.X_train, ds.y_train)
            hooker = TabPFNHookedModel(model)
            _, cache = hooker.forward_with_cache(ds.X_test)
            acts = hooker.get_test_label_token(cache, core_layer)
        elif model_name == "tabicl":
            model = _build_tabicl()
            model.fit(ds.X_train, ds.y_train)
            hooker = TabICLHookedModel(model)
            _, cache = hooker.forward_with_cache(ds.X_test)
            acts = hooker.get_layer_activations(cache, core_layer)
        elif model_name == "iltm":
            model = _build_iltm()
            model.fit(ds.X_train, ds.y_train)
            hooker = iLTMHookedModel(model)
            _, cache = hooker.forward_with_cache(ds.X_test)
            acts = hooker.get_layer_activations(cache, core_layer)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        acts_np = np.asarray(acts, dtype=np.float32)
        if acts_np.ndim != 2 or acts_np.shape[1] != hidden_dim:
            raise ValueError(
                f"Unexpected activation shape for {model_name}: {acts_np.shape}, "
                f"expected [n_samples, {hidden_dim}]"
            )

        n_samples = acts_np.shape[0]
        all_acts.append(acts_np)
        all_alphas.append(np.full(n_samples, float(alpha), dtype=np.float32))
        all_betas.append(np.full(n_samples, float(beta), dtype=np.float32))

        print(
            f"[{model_name}] dataset {ds_idx + 1}/{len(pairs)} "
            f"alpha={alpha}, beta={beta}, activations={acts_np.shape}"
        )

    pooled_acts = np.concatenate(all_acts, axis=0)
    pooled_alphas = np.concatenate(all_alphas, axis=0)
    pooled_betas = np.concatenate(all_betas, axis=0)
    return pooled_acts, pooled_alphas, pooled_betas


def _train_and_analyze_model(
    model_name: str,
    core_layer: int,
    hidden_dim: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    acts, labels_alpha, labels_beta = _extract_model_activations(
        model_name=model_name,
        core_layer=core_layer,
        hidden_dim=hidden_dim,
    )

    act_tensor = torch.FloatTensor(acts)
    sae = TabPFNSparseAutoencoder(
        input_dim=hidden_dim, expansion_factor=EXPANSION_FACTOR
    )
    if hasattr(sae, "train_sae"):
        sae.train_sae(
            act_tensor,
            n_epochs=SAE_EPOCHS,
            batch_size=SAE_BATCH_SIZE,
            lr=SAE_LR,
        )
    else:
        trainer = SAETrainer(sae, lr=SAE_LR, l1_coeff=1e-3)
        trainer.train(acts, epochs=SAE_EPOCHS, batch_size=SAE_BATCH_SIZE)

    with torch.no_grad():
        features_tensor = sae.encode(act_tensor)
        forward_out = sae.forward(act_tensor)
        recon_tensor = forward_out[0] if isinstance(forward_out, tuple) else forward_out

    features = features_tensor.detach().cpu().numpy()
    recon = recon_tensor.detach().cpu().numpy()

    alpha_corrs: list[float] = []
    beta_corrs: list[float] = []
    for feat_idx in range(features.shape[1]):
        alpha_corrs.append(_safe_pearson(features[:, feat_idx], labels_alpha))
        beta_corrs.append(_safe_pearson(features[:, feat_idx], labels_beta))

    alpha_arr = np.asarray(alpha_corrs, dtype=np.float32)
    beta_arr = np.asarray(beta_corrs, dtype=np.float32)

    top_alpha_idx = np.argsort(np.abs(alpha_arr))[::-1][:5]
    top_beta_idx = np.argsort(np.abs(beta_arr))[::-1][:5]

    result: dict[str, Any] = {
        "core_layer": int(core_layer),
        "input_dim": int(hidden_dim),
        "n_sae_features": int(features.shape[1]),
        "top_alpha_features": [
            {"idx": int(i), "corr": float(alpha_arr[i])} for i in top_alpha_idx
        ],
        "top_beta_features": [
            {"idx": int(i), "corr": float(beta_arr[i])} for i in top_beta_idx
        ],
        "max_alpha_corr": float(np.max(np.abs(alpha_arr))),
        "max_beta_corr": float(np.max(np.abs(beta_arr))),
        "recon_loss": float(np.mean((recon - acts) ** 2)),
    }

    return result, alpha_arr, beta_arr


def _evaluate_hypotheses(
    results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    alpha_maxes = np.array(
        [
            float(results["tabpfn"]["max_alpha_corr"]),
            float(results["tabicl"]["max_alpha_corr"]),
            float(results["iltm"]["max_alpha_corr"]),
        ],
        dtype=np.float32,
    )
    beta_maxes = np.array(
        [
            float(results["tabpfn"]["max_beta_corr"]),
            float(results["tabicl"]["max_beta_corr"]),
            float(results["iltm"]["max_beta_corr"]),
        ],
        dtype=np.float32,
    )

    large_models_mean = float(
        np.mean(
            [
                0.5
                * (
                    float(results["tabicl"]["max_alpha_corr"])
                    + float(results["tabicl"]["max_beta_corr"])
                ),
                0.5
                * (
                    float(results["iltm"]["max_alpha_corr"])
                    + float(results["iltm"]["max_beta_corr"])
                ),
            ]
        )
    )
    tabpfn_mean = float(
        0.5
        * (
            float(results["tabpfn"]["max_alpha_corr"])
            + float(results["tabpfn"]["max_beta_corr"])
        )
    )

    h6_supported = bool(np.std(alpha_maxes) < 0.2 and np.std(beta_maxes) < 0.2)
    h7_supported = bool(large_models_mean > tabpfn_mean)
    h8_supported = bool(
        np.min(alpha_maxes) > 0.5
        and np.min(beta_maxes) > 0.5
        and all(
            int(results[m]["n_sae_features"]) >= 4 * int(results[m]["input_dim"])
            for m in ("tabpfn", "tabicl", "iltm")
        )
    )

    return {
        "H6": {
            "description": "Similar coefficient correlation patterns",
            "supported": h6_supported,
        },
        "H7": {
            "description": "Larger model → finer decomposition",
            "supported": h7_supported,
        },
        "H8": {
            "description": "Cross-model feature universality",
            "supported": h8_supported,
        },
    }


def _plot_histograms(
    corr_map: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    model_order = ["tabpfn", "tabicl", "iltm"]
    titles = {"tabpfn": "TabPFN (L6)", "tabicl": "TabICL (L5)", "iltm": "iLTM (L1)"}

    bins = np.linspace(-1.0, 1.0, 41)
    for ax, model_key in zip(axes, model_order, strict=True):
        alpha_corr, beta_corr = corr_map[model_key]
        ax.hist(alpha_corr, bins=bins, alpha=0.55, color="#1f77b4", label="alpha")
        ax.hist(beta_corr, bins=bins, alpha=0.55, color="#ff7f0e", label="beta")
        ax.set_title(titles[model_key])
        ax.set_xlabel("Pearson correlation")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left")
        if model_key == "tabpfn":
            ax.set_ylabel("Number of SAE features")

    fig.suptitle("RD5 SAE comparison: feature-coefficient correlation histograms")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RD5 SAE Comparison: TabPFN vs TabICL vs iLTM")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}, SAE_EPOCHS={SAE_EPOCHS}")

    model_specs = {
        "tabpfn": {"core_layer": 6, "input_dim": 192},
        "tabicl": {"core_layer": 5, "input_dim": 512},
        "iltm": {"core_layer": 1, "input_dim": 512},
    }

    result_payload: dict[str, Any] = {}
    corr_payload: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for model_name, spec in model_specs.items():
        print(
            f"\n[{model_name}] core_layer={spec['core_layer']}, "
            f"input_dim={spec['input_dim']}"
        )
        model_result, alpha_corr, beta_corr = _train_and_analyze_model(
            model_name=model_name,
            core_layer=int(spec["core_layer"]),
            hidden_dim=int(spec["input_dim"]),
        )
        result_payload[model_name] = model_result
        corr_payload[model_name] = (alpha_corr, beta_corr)

        print(
            f"  max|r_alpha|={model_result['max_alpha_corr']:.4f}, "
            f"max|r_beta|={model_result['max_beta_corr']:.4f}, "
            f"recon_loss={model_result['recon_loss']:.6f}"
        )

    result_payload["hypotheses"] = _evaluate_hypotheses(result_payload)

    json_path = RESULTS_DIR / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    plot_path = RESULTS_DIR / "sae_comparison.png"
    _plot_histograms(corr_payload, plot_path)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
