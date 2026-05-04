# pyright: reportMissingImports=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportImplicitStringConcatenation=false
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tabdpt import TabDPTRegressor
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.synthetic_generator import generate_linear_data
from src.hooks.tabdpt_hooker import TabDPTHookedModel
from src.hooks.tabicl_hooker import TabICLHookedModel
from src.hooks.tabpfn_hooker import TabPFNHookedModel
from src.sae.sparse_autoencoder import SAETrainer, TabPFNSparseAutoencoder


def _as_bool_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


QUICK_RUN = _as_bool_env(os.environ.get("QUICK_RUN", "0"))
SEED = int(os.environ.get("SEED", "42"))
DEVICE = os.environ.get("DEVICE", "cuda:0").strip()

N_DATASETS = 5 if QUICK_RUN else 10
N_TRAIN = 50
N_TEST = 10

EPOCHS = 100
LR = 1e-3
L1_COEFF = 1e-3
BATCH_SIZE = 64
EXPANSION_FACTOR = 16

TABDPT_HIDDEN_DIM = 768
TABDPT_LAYER_L1 = 0
TABDPT_LAYER_L8 = 8
TABPFN_LAYER_L6 = 6
TABICL_LAYER_L1 = 1

RESULTS_DIR = ROOT / "results" / "phase7"
RESULTS_PATH = RESULTS_DIR / f"tabdpt_sae_seed{SEED}.json"


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


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"R2 shape mismatch: {true.shape} vs {pred.shape}")
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _resolve_cuda_device() -> str:
    if not DEVICE.startswith("cuda"):
        raise RuntimeError(
            f"GPU is required for phase7_tabdpt_sae. Set DEVICE=cuda:0 (current: {DEVICE})."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return DEVICE


def _dataset_specs() -> list[tuple[float, float]]:
    rng = np.random.default_rng(SEED)
    specs: list[tuple[float, float]] = []
    for _ in range(N_DATASETS):
        alpha = float(rng.uniform(0.5, 5.0))
        beta = float(rng.uniform(0.5, 5.0))
        specs.append((alpha, beta))
    return specs


def _build_tabdpt_model(device: str) -> TabDPTRegressor:
    return TabDPTRegressor(device=device, compile=False, verbose=False)


def _build_tabpfn_model(device: str) -> TabPFNRegressor:
    return TabPFNRegressor(device=device, model_path="tabpfn-v2-regressor.ckpt")


def _build_tabicl_model(device: str) -> TabICLRegressor:
    return TabICLRegressor(device=device, random_state=SEED)


def _collect_tabdpt_activations(
    dataset_specs: list[tuple[float, float]], layer_idx: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    pooled_acts: list[np.ndarray] = []
    pooled_alpha: list[np.ndarray] = []

    for ds_idx, (alpha, beta) in enumerate(dataset_specs):
        ds = generate_linear_data(
            alpha=alpha,
            beta=beta,
            n_train=N_TRAIN,
            n_test=N_TEST,
            random_seed=SEED + ds_idx,
        )
        model = _build_tabdpt_model(device)
        model.fit(ds.X_train, ds.y_train)
        hooker = TabDPTHookedModel(model, device=device)
        acts = hooker.get_activations(ds.X_train, ds.y_train, ds.X_test)

        layer_acts = np.asarray(acts[layer_idx], dtype=np.float32)
        if layer_acts.ndim != 2 or layer_acts.shape[1] != TABDPT_HIDDEN_DIM:
            raise ValueError(
                f"Unexpected TabDPT activation shape: {layer_acts.shape}; expected [n_samples, {TABDPT_HIDDEN_DIM}]"
            )

        pooled_acts.append(layer_acts)
        pooled_alpha.append(np.full(layer_acts.shape[0], alpha, dtype=np.float32))
        print(
            f"[tabdpt:L{layer_idx}] dataset {ds_idx + 1}/{N_DATASETS} done, acts={layer_acts.shape}"
        )

    return np.concatenate(pooled_acts, axis=0), np.concatenate(pooled_alpha, axis=0)


def _collect_tabpfn_l6(
    dataset_specs: list[tuple[float, float]], device: str
) -> tuple[np.ndarray, np.ndarray]:
    pooled_acts: list[np.ndarray] = []
    pooled_alpha: list[np.ndarray] = []

    for ds_idx, (alpha, beta) in enumerate(dataset_specs):
        ds = generate_linear_data(
            alpha=alpha,
            beta=beta,
            n_train=N_TRAIN,
            n_test=N_TEST,
            random_seed=SEED + ds_idx,
        )
        model = _build_tabpfn_model(device)
        model.fit(ds.X_train, ds.y_train)
        hooker = TabPFNHookedModel(model)
        _, cache = hooker.forward_with_cache(ds.X_test)
        acts = np.asarray(
            hooker.get_test_label_token(cache, TABPFN_LAYER_L6), dtype=np.float32
        )

        pooled_acts.append(acts)
        pooled_alpha.append(np.full(acts.shape[0], alpha, dtype=np.float32))
        print(f"[tabpfn:L6] dataset {ds_idx + 1}/{N_DATASETS} done, acts={acts.shape}")

    return np.concatenate(pooled_acts, axis=0), np.concatenate(pooled_alpha, axis=0)


def _collect_tabicl_l1(
    dataset_specs: list[tuple[float, float]], device: str
) -> tuple[np.ndarray, np.ndarray]:
    pooled_acts: list[np.ndarray] = []
    pooled_alpha: list[np.ndarray] = []

    for ds_idx, (alpha, beta) in enumerate(dataset_specs):
        ds = generate_linear_data(
            alpha=alpha,
            beta=beta,
            n_train=N_TRAIN,
            n_test=N_TEST,
            random_seed=SEED + ds_idx,
        )
        model = _build_tabicl_model(device)
        model.fit(ds.X_train, ds.y_train)
        hooker = TabICLHookedModel(model)
        _, cache = hooker.forward_with_cache(ds.X_test)
        acts = np.asarray(
            hooker.get_layer_activations(cache, TABICL_LAYER_L1), dtype=np.float32
        )

        pooled_acts.append(acts)
        pooled_alpha.append(np.full(acts.shape[0], alpha, dtype=np.float32))
        print(f"[tabicl:L1] dataset {ds_idx + 1}/{N_DATASETS} done, acts={acts.shape}")

    return np.concatenate(pooled_acts, axis=0), np.concatenate(pooled_alpha, axis=0)


def _train_topk_sae(
    activations: np.ndarray,
    alpha_targets: np.ndarray,
    input_dim: int,
    device: str,
) -> tuple[TabPFNSparseAutoencoder, dict[str, float]]:
    hidden_dim = input_dim * EXPANSION_FACTOR
    topk_k = max(1, hidden_dim // 16)

    sae = TabPFNSparseAutoencoder(
        input_dim=input_dim,
        expansion_factor=EXPANSION_FACTOR,
        activation="topk",
        topk_k=topk_k,
    ).to(torch.device(device))
    trainer = SAETrainer(sae, lr=LR, l1_coeff=L1_COEFF)
    history = trainer.train(
        activations=activations,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=False,
    )

    with torch.no_grad():
        x = torch.from_numpy(np.asarray(activations, dtype=np.float32)).to(
            torch.device(device)
        )
        recon, feats = sae(x)

    recon_np = recon.detach().cpu().numpy()
    feats_np = feats.detach().cpu().numpy()

    max_r_alpha = 0.0
    for feat_idx in range(feats_np.shape[1]):
        max_r_alpha = max(
            max_r_alpha, _safe_abs_pearson(feats_np[:, feat_idx], alpha_targets)
        )

    active_mask = np.abs(feats_np) > 1e-8
    activation_counts = active_mask.sum(axis=0)
    dead_feature_pct = 100.0 * float((activation_counts == 0).mean())
    l0_sparsity = float(active_mask.sum(axis=1).mean())

    metrics = {
        "reconstruction_r2": _r2_score(activations, recon_np),
        "max_r_alpha": float(max_r_alpha),
        "dead_feature_pct": dead_feature_pct,
        "l0_sparsity": l0_sparsity,
        "history_final_reconstruction_r2": float(history["reconstruction_r2"][-1]),
        "history_final_sparsity": float(history["sparsity"][-1]),
        "topk_k": float(topk_k),
        "n_samples": float(activations.shape[0]),
    }
    return sae, metrics


def _usae_reconstruct_r2(
    sae: TabPFNSparseAutoencoder, activations: np.ndarray
) -> float:
    sae.eval()
    device = next(sae.parameters()).device
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(activations, dtype=np.float32)).to(device)
        z = sae.encode(x)
        recon = sae.decode(z)
    return _r2_score(activations, recon.detach().cpu().numpy())


def _adapt_activations_dim(activations: np.ndarray, target_dim: int) -> np.ndarray:
    acts = np.asarray(activations, dtype=np.float32)
    src_dim = int(acts.shape[1])
    if src_dim == target_dim:
        return acts
    if src_dim > target_dim:
        return acts[:, :target_dim]

    pad_width = target_dim - src_dim
    return np.pad(acts, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)


def _usae_reconstruct_r2_cross(
    sae: TabPFNSparseAutoencoder,
    source_activations: np.ndarray,
) -> float:
    bridged = _adapt_activations_dim(source_activations, sae.input_dim)
    return _usae_reconstruct_r2(sae, bridged)


def main() -> int:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = _resolve_cuda_device()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    specs = _dataset_specs()
    print(
        f"QUICK_RUN={QUICK_RUN}, seed={SEED}, n_datasets={len(specs)}, "
        f"n_train={N_TRAIN}, n_test={N_TEST}, device={device}"
    )

    tabdpt_l1_acts, tabdpt_l1_alpha = _collect_tabdpt_activations(
        specs, TABDPT_LAYER_L1, device
    )
    tabdpt_l8_acts, tabdpt_l8_alpha = _collect_tabdpt_activations(
        specs, TABDPT_LAYER_L8, device
    )
    tabpfn_l6_acts, tabpfn_l6_alpha = _collect_tabpfn_l6(specs, device)
    tabicl_l1_acts, tabicl_l1_alpha = _collect_tabicl_l1(specs, device)

    print("Training SAE on TabDPT L1 activations...")
    sae_tabdpt_l1, tabdpt_l1_metrics = _train_topk_sae(
        activations=tabdpt_l1_acts,
        alpha_targets=tabdpt_l1_alpha,
        input_dim=TABDPT_HIDDEN_DIM,
        device=device,
    )

    print("Training SAE on TabDPT L8 activations...")
    _sae_tabdpt_l8, tabdpt_l8_metrics = _train_topk_sae(
        activations=tabdpt_l8_acts,
        alpha_targets=tabdpt_l8_alpha,
        input_dim=TABDPT_HIDDEN_DIM,
        device=device,
    )

    print("Training baseline SAE on TabPFN L6 activations...")
    sae_tabpfn_l6, _ = _train_topk_sae(
        activations=tabpfn_l6_acts,
        alpha_targets=tabpfn_l6_alpha,
        input_dim=tabpfn_l6_acts.shape[1],
        device=device,
    )

    print("Training baseline SAE on TabICL L1 activations...")
    sae_tabicl_l1, _ = _train_topk_sae(
        activations=tabicl_l1_acts,
        alpha_targets=tabicl_l1_alpha,
        input_dim=tabicl_l1_acts.shape[1],
        device=device,
    )

    usae_results: dict[str, Any] = {
        "tabdpt_l1_train": {
            "self_tabdpt_l1_r2": _usae_reconstruct_r2(sae_tabdpt_l1, tabdpt_l1_acts),
            "to_tabpfn_l6_r2": _usae_reconstruct_r2_cross(
                sae_tabdpt_l1, tabpfn_l6_acts
            ),
            "to_tabicl_l1_r2": _usae_reconstruct_r2_cross(
                sae_tabdpt_l1, tabicl_l1_acts
            ),
        },
        "tabpfn_l6_train": {
            "self_tabpfn_l6_r2": _usae_reconstruct_r2(sae_tabpfn_l6, tabpfn_l6_acts),
            "to_tabdpt_l1_r2": _usae_reconstruct_r2_cross(
                sae_tabpfn_l6, tabdpt_l1_acts
            ),
        },
        "tabicl_l1_train": {
            "self_tabicl_l1_r2": _usae_reconstruct_r2(sae_tabicl_l1, tabicl_l1_acts),
            "to_tabdpt_l1_r2": _usae_reconstruct_r2_cross(
                sae_tabicl_l1, tabdpt_l1_acts
            ),
        },
    }

    payload = {
        "meta": {
            "seed": SEED,
            "quick_run": QUICK_RUN,
            "device": device,
            "n_datasets": len(specs),
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "dataset_specs": [[a, b] for a, b in specs],
            "sae": {
                "architecture": "TabPFNSparseAutoencoder",
                "activation": "topk",
                "expansion_factor": EXPANSION_FACTOR,
                "epochs": EPOCHS,
                "learning_rate": LR,
                "l1_coeff": L1_COEFF,
                "batch_size": BATCH_SIZE,
            },
            "usae_dim_bridge": "truncate if source_dim>target_dim; zero-pad if source_dim<target_dim",
            "layer_mapping": {
                "tabdpt_l1": TABDPT_LAYER_L1,
                "tabdpt_l8": TABDPT_LAYER_L8,
                "tabpfn_l6": TABPFN_LAYER_L6,
                "tabicl_l1": TABICL_LAYER_L1,
            },
        },
        "tabdpt_sae": {
            "tabdpt_l1": tabdpt_l1_metrics,
            "tabdpt_l8": tabdpt_l8_metrics,
        },
        "usae_cross_model": usae_results,
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
