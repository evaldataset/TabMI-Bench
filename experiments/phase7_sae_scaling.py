# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportExplicitAny=false, reportImplicitStringConcatenation=false
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
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


RESULTS_DIR = ROOT / "results" / "phase7"
JSON_PATH = RESULTS_DIR / f"sae_scaling_seed{cfg.SEED}.json"

EXPANSION_FACTORS_FULL = [4, 16, 32, 64, 128, 256]
EXPANSION_FACTORS_QUICK = [4, 256]
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
L1_COEFF = 1e-3
N_DATASETS = 10
N_TRAIN = 50
N_TEST = 50


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


def _resolve_cuda_device() -> str:
    env_device = os.environ.get("DEVICE")
    device = env_device.strip() if env_device else cfg.DEVICE
    if not device.startswith("cuda"):
        raise RuntimeError(
            f"GPU is required for phase7 SAE scaling. Set DEVICE=cuda:0 (current: {device})."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is False."
        )
    return device


def _build_dataset_specs() -> list[tuple[float, float]]:
    rng = np.random.default_rng(cfg.SEED)
    specs: list[tuple[float, float]] = []
    for _ in range(N_DATASETS):
        alpha = float(rng.uniform(0.5, 5.0))
        beta = float(rng.uniform(0.5, 5.0))
        specs.append((alpha, beta))
    return specs


def _collect_activations(
    model_name: str,
    layer_idx: int,
    hidden_dim: int,
    dataset_specs: list[tuple[float, float]],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    all_acts: list[np.ndarray] = []
    all_alpha: list[np.ndarray] = []

    if model_name == "tabpfn":
        model = TabPFNRegressor(device=device, model_path="tabpfn-v2-regressor.ckpt")
    elif model_name == "tabicl":
        model = TabICLRegressor(device=device, random_state=cfg.SEED)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    for ds_idx, (alpha, beta) in enumerate(dataset_specs):
        ds = generate_linear_data(
            alpha=alpha,
            beta=beta,
            n_train=N_TRAIN,
            n_test=N_TEST,
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
    model_name: str,
    device: str,
) -> dict[str, float]:
    hidden_dim = input_dim * expansion
    topk_k = max(1, hidden_dim // 16)
    fallback_batches = [BATCH_SIZE]
    if model_name == "tabicl" and expansion == 256:
        fallback_batches = [64, 32, 16]

    device_obj = torch.device(device)

    for batch_size in fallback_batches:
        try:
            torch.manual_seed(cfg.SEED)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_obj)

            sae = TabPFNSparseAutoencoder(
                input_dim=input_dim,
                expansion_factor=expansion,
                activation="topk",
                topk_k=topk_k,
            ).to(device_obj)
            trainer = SAETrainer(sae, lr=LR, l1_coeff=L1_COEFF)

            start = time.perf_counter()
            history = trainer.train(
                activations=activations,
                epochs=EPOCHS,
                batch_size=batch_size,
                verbose=False,
            )
            training_time_seconds = float(time.perf_counter() - start)

            with torch.no_grad():
                act_tensor = torch.from_numpy(
                    np.asarray(activations, dtype=np.float32)
                ).to(device_obj)
                _recon, feats = sae(act_tensor)

            feats_np = feats.detach().cpu().numpy()

            max_r_alpha = 0.0
            for feat_idx in range(feats_np.shape[1]):
                max_r_alpha = max(
                    max_r_alpha,
                    _safe_abs_pearson(feats_np[:, feat_idx], alpha_targets),
                )

            active_mask = np.abs(feats_np) > 1e-8
            activation_counts = active_mask.sum(axis=0)
            dead_feature_pct = 100.0 * float((activation_counts == 0).mean())
            l0_sparsity = float(active_mask.sum(axis=1).mean())
            gpu_memory_gb = float(torch.cuda.max_memory_allocated(device_obj) / 1e9)

            return {
                "reconstruction_r2": float(history["reconstruction_r2"][-1]),
                "max_r_alpha": float(max_r_alpha),
                "dead_feature_pct": float(dead_feature_pct),
                "l0_sparsity": float(l0_sparsity),
                "training_time_seconds": training_time_seconds,
                "gpu_memory_gb": gpu_memory_gb,
                "batch_size_used": float(batch_size),
                "topk_k": float(topk_k),
            }
        except RuntimeError as err:
            message = str(err).lower()
            is_oom = (
                "out of memory" in message or "cuda error: out of memory" in message
            )
            is_last_attempt = batch_size == fallback_batches[-1]
            if not is_oom or is_last_attempt:
                raise
            print(
                f"OOM at model={model_name}, expansion={expansion}, batch_size={batch_size}; retrying "
                f"with batch_size={fallback_batches[fallback_batches.index(batch_size) + 1]}"
            )
            torch.cuda.empty_cache()

    raise RuntimeError("Unreachable: fallback loop should return or raise.")


def main() -> int:
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    device = _resolve_cuda_device()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    expansions = EXPANSION_FACTORS_QUICK if cfg.QUICK_RUN else EXPANSION_FACTORS_FULL
    if cfg.QUICK_RUN:
        model_specs: dict[str, dict[str, int | str]] = {
            "tabpfn_l6": {"model": "tabpfn", "layer": 6, "dim": 192},
            "tabpfn_l8": {"model": "tabpfn", "layer": 8, "dim": 192},
        }
    else:
        model_specs = {
            "tabpfn_l6": {"model": "tabpfn", "layer": 6, "dim": 192},
            "tabpfn_l8": {"model": "tabpfn", "layer": 8, "dim": 192},
            "tabicl_l1": {"model": "tabicl", "layer": 1, "dim": 512},
        }

    dataset_specs = _build_dataset_specs()
    results: dict[str, object] = {}

    for target_name, spec in model_specs.items():
        model_name = str(spec["model"])
        layer_idx = int(spec["layer"])
        hidden_dim = int(spec["dim"])

        print(
            f"[{target_name}] collecting activations "
            f"(model={model_name}, layer={layer_idx}, dim={hidden_dim}, n_datasets={len(dataset_specs)})"
        )
        activations, alpha_targets = _collect_activations(
            model_name=model_name,
            layer_idx=layer_idx,
            hidden_dim=hidden_dim,
            dataset_specs=dataset_specs,
            device=device,
        )
        print(f"[{target_name}] pooled activations: {activations.shape}")

        target_results: dict[str, object] = {}
        for expansion in expansions:
            print(f"[{target_name}] training SAE topk x{expansion}...")
            metrics = _train_sae(
                activations=activations,
                alpha_targets=alpha_targets,
                input_dim=hidden_dim,
                expansion=expansion,
                model_name=model_name,
                device=device,
            )
            target_results[str(expansion)] = metrics
            print(
                f"  r2={metrics['reconstruction_r2']:.4f}, max|r_alpha|={metrics['max_r_alpha']:.4f}, "
                f"dead%={metrics['dead_feature_pct']:.2f}, l0={metrics['l0_sparsity']:.2f}, "
                f"time={metrics['training_time_seconds']:.1f}s, gpu_mem={metrics['gpu_memory_gb']:.2f}GB"
            )

        results[target_name] = target_results

    payload = {
        "quick_run": bool(cfg.QUICK_RUN),
        "seed": int(cfg.SEED),
        "device": device,
        "expansion_factors": expansions,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "l1_coeff": L1_COEFF,
        "dataset_counts": {k: N_DATASETS for k in model_specs.keys()},
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "model_specs": model_specs,
        "activation": "topk",
        "topk_k_formula": "hidden_dim//16",
        "results": results,
    }

    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
