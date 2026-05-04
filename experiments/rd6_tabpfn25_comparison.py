# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportUnusedParameter=false
"""M28: TabPFN v2 vs v2.5 Internal Representation Comparison.

Compares the mechanistic interpretability profiles of TabPFN v2 (12 layers,
6 heads, nhid_factor=4) with TabPFN v2.5 (18 layers, 3 heads, nhid_factor=2,
64 thinking tokens).

Analyses:
  1. Coefficient probing (α, β) across all layers — where is the information?
  2. Intermediary probing (a·b) — where is intermediate computation?
  3. Logit lens — where does the output prediction form?
  4. CKA — intra-model layer similarity + cross-model CKA
  5. Architecture summary — thinking tokens, parameter counts, key diffs
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from rd5_config import cfg
from src.data.synthetic_generator import generate_linear_data
from src.hooks.tabpfn_hooker import TabPFNHookedModel
from src.probing.linear_probe import probe_layer

RESULTS_DIR = ROOT / "results" / "rd6" / "tabpfn25_comparison"

# Model paths
V2_CKPT = str(ROOT / "tabpfn-v2-regressor.ckpt")
V25_CKPT = str(ROOT / "tabpfn-v2.5-regressor-v2.5_default.ckpt")


@dataclass
class DatasetSpec:
    alpha: float
    beta: float
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def generate_datasets(coef_pairs: list[tuple[int, int]]) -> list[DatasetSpec]:
    """Generate synthetic linear datasets for coefficient probing."""
    rng = np.random.default_rng(cfg.SEED)
    x1_train = rng.normal(size=cfg.N_TRAIN).astype(np.float32)
    x2_train = rng.normal(size=cfg.N_TRAIN).astype(np.float32)
    x1_test = rng.normal(size=cfg.N_TEST).astype(np.float32)
    x2_test = rng.normal(size=cfg.N_TEST).astype(np.float32)

    X_train = np.column_stack([x1_train, x2_train]).astype(np.float32)
    X_test = np.column_stack([x1_test, x2_test]).astype(np.float32)

    datasets: list[DatasetSpec] = []
    for alpha, beta in coef_pairs:
        y_train = (alpha * x1_train + beta * x2_train).astype(np.float32)
        y_test = (alpha * x1_test + beta * x2_test).astype(np.float32)
        datasets.append(
            DatasetSpec(
                alpha=float(alpha),
                beta=float(beta),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
        )
    return datasets


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _build_v2() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path=V2_CKPT)


def _build_v25() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path=V25_CKPT)


# ---------------------------------------------------------------------------
# Analysis 1: Coefficient Probing
# ---------------------------------------------------------------------------


def run_coefficient_probing(
    model_name: str,
    build_fn: Any,
    datasets: list[DatasetSpec],
) -> dict[str, Any]:
    """Run coefficient probing (α, β) across all layers."""
    pooled_activations: dict[int, list[np.ndarray]] = {}
    pooled_alphas: list[np.ndarray] = []
    pooled_betas: list[np.ndarray] = []
    layer_indices: list[int] | None = None

    for idx, ds in enumerate(datasets, start=1):
        model = build_fn()
        model.fit(ds.X_train, ds.y_train)
        hooker = TabPFNHookedModel(model)
        _, cache = hooker.forward_with_cache(ds.X_test)

        n_layers = hooker._n_layers
        if layer_indices is None:
            layer_indices = list(range(n_layers))
            pooled_activations = {li: [] for li in layer_indices}

        for li in layer_indices:
            act = np.asarray(hooker.get_test_label_token(cache, li), dtype=np.float32)
            pooled_activations[li].append(act)

        pooled_alphas.append(np.full(ds.X_test.shape[0], ds.alpha, dtype=np.float32))
        pooled_betas.append(np.full(ds.X_test.shape[0], ds.beta, dtype=np.float32))

        if idx % 5 == 0 or idx == len(datasets):
            print(f"  [{model_name}] probing dataset {idx}/{len(datasets)}")

    assert layer_indices is not None
    targets_alpha = np.concatenate(pooled_alphas)
    targets_beta = np.concatenate(pooled_betas)

    alpha_r2: list[float] = []
    beta_r2: list[float] = []
    for li in layer_indices:
        acts = np.vstack(pooled_activations[li])
        a = probe_layer(acts, targets_alpha, complexities=[0], random_seed=cfg.SEED)
        b = probe_layer(acts, targets_beta, complexities=[0], random_seed=cfg.SEED)
        alpha_r2.append(float(a[0]["r2"]))
        beta_r2.append(float(b[0]["r2"]))

    return {
        "alpha_r2_by_layer": alpha_r2,
        "beta_r2_by_layer": beta_r2,
        "n_layers": len(layer_indices),
        "peak_layer_alpha": int(np.argmax(alpha_r2)),
        "peak_layer_beta": int(np.argmax(beta_r2)),
    }


# ---------------------------------------------------------------------------
# Analysis 2: Intermediary Probing (a·b)
# ---------------------------------------------------------------------------


def run_intermediary_probing(
    model_name: str,
    build_fn: Any,
    n_datasets: int,
) -> dict[str, Any]:
    """Probe for intermediate computation a·b in z = a·b + c."""
    pooled_by_layer: dict[int, list[np.ndarray]] = {}
    pooled_targets: list[np.ndarray] = []
    layer_indices: list[int] | None = None

    rng = np.random.default_rng(cfg.SEED)

    for ds_idx in range(n_datasets):
        a = rng.uniform(1.0, 5.0)
        b = rng.uniform(1.0, 5.0)
        c = rng.uniform(-3.0, 3.0)
        x_train = rng.normal(size=(cfg.N_TRAIN, 2)).astype(np.float32)
        x_test = rng.normal(size=(cfg.N_TEST, 2)).astype(np.float32)
        y_train = (a * x_train[:, 0] * b * x_train[:, 1] + c).astype(np.float32)
        y_test = (a * x_test[:, 0] * b * x_test[:, 1] + c).astype(np.float32)
        target_ab = (a * x_test[:, 0] * b * x_test[:, 1]).astype(np.float32)

        model = build_fn()
        model.fit(x_train, y_train)
        hooker = TabPFNHookedModel(model)
        _, cache = hooker.forward_with_cache(x_test)

        n_layers = hooker._n_layers
        if layer_indices is None:
            layer_indices = list(range(n_layers))
            pooled_by_layer = {li: [] for li in layer_indices}

        for li in layer_indices:
            act = np.asarray(hooker.get_test_label_token(cache, li), dtype=np.float32)
            pooled_by_layer[li].append(act)

        pooled_targets.append(target_ab)

    assert layer_indices is not None
    all_targets = np.concatenate(pooled_targets)

    r2_by_layer: list[float] = []
    for li in layer_indices:
        acts = np.vstack(pooled_by_layer[li])
        result = probe_layer(acts, all_targets, complexities=[0], random_seed=cfg.SEED)
        r2_by_layer.append(float(result[0]["r2"]))

    return {
        "r2_by_layer": r2_by_layer,
        "n_layers": len(layer_indices),
        "peak_layer": int(np.argmax(r2_by_layer)),
    }


# ---------------------------------------------------------------------------
# Analysis 3: Logit Lens
# ---------------------------------------------------------------------------


def _decode_logits_to_prediction(
    model: TabPFNRegressor, logits: np.ndarray
) -> np.ndarray:
    """Decode raw logit lens output into scalar predictions via BarDistribution."""
    import torch
    logits_tensor = torch.from_numpy(logits.astype(np.float32, copy=False))
    if hasattr(model, "raw_space_bardist_"):
        pred_tensor = model.raw_space_bardist_.mean(logits_tensor)
    elif hasattr(model, "norm_bardist_"):
        pred_tensor = model.norm_bardist_.mean(logits_tensor)
    elif hasattr(model, "bardist_"):
        pred_tensor = model.bardist_.mean(logits_tensor)
    else:
        pred_tensor = logits_tensor.argmax(dim=-1).float()
    return pred_tensor.detach().cpu().numpy()


def run_logit_lens(
    model_name: str,
    build_fn: Any,
    n_datasets: int,
) -> dict[str, Any]:
    """Apply logit lens across all layers to see where prediction forms."""
    from sklearn.metrics import r2_score as _r2_score

    all_r2: list[list[float]] = []  # [dataset][layer]

    rng = np.random.default_rng(cfg.SEED)

    for ds_idx in range(n_datasets):
        alpha = rng.uniform(1.0, 5.0)
        beta = rng.uniform(1.0, 5.0)
        x_train = rng.normal(size=(cfg.N_TRAIN, 2)).astype(np.float32)
        x_test = rng.normal(size=(cfg.N_TEST, 2)).astype(np.float32)
        y_train = (alpha * x_train[:, 0] + beta * x_train[:, 1]).astype(np.float32)
        y_test = (alpha * x_test[:, 0] + beta * x_test[:, 1]).astype(np.float32)

        model = build_fn()
        model.fit(x_train, y_train)
        hooker = TabPFNHookedModel(model)
        pred, cache = hooker.forward_with_cache(x_test)

        layer_r2: list[float] = []
        for li in range(hooker._n_layers):
            logits = hooker.apply_logit_lens(cache, li)
            pred_from_lens = _decode_logits_to_prediction(model, logits)
            r2 = float(_r2_score(y_test, pred_from_lens))
            layer_r2.append(r2)

        all_r2.append(layer_r2)

    # Average across datasets
    avg_r2 = np.mean(all_r2, axis=0).tolist()
    return {
        "r2_by_layer": avg_r2,
        "n_layers": len(avg_r2),
        "peak_layer": int(np.argmax(avg_r2)),
    }



# ---------------------------------------------------------------------------
# Analysis 4: CKA Comparison
# ---------------------------------------------------------------------------


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered Kernel Alignment between two representation matrices."""
    n = X.shape[0]
    K_X = X @ X.T
    K_Y = Y @ Y.T
    H = np.eye(n) - np.ones((n, n)) / n
    K_X_c = H @ K_X @ H
    K_Y_c = H @ K_Y @ H
    hsic_xy = np.trace(K_X_c @ K_Y_c) / (n - 1) ** 2
    hsic_xx = np.trace(K_X_c @ K_X_c) / (n - 1) ** 2
    hsic_yy = np.trace(K_Y_c @ K_Y_c) / (n - 1) ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 1e-10 else 0.0


def run_cka_analysis(
    build_v2_fn: Any,
    build_v25_fn: Any,
    n_datasets: int,
) -> dict[str, Any]:
    """Compute intra-model CKA matrices and cross-model CKA."""
    v2_layers_pooled: dict[int, list[np.ndarray]] = {}
    v25_layers_pooled: dict[int, list[np.ndarray]] = {}
    v2_n_layers: int = 0
    v25_n_layers: int = 0

    for ds_idx in range(n_datasets):
        dataset = generate_linear_data(
            alpha=2.0,
            beta=3.0,
            n_train=cfg.N_TRAIN,
            n_test=cfg.N_TEST,
            random_seed=cfg.SEED + ds_idx,
        )

        # v2
        m2 = build_v2_fn()
        m2.fit(dataset.X_train, dataset.y_train)
        h2 = TabPFNHookedModel(m2)
        _, c2 = h2.forward_with_cache(dataset.X_test)
        v2_n_layers = h2._n_layers

        if ds_idx == 0:
            v2_layers_pooled = {i: [] for i in range(v2_n_layers)}

        for li in range(v2_n_layers):
            act = np.asarray(h2.get_test_label_token(c2, li), dtype=np.float64)
            v2_layers_pooled[li].append(act)

        # v2.5
        m25 = build_v25_fn()
        m25.fit(dataset.X_train, dataset.y_train)
        h25 = TabPFNHookedModel(m25)
        _, c25 = h25.forward_with_cache(dataset.X_test)
        v25_n_layers = h25._n_layers

        if ds_idx == 0:
            v25_layers_pooled = {i: [] for i in range(v25_n_layers)}

        for li in range(v25_n_layers):
            act = np.asarray(h25.get_test_label_token(c25, li), dtype=np.float64)
            v25_layers_pooled[li].append(act)

        print(f"  [CKA] dataset {ds_idx + 1}/{n_datasets}")

    # Stack activations
    v2_acts = [np.vstack(v2_layers_pooled[i]) for i in range(v2_n_layers)]
    v25_acts = [np.vstack(v25_layers_pooled[i]) for i in range(v25_n_layers)]

    # Intra-model CKA
    v2_cka = np.zeros((v2_n_layers, v2_n_layers))
    for i in range(v2_n_layers):
        for j in range(v2_n_layers):
            v2_cka[i, j] = compute_cka(v2_acts[i], v2_acts[j])

    v25_cka = np.zeros((v25_n_layers, v25_n_layers))
    for i in range(v25_n_layers):
        for j in range(v25_n_layers):
            v25_cka[i, j] = compute_cka(v25_acts[i], v25_acts[j])

    # Cross-model CKA: v2 layer i vs v2.5 layer j
    cross_cka = np.zeros((v2_n_layers, v25_n_layers))
    for i in range(v2_n_layers):
        for j in range(v25_n_layers):
            cross_cka[i, j] = compute_cka(v2_acts[i], v25_acts[j])

    # Adjacent CKA for block detection
    v2_adj = [float(v2_cka[i, i + 1]) for i in range(v2_n_layers - 1)]
    v25_adj = [float(v25_cka[i, i + 1]) for i in range(v25_n_layers - 1)]

    return {
        "v2_cka_matrix": v2_cka.tolist(),
        "v25_cka_matrix": v25_cka.tolist(),
        "cross_cka_matrix": cross_cka.tolist(),
        "v2_adjacent_cka": v2_adj,
        "v25_adjacent_cka": v25_adj,
        "v2_n_layers": v2_n_layers,
        "v25_n_layers": v25_n_layers,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_probing_comparison(
    v2_result: dict[str, Any],
    v25_result: dict[str, Any],
    title_prefix: str,
    ylabel: str,
    key: str,
    save_path: Path,
) -> None:
    """Plot v2 vs v2.5 probing profiles with normalized layer axis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw layer indices
    ax = axes[0]
    v2_vals = v2_result[key]
    v25_vals = v25_result[key]
    ax.plot(range(len(v2_vals)), v2_vals, "o-", lw=2, label=f"v2 ({len(v2_vals)}L)")
    ax.plot(
        range(len(v25_vals)), v25_vals, "s-", lw=2, label=f"v2.5 ({len(v25_vals)}L)"
    )
    ax.set_xlabel("Layer Index")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix} (raw layers)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: normalized layer position
    ax = axes[1]
    v2_x = np.linspace(0, 1, len(v2_vals))
    v25_x = np.linspace(0, 1, len(v25_vals))
    ax.plot(v2_x, v2_vals, "o-", lw=2, label=f"v2 ({len(v2_vals)}L)")
    ax.plot(v25_x, v25_vals, "s-", lw=2, label=f"v2.5 ({len(v25_vals)}L)")
    ax.set_xlabel("Normalized Layer Position")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix} (normalized)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cka_heatmaps(cka_results: dict[str, Any], save_path: Path) -> None:
    """Plot v2, v2.5 intra-CKA, and cross-model CKA heatmaps."""
    v2_cka = np.asarray(cka_results["v2_cka_matrix"])
    v25_cka = np.asarray(cka_results["v25_cka_matrix"])
    cross_cka = np.asarray(cka_results["cross_cka_matrix"])

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # v2 CKA
    im0 = axes[0].imshow(v2_cka, vmin=0, vmax=1, cmap="viridis")
    axes[0].set_title("TabPFN v2 CKA (12×12)")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Layer")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # v2.5 CKA
    im1 = axes[1].imshow(v25_cka, vmin=0, vmax=1, cmap="viridis")
    axes[1].set_title("TabPFN v2.5 CKA (18×18)")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Layer")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Cross-model CKA
    im2 = axes[2].imshow(cross_cka, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[2].set_title("Cross-Model CKA (v2×v2.5)")
    axes[2].set_xlabel("v2.5 Layer")
    axes[2].set_ylabel("v2 Layer")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_logit_lens(
    v2_result: dict[str, Any],
    v25_result: dict[str, Any],
    save_path: Path,
) -> None:
    """Plot logit lens R² comparison."""
    plot_probing_comparison(
        v2_result,
        v25_result,
        title_prefix="Logit Lens R²",
        ylabel="R²(predicted vs actual)",
        key="r2_by_layer",
        save_path=save_path,
    )


# ---------------------------------------------------------------------------
# Architecture Summary
# ---------------------------------------------------------------------------


def get_architecture_summary() -> dict[str, Any]:
    """Summarize key architectural differences between v2 and v2.5."""
    return {
        "v2": {
            "nlayers": 12,
            "nhead": 6,
            "d_k": 32,
            "emsize": 192,
            "nhid_factor": 4,
            "mlp_hidden": 768,
            "features_per_group": 2,
            "encoder_type": "linear",
            "thinking_tokens": 0,
            "decoder_hidden": 768,
            "total_params_state_dict": 83,
        },
        "v2.5": {
            "nlayers": 18,
            "nhead": 3,
            "d_k": 64,
            "emsize": 192,
            "nhid_factor": 2,
            "mlp_hidden": 384,
            "features_per_group": 3,
            "encoder_type": "mlp (1024 hidden, 2 layers)",
            "thinking_tokens": 64,
            "decoder_hidden": 384,
            "total_params_state_dict": 121,
        },
        "key_differences": [
            "v2.5 has 50% more layers (18 vs 12) but fewer heads (3 vs 6)",
            "v2.5 uses larger head dim (d_k=64 vs 32) but smaller MLP (2× vs 4×)",
            "v2.5 adds 64 'thinking tokens' — latent chain-of-thought rows",
            "v2.5 uses MLP encoder (1024 hidden) vs v2's linear encoder",
            "Same emsize=192, same output buckets=5000",
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    quick = cfg.QUICK_RUN
    print(f"M28: TabPFN v2 vs v2.5 Comparison")
    print(f"  QUICK_RUN={quick}, SEED={cfg.SEED}")

    # Dataset configuration
    if quick:
        coef_values = [1, 3, 5]
        n_intermediary_datasets = 3
        n_logit_datasets = 3
        n_cka_datasets = 3
    else:
        coef_values = [1, 2, 3, 4, 5]
        n_intermediary_datasets = 10
        n_logit_datasets = 10
        n_cka_datasets = 10

    coef_pairs = [(a, b) for a in coef_values for b in coef_values]
    datasets = generate_datasets(coef_pairs)
    print(f"  Coefficient datasets: {len(datasets)}")
    print(f"  Intermediary/Logit/CKA datasets: {n_intermediary_datasets}")

    # 1. Coefficient probing
    print("\n[1/5] Coefficient Probing...")
    v2_coef = run_coefficient_probing("v2", _build_v2, datasets)
    v25_coef = run_coefficient_probing("v2.5", _build_v25, datasets)
    print(
        f"  v2:  peak α=L{v2_coef['peak_layer_alpha']}, peak β=L{v2_coef['peak_layer_beta']}"
    )
    print(
        f"  v2.5: peak α=L{v25_coef['peak_layer_alpha']}, peak β=L{v25_coef['peak_layer_beta']}"
    )

    plot_probing_comparison(
        v2_coef,
        v25_coef,
        title_prefix="Coefficient α R²",
        ylabel="R²(α)",
        key="alpha_r2_by_layer",
        save_path=RESULTS_DIR / "coefficient_alpha_comparison.png",
    )
    plot_probing_comparison(
        v2_coef,
        v25_coef,
        title_prefix="Coefficient β R²",
        ylabel="R²(β)",
        key="beta_r2_by_layer",
        save_path=RESULTS_DIR / "coefficient_beta_comparison.png",
    )

    # 2. Intermediary probing
    print("\n[2/5] Intermediary Probing (a·b)...")
    v2_inter = run_intermediary_probing("v2", _build_v2, n_intermediary_datasets)
    v25_inter = run_intermediary_probing("v2.5", _build_v25, n_intermediary_datasets)
    print(f"  v2:  peak a·b at L{v2_inter['peak_layer']}")
    print(f"  v2.5: peak a·b at L{v25_inter['peak_layer']}")

    plot_probing_comparison(
        v2_inter,
        v25_inter,
        title_prefix="Intermediary (a·b) R²",
        ylabel="R²(a·b)",
        key="r2_by_layer",
        save_path=RESULTS_DIR / "intermediary_comparison.png",
    )

    # 3. Logit lens
    print("\n[3/5] Logit Lens...")
    v2_ll = run_logit_lens("v2", _build_v2, n_logit_datasets)
    v25_ll = run_logit_lens("v2.5", _build_v25, n_logit_datasets)
    print(f"  v2:  prediction forms at L{v2_ll['peak_layer']}")
    print(f"  v2.5: prediction forms at L{v25_ll['peak_layer']}")

    plot_logit_lens(v2_ll, v25_ll, RESULTS_DIR / "logit_lens_comparison.png")

    # 4. CKA
    print("\n[4/5] CKA Analysis...")
    cka_results = run_cka_analysis(_build_v2, _build_v25, n_cka_datasets)
    plot_cka_heatmaps(cka_results, RESULTS_DIR / "cka_heatmaps.png")

    # 5. Architecture summary
    print("\n[5/5] Architecture Summary...")
    arch = get_architecture_summary()

    elapsed = time.time() - t0

    # Assemble results
    payload: dict[str, Any] = {
        "experiment": "M28: TabPFN v2 vs v2.5 Comparison",
        "quick_run": bool(quick),
        "seed": int(cfg.SEED),
        "n_train": int(cfg.N_TRAIN),
        "n_test": int(cfg.N_TEST),
        "elapsed_seconds": round(elapsed, 1),
        "coefficient_probing": {
            "v2": v2_coef,
            "v2.5": v25_coef,
        },
        "intermediary_probing": {
            "v2": v2_inter,
            "v2.5": v25_inter,
        },
        "logit_lens": {
            "v2": v2_ll,
            "v2.5": v25_ll,
        },
        "cka": cka_results,
        "architecture": arch,
        "artifacts": [
            "coefficient_alpha_comparison.png",
            "coefficient_beta_comparison.png",
            "intermediary_comparison.png",
            "logit_lens_comparison.png",
            "cka_heatmaps.png",
        ],
    }

    json_path = RESULTS_DIR / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"M28 complete in {elapsed:.1f}s")
    print(f"Saved: {json_path}")
    print(f"Plots: {len(payload['artifacts'])} PNG files")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
