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
from iltm import iLTMRegressor
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportImplicitStringConcatenation=false

from rd5_config import cfg
from src.data.synthetic_generator import generate_linear_data
from src.hooks.activation_patcher import TabPFNActivationPatcher
from src.hooks.iltm_hooker import iLTMHookedModel
from src.hooks.steering_vector import TabPFNSteeringVector
from src.hooks.tabicl_hooker import TabICLHookedModel
from src.hooks.tabicl_steering import TabICLSteeringVector
from src.hooks.tabpfn_hooker import TabPFNHookedModel
from src.probing.linear_probe import probe_layer

RESULTS_DIR = ROOT / "results" / "neurips"
RESULTS_JSON_PATH = RESULTS_DIR / "c1c2_baselines.json"
FIGURE_PATH = RESULTS_DIR / "c1c2_patching_comparison.png"

LAMBDA_VALUES = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
STEERING_LAYERS = {"tabpfn": 6, "tabicl": 5}


def _build_model(model_name: str) -> Any:
    if model_name == "tabpfn":
        return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")
    if model_name == "tabicl":
        return TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)
    if model_name == "iltm":
        return iLTMRegressor(device="cpu", n_ensemble=1, seed=cfg.SEED)
    raise ValueError(f"Unknown model_name: {model_name}")


def _select_coef_pairs(model_name: str) -> list[tuple[int, int]]:
    pairs = cfg.coef_pairs(model_name)
    if not cfg.QUICK_RUN:
        return pairs
    if model_name == "iltm":
        return pairs[:2]
    return pairs[:3]


def _collect_layer_activations_for_alpha(
    model_name: str,
    shuffle_labels: bool,
) -> tuple[list[int], dict[int, np.ndarray], np.ndarray]:
    coef_pairs = _select_coef_pairs(model_name)
    pooled_by_layer: dict[int, list[np.ndarray]] = {}
    pooled_alpha: list[np.ndarray] = []
    layer_indices: list[int] | None = None
    rng = np.random.default_rng(cfg.SEED + (101 if shuffle_labels else 0))

    x_rng = np.random.default_rng(cfg.SEED)
    x1_train = x_rng.normal(size=cfg.N_TRAIN).astype(np.float32)
    x2_train = x_rng.normal(size=cfg.N_TRAIN).astype(np.float32)
    x1_test = x_rng.normal(size=cfg.N_TEST).astype(np.float32)
    x2_test = x_rng.normal(size=cfg.N_TEST).astype(np.float32)
    X_train = np.column_stack([x1_train, x2_train]).astype(np.float32)
    X_test = np.column_stack([x1_test, x2_test]).astype(np.float32)

    for alpha, beta in coef_pairs:
        y_train = (float(alpha) * x1_train + float(beta) * x2_train).astype(np.float32)
        if shuffle_labels:
            y_train = y_train[rng.permutation(y_train.shape[0])]

        model = _build_model(model_name)
        model.fit(X_train, y_train)

        get_activation: Any
        if model_name == "tabpfn":
            tabpfn_hooker: Any = TabPFNHookedModel(model)
            _, cache = tabpfn_hooker.forward_with_cache(X_test)
            get_activation = lambda layer_idx: np.asarray(  # noqa: E731
                tabpfn_hooker.get_test_label_token(cache, layer_idx), dtype=np.float32
            )
            n_layers_current = int(getattr(tabpfn_hooker, "_n_layers", 12))
        elif model_name == "tabicl":
            tabicl_hooker: Any = TabICLHookedModel(model)
            _, cache = tabicl_hooker.forward_with_cache(X_test)
            get_activation = lambda layer_idx: np.asarray(  # noqa: E731
                tabicl_hooker.get_layer_activations(cache, layer_idx), dtype=np.float32
            )
            n_layers_current = int(tabicl_hooker.num_layers)
        else:
            iltm_hooker: Any = iLTMHookedModel(model)
            _, cache = iltm_hooker.forward_with_cache(X_test)
            get_activation = lambda layer_idx: np.asarray(  # noqa: E731
                iltm_hooker.get_layer_activations(cache, layer_idx), dtype=np.float32
            )
            n_layers_current = int(iltm_hooker.num_layers)

        if layer_indices is None:
            if model_name == "iltm":
                layer_indices = cfg.layer_indices("iltm")
            else:
                n_layers = int(n_layers_current)
                layer_indices = list(range(n_layers))
            pooled_by_layer = {idx: [] for idx in layer_indices}

        assert layer_indices is not None
        for layer_idx in layer_indices:
            act = get_activation(layer_idx)
            pooled_by_layer[layer_idx].append(act)

        pooled_alpha.append(np.full(X_test.shape[0], float(alpha), dtype=np.float32))

    if layer_indices is None:
        raise RuntimeError(f"No layers collected for model={model_name}")

    merged_by_layer = {
        layer_idx: np.vstack(layer_chunks)
        for layer_idx, layer_chunks in pooled_by_layer.items()
    }
    targets_alpha = np.concatenate(pooled_alpha)
    return layer_indices, merged_by_layer, targets_alpha


def _probe_r2_for_targets(
    layer_indices: list[int],
    activations_by_layer: dict[int, np.ndarray],
    targets: np.ndarray,
) -> list[float]:
    r2_by_layer: list[float] = []
    for layer_idx in layer_indices:
        probe = probe_layer(
            activations=activations_by_layer[layer_idx],
            targets=targets,
            complexities=[0],
            random_seed=cfg.SEED,
        )
        r2_by_layer.append(float(probe[0]["r2"]))
    return r2_by_layer


def _run_c1_probings() -> dict[str, Any]:
    rng = np.random.default_rng(cfg.SEED)
    out: dict[str, Any] = {}
    for model_name in ["tabpfn", "tabicl", "iltm"]:
        layers, acts_clean, alpha_targets = _collect_layer_activations_for_alpha(
            model_name=model_name,
            shuffle_labels=False,
        )
        real_r2 = _probe_r2_for_targets(layers, acts_clean, alpha_targets)
        random_targets = rng.normal(0.0, 1.0, size=alpha_targets.shape).astype(
            np.float32
        )
        random_target_r2 = _probe_r2_for_targets(layers, acts_clean, random_targets)

        _, acts_shuffled, _ = _collect_layer_activations_for_alpha(
            model_name=model_name,
            shuffle_labels=True,
        )
        shuffled_label_r2 = _probe_r2_for_targets(layers, acts_shuffled, alpha_targets)

        out[model_name] = {
            "layers": layers,
            "real_alpha_r2": real_r2,
            "random_target_r2": random_target_r2,
            "shuffled_label_r2": shuffled_label_r2,
            "mean_r2": {
                "real": float(np.mean(real_r2)),
                "random_target": float(np.mean(random_target_r2)),
                "shuffled_label": float(np.mean(shuffled_label_r2)),
            },
            "peak_layer": {
                "real": int(np.argmax(np.asarray(real_r2))),
                "random_target": int(np.argmax(np.asarray(random_target_r2))),
                "shuffled_label": int(np.argmax(np.asarray(shuffled_label_r2))),
            },
        }
    return out


def _corr_abs(lambdas: list[float], shifts: np.ndarray) -> float:
    x = np.asarray(lambdas, dtype=np.float64)
    if np.allclose(np.std(shifts), 0.0):
        return 0.0
    r = np.corrcoef(x, shifts)[0, 1]
    if np.isnan(r):
        return 0.0
    return float(abs(r))


def _compute_shift_metrics(mean_preds: dict[float, float]) -> dict[str, Any]:
    baseline = float(mean_preds[0.0])
    shifts = np.asarray(
        [float(mean_preds[lam]) - baseline for lam in LAMBDA_VALUES], dtype=np.float64
    )
    return {
        "baseline_mean_pred": baseline,
        "shifts": [float(v) for v in shifts],
        "abs_pearson_r_lambda_shift": _corr_abs(LAMBDA_VALUES, shifts),
    }


def _run_c1_random_vector_steering() -> dict[str, Any]:
    ds_high = generate_linear_data(
        alpha=5.0,
        beta=1.0,
        n_train=cfg.N_TRAIN,
        n_test=cfg.N_TEST,
        random_seed=cfg.SEED,
    )
    ds_low = generate_linear_data(
        alpha=1.0,
        beta=1.0,
        n_train=cfg.N_TRAIN,
        n_test=cfg.N_TEST,
        random_seed=cfg.SEED,
    )

    out: dict[str, Any] = {}
    rng = np.random.default_rng(cfg.SEED + 2026)

    tabpfn_model = TabPFNRegressor(
        device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt"
    )
    tabpfn_steerer = TabPFNSteeringVector(tabpfn_model)
    tabpfn_layer = STEERING_LAYERS["tabpfn"]
    n_val = max(int(len(ds_high.y_train) * 0.2), 2)
    X_val = ds_high.X_train[:n_val]
    tabpfn_direction = tabpfn_steerer.extract_direction(
        ds_high.X_train[n_val:],
        ds_high.y_train[n_val:],
        ds_low.X_train[n_val:],
        ds_low.y_train[n_val:],
        ds_high.X_test,
        layer=tabpfn_layer,
        X_val=X_val,
    )
    tabpfn_real = tabpfn_steerer.sweep_lambda(
        ds_high.X_test,
        layer=tabpfn_layer,
        direction=tabpfn_direction,
        lambdas=LAMBDA_VALUES,
    )
    tabpfn_random_direction = rng.normal(size=tabpfn_direction.shape)
    tabpfn_random_direction = tabpfn_random_direction / np.linalg.norm(
        tabpfn_random_direction
    )
    tabpfn_rand = tabpfn_steerer.sweep_lambda(
        ds_high.X_test,
        layer=tabpfn_layer,
        direction=tabpfn_random_direction,
        lambdas=LAMBDA_VALUES,
    )

    out["tabpfn"] = {
        "layer": tabpfn_layer,
        "real": _compute_shift_metrics(tabpfn_real["mean_preds"]),
        "random": _compute_shift_metrics(tabpfn_rand["mean_preds"]),
    }

    tabicl_model = TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)
    tabicl_steerer = TabICLSteeringVector(tabicl_model)
    tabicl_layer = STEERING_LAYERS["tabicl"]
    tabicl_direction = tabicl_steerer.extract_direction(
        ds_high.X_train[n_val:],
        ds_high.y_train[n_val:],
        ds_low.X_train[n_val:],
        ds_low.y_train[n_val:],
        ds_high.X_test,
        layer=tabicl_layer,
        X_val=X_val,
    )
    tabicl_real = tabicl_steerer.sweep_lambda(
        ds_high.X_test,
        layer=tabicl_layer,
        direction=tabicl_direction,
        lambdas=LAMBDA_VALUES,
    )
    tabicl_random_direction = rng.normal(size=tabicl_direction.shape)
    tabicl_random_direction = tabicl_random_direction / np.linalg.norm(
        tabicl_random_direction
    )
    tabicl_rand = tabicl_steerer.sweep_lambda(
        ds_high.X_test,
        layer=tabicl_layer,
        direction=tabicl_random_direction,
        lambdas=LAMBDA_VALUES,
    )

    out["tabicl"] = {
        "layer": tabicl_layer,
        "real": _compute_shift_metrics(tabicl_real["mean_preds"]),
        "random": _compute_shift_metrics(tabicl_rand["mean_preds"]),
    }
    return out


def _generate_patching_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.SEED)
    n_total = cfg.N_TRAIN + cfg.N_TEST
    X = rng.uniform(0.5, 3.0, (n_total, 3)).astype(np.float32)
    X_train, X_test = X[: cfg.N_TRAIN], X[cfg.N_TRAIN :]
    y_train = (X_train[:, 0] * X_train[:, 1] + X_train[:, 2]).astype(np.float32)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train}


def _make_noised_input(X_test: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(cfg.SEED + 77)
    scale = np.std(X_test, axis=0, keepdims=True)
    noise = rng.normal(0.0, 0.5 * scale, size=X_test.shape).astype(np.float32)
    return (X_test + noise).astype(np.float32)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _recovery(clean: np.ndarray, noised: np.ndarray, patched: np.ndarray) -> float:
    denom = _mse(noised, clean)
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - _mse(patched, clean) / denom)


def _tabicl_standard_patching(
    model: TabICLRegressor,
    X_test: np.ndarray,
    X_test_noised: np.ndarray,
) -> list[float]:
    hooker = TabICLHookedModel(model)
    preds_clean, cache_clean = hooker.forward_with_cache(X_test)
    preds_clean = np.asarray(preds_clean, dtype=np.float32)
    preds_noised = np.asarray(model.predict(X_test_noised), dtype=np.float32)
    blocks: Any = model.model_.icl_predictor.tf_icl.blocks
    recoveries: list[float] = []

    for layer_idx in range(len(blocks)):
        clean_activation = cache_clean["layers"][layer_idx]

        def patch_hook(
            module: torch.nn.Module, inputs: Any, output: torch.Tensor
        ) -> torch.Tensor:  # noqa: ARG001
            return clean_activation.to(output.device)

        handle = blocks[layer_idx].register_forward_hook(patch_hook)
        try:
            preds_patched = np.asarray(model.predict(X_test_noised), dtype=np.float32)
        finally:
            handle.remove()
        recoveries.append(_recovery(preds_clean, preds_noised, preds_patched))

    return recoveries


def _tabpfn_standard_patching(
    model: TabPFNRegressor,
    X_test: np.ndarray,
    X_test_noised: np.ndarray,
) -> list[float]:
    patcher = TabPFNActivationPatcher(model)
    preds_clean, clean_cache = patcher.run_with_cache(X_test)
    preds_clean = np.asarray(preds_clean, dtype=np.float32)
    preds_noised = np.asarray(model.predict(X_test_noised), dtype=np.float32)
    encoder: Any = model.model_.transformer_encoder
    layers = encoder.layers
    n_layers = len(layers)
    recoveries: list[float] = []
    for layer_idx in range(n_layers):
        preds_patched = np.asarray(
            patcher.patched_run(X_test_noised, clean_cache, patch_layer=layer_idx),
            dtype=np.float32,
        )
        recoveries.append(_recovery(preds_clean, preds_noised, preds_patched))
    return recoveries


def _tabicl_noising_tracing(model: TabICLRegressor, X_test: np.ndarray) -> list[float]:
    preds_clean = np.asarray(model.predict(X_test), dtype=np.float32)
    blocks: Any = model.model_.icl_predictor.tf_icl.blocks
    mse_by_layer: list[float] = []

    for layer_idx, block in enumerate(blocks):
        stds: list[float] = []

        def stats_hook(
            module: torch.nn.Module, inputs: Any, output: torch.Tensor
        ) -> None:  # noqa: ARG001
            stds.append(float(output.std().item()))

        stats_handle = block.register_forward_hook(stats_hook)
        try:
            _ = model.predict(X_test)
        finally:
            stats_handle.remove()

        act_std = float(np.mean(stds)) if stds else 1.0
        rng = np.random.default_rng(cfg.SEED + 2000 + layer_idx)

        def noise_hook(
            module: torch.nn.Module, inputs: Any, output: torch.Tensor
        ) -> torch.Tensor:  # noqa: ARG001
            noise = torch.from_numpy(
                rng.normal(0.0, 0.5 * act_std, size=tuple(output.shape)).astype(
                    np.float32
                )
            ).to(output.device)
            return output + noise

        noise_handle = block.register_forward_hook(noise_hook)
        try:
            preds_noised = np.asarray(model.predict(X_test), dtype=np.float32)
        finally:
            noise_handle.remove()
        mse_by_layer.append(_mse(preds_noised, preds_clean))

    max_mse = max(mse_by_layer) if max(mse_by_layer) > 0.0 else 1.0
    return [float(v / max_mse) for v in mse_by_layer]


def _tabpfn_noising_tracing(model: TabPFNRegressor, X_test: np.ndarray) -> list[float]:
    preds_clean = np.asarray(model.predict(X_test), dtype=np.float32)
    encoder: Any = model.model_.transformer_encoder
    layers = encoder.layers
    mse_by_layer: list[float] = []

    for layer_idx, layer in enumerate(layers):
        stds: list[float] = []

        def stats_hook(
            module: torch.nn.Module, inputs: Any, output: torch.Tensor
        ) -> None:  # noqa: ARG001
            stds.append(float(output.std().item()))

        stats_handle = layer.register_forward_hook(stats_hook)
        try:
            _ = model.predict(X_test)
        finally:
            stats_handle.remove()

        act_std = float(np.mean(stds)) if stds else 1.0
        rng = np.random.default_rng(cfg.SEED + 3000 + layer_idx)

        def noise_hook(
            module: torch.nn.Module, inputs: Any, output: torch.Tensor
        ) -> torch.Tensor:  # noqa: ARG001
            noise = torch.from_numpy(
                rng.normal(0.0, 0.5 * act_std, size=tuple(output.shape)).astype(
                    np.float32
                )
            ).to(output.device)
            return output + noise

        noise_handle = layer.register_forward_hook(noise_hook)
        try:
            preds_noised = np.asarray(model.predict(X_test), dtype=np.float32)
        finally:
            noise_handle.remove()
        mse_by_layer.append(_mse(preds_noised, preds_clean))

    max_mse = max(mse_by_layer) if max(mse_by_layer) > 0.0 else 1.0
    return [float(v / max_mse) for v in mse_by_layer]


def _run_c2_patching_comparison() -> dict[str, Any]:
    ds = _generate_patching_data()
    X_train = ds["X_train"]
    y_train = ds["y_train"]
    X_test = ds["X_test"]
    X_test_noised = _make_noised_input(X_test)

    tabpfn = TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")
    tabpfn.fit(X_train, y_train)
    tabpfn_standard = _tabpfn_standard_patching(tabpfn, X_test, X_test_noised)
    tabpfn_noising = _tabpfn_noising_tracing(tabpfn, X_test)

    tabicl = TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)
    tabicl.fit(X_train, y_train)
    tabicl_standard = _tabicl_standard_patching(tabicl, X_test, X_test_noised)
    tabicl_noising = _tabicl_noising_tracing(tabicl, X_test)

    return {
        "tabpfn": {
            "standard_recovery": tabpfn_standard,
            "noising_sensitivity": tabpfn_noising,
            "standard_peak_layer": int(np.argmax(np.asarray(tabpfn_standard))),
            "noising_peak_layer": int(np.argmax(np.asarray(tabpfn_noising))),
        },
        "tabicl": {
            "standard_recovery": tabicl_standard,
            "noising_sensitivity": tabicl_noising,
            "standard_peak_layer": int(np.argmax(np.asarray(tabicl_standard))),
            "noising_peak_layer": int(np.argmax(np.asarray(tabicl_noising))),
        },
    }


def _plot_c2(c2_results: dict[str, Any], save_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

    tabpfn_std = c2_results["tabpfn"]["standard_recovery"]
    tabpfn_noise = c2_results["tabpfn"]["noising_sensitivity"]
    tabicl_std = c2_results["tabicl"]["standard_recovery"]
    tabicl_noise = c2_results["tabicl"]["noising_sensitivity"]

    axes[0, 0].plot(np.arange(len(tabpfn_std)), tabpfn_std, marker="o", color="#1f77b4")
    axes[0, 0].set_title("TabPFN - Standard Patching")
    axes[0, 0].set_ylabel("Recovery")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(
        np.arange(len(tabpfn_noise)), tabpfn_noise, marker="o", color="#d62728"
    )
    axes[0, 1].set_title("TabPFN - Noising-based")
    axes[0, 1].set_ylabel("Normalized Sensitivity")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(np.arange(len(tabicl_std)), tabicl_std, marker="o", color="#1f77b4")
    axes[1, 0].set_title("TabICL - Standard Patching")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Recovery")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(
        np.arange(len(tabicl_noise)), tabicl_noise, marker="o", color="#d62728"
    )
    axes[1, 1].set_title("TabICL - Noising-based")
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Normalized Sensitivity")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("NeurIPS C2: Standard vs Noising-based Layer Sensitivity")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.SEED)

    c1_probings = _run_c1_probings()
    c1_steering = _run_c1_random_vector_steering()
    c2 = _run_c2_patching_comparison()

    payload: dict[str, Any] = {
        "meta": {
            "quick_run": bool(cfg.QUICK_RUN),
            "seed": int(cfg.SEED),
            "n_train": int(cfg.N_TRAIN),
            "n_test": int(cfg.N_TEST),
            "device": cfg.DEVICE,
        },
        "c1": {
            "probing_baselines": c1_probings,
            "random_vector_steering": c1_steering,
        },
        "c2": c2,
    }

    with RESULTS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _plot_c2(c2, FIGURE_PATH)
    print(f"Saved JSON: {RESULTS_JSON_PATH}")
    print(f"Saved figure: {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
