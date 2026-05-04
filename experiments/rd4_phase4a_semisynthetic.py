# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportImplicitStringConcatenation=false
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rd5_config import cfg

from src.data.synthetic_generator import generate_semi_synthetic_data  # noqa: E402
from src.hooks.tabpfn_hooker import TabPFNHookedModel  # noqa: E402
from src.probing.linear_probe import probe_all_layers  # noqa: E402
from src.visualization.plots import plot_layer_r2  # noqa: E402


QUICK_RUN = True
RANDOM_SEED = 42
N_LAYERS = 12


FUNCTIONS: dict[str, str] = {
    "sin_ab_plus_c2": "z = sin(a*b) + c^2",
    "a2_plus_bc": "z = a^2 + b*c",
    "exp_neg_a_plus_bc": "z = exp(-a) + b*c",
}


@dataclass(frozen=True)
class ExperimentConfig:
    func_name: str
    noise_sigma: float
    missing_rate: float
    n_features: int


def _to_numpy(array_like: Any) -> np.ndarray:
    if isinstance(array_like, np.ndarray):
        return array_like
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _build_model() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _inject_missing_values(
    X: np.ndarray,
    missing_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if missing_rate <= 0.0:
        return X

    X_nan = X.copy()
    n_rows, n_cols = X_nan.shape
    nan_count = int(n_rows * n_cols * missing_rate)
    if nan_count == 0:
        return X_nan

    flat_indices = rng.choice(n_rows * n_cols, size=nan_count, replace=False)
    rows = flat_indices // n_cols
    cols = flat_indices % n_cols
    X_nan[rows, cols] = np.nan
    return X_nan


def _generate_custom_data(
    func_name: str,
    noise_sigma: float,
    missing_rate: float,
    n_features: int,
    n_train: int,
    n_test: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    X_train = rng.uniform(0.1, 3.0, size=(n_train, n_features))
    X_test = rng.uniform(0.1, 3.0, size=(n_test, n_features))

    def _safe_col(X: np.ndarray, idx: int) -> np.ndarray:
        if idx < X.shape[1]:
            return X[:, idx]
        return np.zeros(X.shape[0], dtype=X.dtype)

    a_train, b_train, c_train = (
        _safe_col(X_train, 0),
        _safe_col(X_train, 1),
        _safe_col(X_train, 2),
    )
    a_test, b_test, c_test = (
        _safe_col(X_test, 0),
        _safe_col(X_test, 1),
        _safe_col(X_test, 2),
    )

    if func_name == "sin_ab_plus_c2":
        y_train = np.sin(a_train * b_train) + c_train**2
        y_test = np.sin(a_test * b_test) + c_test**2
    elif func_name == "a2_plus_bc":
        y_train = a_train**2 + b_train * c_train
        y_test = a_test**2 + b_test * c_test
    elif func_name == "exp_neg_a_plus_bc":
        y_train = np.exp(-a_train) + b_train * c_train
        y_test = np.exp(-a_test) + b_test * c_test
    else:
        raise ValueError(f"Unknown function: {func_name}")

    if noise_sigma > 0.0:
        y_train = y_train + rng.normal(0.0, noise_sigma, size=n_train)
        y_test = y_test + rng.normal(0.0, noise_sigma, size=n_test)

    X_train = _inject_missing_values(X_train, missing_rate, rng)
    X_test = _inject_missing_values(X_test, missing_rate, rng)
    return X_train, y_train, X_test, y_test


def _generate_dataset(
    cfg: ExperimentConfig,
    n_train: int,
    n_test: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cfg.n_features >= 3 and cfg.func_name in {"sin_ab_plus_c2", "a2_plus_bc"}:
        func_type = "nonlinear" if cfg.func_name == "sin_ab_plus_c2" else "polynomial"
        X_train, y_train, X_test, y_test = generate_semi_synthetic_data(
            func_type=func_type,
            noise_sigma=0.0,
            missing_rate=cfg.missing_rate,
            n_features=cfg.n_features,
            n_train=n_train,
            n_test=n_test,
            random_seed=random_seed,
        )
        if cfg.noise_sigma > 0.0:
            rng = np.random.default_rng(random_seed + 10_000)
            y_train = y_train + rng.normal(0.0, cfg.noise_sigma, size=n_train)
            y_test = y_test + rng.normal(0.0, cfg.noise_sigma, size=n_test)
        return X_train, y_train, X_test, y_test

    return _generate_custom_data(
        func_name=cfg.func_name,
        noise_sigma=cfg.noise_sigma,
        missing_rate=cfg.missing_rate,
        n_features=cfg.n_features,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
    )


def _extract_test_label_token_per_layer(cache: dict[str, Any]) -> list[np.ndarray]:
    single_eval_pos = int(cache["single_eval_pos"])
    per_layer: list[np.ndarray] = []
    for layer_idx in range(N_LAYERS):
        layer_tensor = cache["layers"][layer_idx]
        test_tok = _to_numpy(layer_tensor[0, single_eval_pos:, -1, :])
        per_layer.append(test_tok)
    return per_layer


def run_configuration(
    cfg: ExperimentConfig,
    n_datasets: int,
    n_train: int,
    n_test: int,
) -> np.ndarray:
    print(
        f"\n[RUN] func={cfg.func_name}, sigma={cfg.noise_sigma}, missing={cfg.missing_rate}, "
        f"d={cfg.n_features}, n_datasets={n_datasets}"
    )
    pooled_per_layer: list[list[np.ndarray]] = [[] for _ in range(N_LAYERS)]
    pooled_targets: list[np.ndarray] = []

    config_start = time.time()
    for ds_idx in range(n_datasets):
        ds_start = time.time()
        ds_seed = RANDOM_SEED + ds_idx
        X_train, y_train, X_test, y_test = _generate_dataset(
            cfg=cfg,
            n_train=n_train,
            n_test=n_test,
            random_seed=ds_seed,
        )

        model = _build_model()
        model.fit(X_train, y_train)

        hooker = TabPFNHookedModel(model)
        _, cache = hooker.forward_with_cache(X_test)
        per_layer = _extract_test_label_token_per_layer(cache)

        for layer_idx in range(N_LAYERS):
            pooled_per_layer[layer_idx].append(per_layer[layer_idx])
        pooled_targets.append(np.asarray(y_test))

        ds_elapsed = time.time() - ds_start
        total_elapsed = time.time() - config_start
        print(
            f"  dataset {ds_idx + 1:>2}/{n_datasets} done "
            f"(step={ds_elapsed:.1f}s, total={total_elapsed / 60:.1f}m)"
        )

    activations_per_layer = [
        np.vstack(layer_chunks).astype(np.float32) for layer_chunks in pooled_per_layer
    ]
    targets = np.concatenate(pooled_targets).astype(np.float32)
    probe = probe_all_layers(
        activations_per_layer,
        targets,
        complexities=[0],
        random_seed=RANDOM_SEED,
    )
    return _to_numpy(probe["r2"])[:, 0]


def _save_comparison_plot(
    curves: dict[str, np.ndarray],
    title: str,
    save_path: Path,
) -> None:
    if not curves:
        return
    matrix = np.column_stack([curves[label] for label in curves])
    fig = plot_layer_r2(
        matrix,
        title=title,
        save_path=str(save_path),
        complexity_labels=list(curves.keys()),
    )
    plt.close(fig)


def main() -> None:
    results_dir = ROOT / "results" / "rd4_phase4a"
    results_dir.mkdir(parents=True, exist_ok=True)

    if QUICK_RUN:
        func_names = ["sin_ab_plus_c2"]
        noise_levels = [0.0, 0.5, 2.0]
        missing_rates = [0.0]
        feature_dims = [2]
        n_datasets = 5
    else:
        func_names = ["sin_ab_plus_c2", "a2_plus_bc", "exp_neg_a_plus_bc"]
        noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0]
        missing_rates = [0.0, 0.1, 0.2, 0.3]
        feature_dims = [2, 5, 10]
        n_datasets = 20

    n_train = 100
    n_test = 20

    print("=" * 80)
    print("RD-4 Phase 4A: Semi-synthetic data probing")
    print("=" * 80)
    print(f"QUICK_RUN={QUICK_RUN}")

    results_payload: dict[str, Any] = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "n_layers": N_LAYERS,
        "n_datasets": n_datasets,
        "n_train": n_train,
        "n_test": n_test,
        "functions": FUNCTIONS,
        "results": {},
    }

    all_curves: dict[tuple[str, float, float, int], np.ndarray] = {}

    for func_name in func_names:
        for noise_sigma in noise_levels:
            for missing_rate in missing_rates:
                for n_features in feature_dims:
                    cfg = ExperimentConfig(
                        func_name=func_name,
                        noise_sigma=noise_sigma,
                        missing_rate=missing_rate,
                        n_features=n_features,
                    )
                    r2_curve = run_configuration(
                        cfg=cfg,
                        n_datasets=n_datasets,
                        n_train=n_train,
                        n_test=n_test,
                    )

                    key = (func_name, noise_sigma, missing_rate, n_features)
                    all_curves[key] = r2_curve
                    results_payload["results"][
                        f"func={func_name}|sigma={noise_sigma}|missing={missing_rate}|d={n_features}"
                    ] = {
                        "func_name": func_name,
                        "formula": FUNCTIONS[func_name],
                        "noise_sigma": noise_sigma,
                        "missing_rate": missing_rate,
                        "n_features": n_features,
                        "r2_curve": r2_curve.tolist(),
                        "max_r2": float(np.max(r2_curve)),
                        "best_layer": int(np.argmax(r2_curve)),
                    }

    with (results_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    noise_curves: dict[str, np.ndarray] = {}
    fixed_func = func_names[0]
    for sigma in noise_levels:
        key = (fixed_func, sigma, 0.0, 2)
        if key in all_curves:
            noise_curves[f"sigma={sigma}"] = all_curves[key]
    _save_comparison_plot(
        curves=noise_curves,
        title=f"Noise effect on linear probe R² ({fixed_func}, missing=0, d=2)",
        save_path=results_dir / "noise_effect_r2.png",
    )

    missing_curves: dict[str, np.ndarray] = {}
    for miss in missing_rates:
        key = (fixed_func, 0.5, miss, 2)
        if key in all_curves:
            missing_curves[f"missing={int(miss * 100)}%"] = all_curves[key]
    if missing_curves:
        _save_comparison_plot(
            curves=missing_curves,
            title=f"Missing-value effect on linear probe R² ({fixed_func}, sigma=0.5, d=2)",
            save_path=results_dir / "missing_effect_r2.png",
        )

    function_curves: dict[str, np.ndarray] = {}
    for func_name in func_names:
        key = (func_name, 0.5, 0.0, 2)
        if key in all_curves:
            function_curves[func_name] = all_curves[key]
    if function_curves:
        _save_comparison_plot(
            curves=function_curves,
            title="Function-type effect on linear probe R² (sigma=0.5, missing=0, d=2)",
            save_path=results_dir / "function_effect_r2.png",
        )

    criterion_key = ("sin_ab_plus_c2", 0.5, 0.0, 2)
    robust_pass = False
    criterion_curve = all_curves.get(criterion_key)
    if criterion_curve is not None:
        robust_pass = bool(np.all(criterion_curve[5:9] > 0.5))

    summary_lines = [
        "RD-4 Phase 4A Semi-synthetic Experiment Summary",
        f"quick_run={QUICK_RUN}",
        f"num_configurations={len(all_curves)}",
        f"robustness_criterion_sigma0.5_layer5to8_gt0.5={robust_pass}",
    ]

    if criterion_curve is not None:
        summary_lines.append(
            "sigma0.5_reference_curve="
            + ",".join(f"{v:.4f}" for v in criterion_curve.tolist())
        )

    for key, curve in all_curves.items():
        func_name, sigma, miss, n_features = key
        summary_lines.append(
            f"func={func_name} sigma={sigma} missing={miss} d={n_features} "
            f"best_layer={int(np.argmax(curve))} max_r2={float(np.max(curve)):.4f}"
        )

    summary_text = "\n".join(summary_lines) + "\n"
    with (results_dir / "summary.txt").open("w", encoding="utf-8") as f:
        _ = f.write(summary_text)

    print("\nDone.")
    print(f"Saved: {results_dir / 'results.json'}")
    print(f"Saved: {results_dir / 'noise_effect_r2.png'}")
    print(f"Saved: {results_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
