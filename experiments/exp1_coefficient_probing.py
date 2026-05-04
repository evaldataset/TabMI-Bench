# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rd5_config import cfg

from src.data.synthetic_generator import (  # noqa: E402
    generate_multiple_linear_fits,
    generate_switch_variable_data,
)
from src.hooks.tabpfn_hooker import TabPFNHookedModel  # noqa: E402
from src.probing.linear_probe import probe_all_layers  # noqa: E402
from src.visualization.plots import plot_layer_r2  # noqa: E402


QUICK_RUN = True
RANDOM_SEED = 42
COMPLEXITIES = [0, 1, 2, 3]
N_LAYERS = 12


def _to_numpy(array_like: Any) -> np.ndarray:
    if isinstance(array_like, np.ndarray):
        return array_like
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _extract_test_label_token_per_layer(cache: dict[str, Any]) -> list[np.ndarray]:
    single_eval_pos = int(cache["single_eval_pos"])
    per_layer: list[np.ndarray] = []
    for layer_idx in range(N_LAYERS):
        layer_tensor = cache["layers"][layer_idx]
        test_tok = _to_numpy(layer_tensor[0, single_eval_pos:, -1, :])
        per_layer.append(test_tok)
    return per_layer


def _build_model() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def run_configuration_1(
    n_fits: int, n_train: int, n_test: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    print("\n[Config 1] Multiple Fits, Pooled")
    print(f"- n_fits={n_fits}, n_train={n_train}, n_test={n_test}")

    datasets = generate_multiple_linear_fits(
        n_fits=n_fits,
        n_train=n_train,
        n_test=n_test,
        random_seed=RANDOM_SEED,
    )

    pooled_per_layer: list[list[np.ndarray]] = [[] for _ in range(N_LAYERS)]
    pooled_alphas: list[np.ndarray] = []
    pooled_betas: list[np.ndarray] = []

    start_time = time.time()
    for fit_idx, ds in enumerate(datasets, start=1):
        fit_start = time.time()
        model = _build_model()
        model.fit(ds.X_train, ds.y_train)

        hooker = TabPFNHookedModel(model)
        _, cache = hooker.forward_with_cache(ds.X_test)
        per_layer = _extract_test_label_token_per_layer(cache)

        for layer_idx in range(N_LAYERS):
            pooled_per_layer[layer_idx].append(per_layer[layer_idx])

        pooled_alphas.append(np.full(ds.X_test.shape[0], ds.alpha, dtype=np.float32))
        pooled_betas.append(np.full(ds.X_test.shape[0], ds.beta, dtype=np.float32))

        elapsed_fit = time.time() - fit_start
        elapsed_total = time.time() - start_time
        print(
            f"  Fit {fit_idx:>2}/{n_fits} done "
            f"(alpha={ds.alpha:.3f}, beta={ds.beta:.3f}, "
            f"fit_time={elapsed_fit:.1f}s, total={elapsed_total / 60:.1f}m)"
        )

    activations_per_layer = [
        np.vstack(layer_chunks) for layer_chunks in pooled_per_layer
    ]
    targets_alpha = np.concatenate(pooled_alphas)
    targets_beta = np.concatenate(pooled_betas)

    alpha_probe = probe_all_layers(
        activations_per_layer,
        targets_alpha,
        complexities=COMPLEXITIES,
        random_seed=RANDOM_SEED,
    )
    beta_probe = probe_all_layers(
        activations_per_layer,
        targets_beta,
        complexities=COMPLEXITIES,
        random_seed=RANDOM_SEED,
    )

    return alpha_probe, beta_probe


def run_configuration_2(
    n_coefficient_pairs: int, n_samples_per_pair: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    print("\n[Config 2] Single Fit, Switch Variable")
    print(
        f"- n_coefficient_pairs={n_coefficient_pairs}, "
        f"n_samples_per_pair={n_samples_per_pair}"
    )

    X, y, true_alphas, true_betas = generate_switch_variable_data(
        n_coefficient_pairs=n_coefficient_pairs,
        n_samples_per_pair=n_samples_per_pair,
        random_seed=RANDOM_SEED,
    )

    n_total = X.shape[0]
    if n_total >= 1000:
        split_idx = 800
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train = y[:split_idx]
        test_alphas = true_alphas[split_idx:]
        test_betas = true_betas[split_idx:]
    else:
        train_idx_parts: list[np.ndarray] = []
        test_idx_parts: list[np.ndarray] = []
        for pair_idx in range(n_coefficient_pairs):
            start = pair_idx * n_samples_per_pair
            end = start + n_samples_per_pair
            pair_indices = np.arange(start, end)
            pair_split = int(0.8 * n_samples_per_pair)
            train_idx_parts.append(pair_indices[:pair_split])
            test_idx_parts.append(pair_indices[pair_split:])

        train_indices = np.concatenate(train_idx_parts)
        test_indices = np.concatenate(test_idx_parts)
        X_train, X_test = X[train_indices], X[test_indices]
        y_train = y[train_indices]
        test_alphas = true_alphas[test_indices]
        test_betas = true_betas[test_indices]

    print(
        f"- total_samples={n_total}, train={X_train.shape[0]}, test={X_test.shape[0]}"
    )

    model = _build_model()
    model.fit(X_train, y_train)

    hooker = TabPFNHookedModel(model)
    _, cache = hooker.forward_with_cache(X_test)
    activations_per_layer = _extract_test_label_token_per_layer(cache)

    alpha_probe = probe_all_layers(
        activations_per_layer,
        test_alphas,
        complexities=COMPLEXITIES,
        random_seed=RANDOM_SEED,
    )
    beta_probe = probe_all_layers(
        activations_per_layer,
        test_betas,
        complexities=COMPLEXITIES,
        random_seed=RANDOM_SEED,
    )

    return alpha_probe, beta_probe


def _save_config_plot(
    alpha_probe: dict[str, Any],
    beta_probe: dict[str, Any],
    title: str,
    save_path: Path,
) -> None:
    alpha_linear = _to_numpy(alpha_probe["r2"])[:, 0]
    beta_linear = _to_numpy(beta_probe["r2"])[:, 0]
    r2_curves = np.column_stack([alpha_linear, beta_linear])

    fig = plot_layer_r2(
        r2_curves,
        title=title,
        save_path=str(save_path),
        complexity_labels=["alpha (complexity=0)", "beta (complexity=0)"],
    )
    plt.close(fig)


def _print_summary(
    name: str, alpha_linear_r2: np.ndarray, beta_linear_r2: np.ndarray
) -> None:
    alpha_best_layer = int(np.argmax(alpha_linear_r2))
    beta_best_layer = int(np.argmax(beta_linear_r2))
    alpha_best = float(alpha_linear_r2[alpha_best_layer])
    beta_best = float(beta_linear_r2[beta_best_layer])

    if alpha_best >= beta_best:
        overall_target = "alpha"
        overall_layer = alpha_best_layer
        overall_r2 = alpha_best
    else:
        overall_target = "beta"
        overall_layer = beta_best_layer
        overall_r2 = beta_best

    print(f"\n{name} summary (complexity=0):")
    print(f"- alpha: max R²={alpha_best:.4f} at layer {alpha_best_layer}")
    print(f"- beta:  max R²={beta_best:.4f} at layer {beta_best_layer}")
    print(
        f"- overall: {overall_target} max R²={overall_r2:.4f} at layer {overall_layer}"
    )


def main() -> None:
    results_dir = ROOT / "results" / "exp1"
    results_dir.mkdir(parents=True, exist_ok=True)

    if QUICK_RUN:
        n_fits = 10
        n_coefficient_pairs = 5
    else:
        n_fits = 100
        n_coefficient_pairs = 20

    n_train = 50
    n_test = 10
    n_samples_per_pair = 50

    print("=" * 72)
    print("Experiment 1: Coefficient Probing")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}")

    cfg1_alpha_probe, cfg1_beta_probe = run_configuration_1(
        n_fits=n_fits,
        n_train=n_train,
        n_test=n_test,
    )
    cfg2_alpha_probe, cfg2_beta_probe = run_configuration_2(
        n_coefficient_pairs=n_coefficient_pairs,
        n_samples_per_pair=n_samples_per_pair,
    )

    cfg1_alpha_linear = _to_numpy(cfg1_alpha_probe["r2"])[:, 0]
    cfg1_beta_linear = _to_numpy(cfg1_beta_probe["r2"])[:, 0]
    cfg2_alpha_linear = _to_numpy(cfg2_alpha_probe["r2"])[:, 0]
    cfg2_beta_linear = _to_numpy(cfg2_beta_probe["r2"])[:, 0]

    _save_config_plot(
        cfg1_alpha_probe,
        cfg1_beta_probe,
        title="Config 1 (Multiple Fits, Pooled): Linear Probe R²",
        save_path=results_dir / "config1_r2_curves.png",
    )
    _save_config_plot(
        cfg2_alpha_probe,
        cfg2_beta_probe,
        title="Config 2 (Single Fit, Switch Variable): Linear Probe R²",
        save_path=results_dir / "config2_r2_curves.png",
    )

    results_payload = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "complexities": COMPLEXITIES,
        "config1_alpha_r2": cfg1_alpha_linear.tolist(),
        "config1_beta_r2": cfg1_beta_linear.tolist(),
        "config2_alpha_r2": cfg2_alpha_linear.tolist(),
        "config2_beta_r2": cfg2_beta_linear.tolist(),
    }

    with (results_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    _print_summary("Config 1", cfg1_alpha_linear, cfg1_beta_linear)
    _print_summary("Config 2", cfg2_alpha_linear, cfg2_beta_linear)

    has_expected_peak = bool(
        np.any(cfg1_alpha_linear[6:9] > 0.5)
        or np.any(cfg1_beta_linear[6:9] > 0.5)
        or np.any(cfg2_alpha_linear[6:9] > 0.5)
        or np.any(cfg2_beta_linear[6:9] > 0.5)
    )
    print(f"\nExpected signal check (Layer 6-8, R² > 0.5): {has_expected_peak}")
    print(f"Saved outputs to: {results_dir}")


if __name__ == "__main__":
    main()
