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

from src.data.synthetic_generator import generate_quadratic_data  # noqa: E402
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


def _build_model() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _extract_test_label_token_per_layer(cache: dict[str, Any]) -> list[np.ndarray]:
    single_eval_pos = int(cache["single_eval_pos"])
    per_layer: list[np.ndarray] = []
    for layer_idx in range(N_LAYERS):
        layer_tensor = cache["layers"][layer_idx]
        test_tok = _to_numpy(layer_tensor[0, single_eval_pos:, -1, :])
        per_layer.append(test_tok)
    return per_layer


def _generate_dataset_with_fixed_coefficients(
    a: float, b: float, c: float, n_train: int, n_test: int, random_seed: int
) -> Any:
    return generate_quadratic_data(
        a_range=(0.5, a),
        b_range=(0.5, b),
        c_range=(0.5, c),
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed,
    )


def run_experiment(
    n_datasets: int, n_train: int, n_test: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(RANDOM_SEED)

    pooled_per_layer: list[list[np.ndarray]] = [[] for _ in range(N_LAYERS)]
    pooled_intermediary: list[np.ndarray] = []
    pooled_final: list[np.ndarray] = []

    start_time = time.time()
    for ds_idx in range(n_datasets):
        fit_start = time.time()

        a = float(rng.uniform(0.5, 3.0))
        b = float(rng.uniform(0.5, 3.0))
        c = float(rng.uniform(0.5, 3.0))

        ds = _generate_dataset_with_fixed_coefficients(
            a=a,
            b=b,
            c=c,
            n_train=n_train,
            n_test=n_test,
            random_seed=RANDOM_SEED + ds_idx,
        )

        model = _build_model()
        model.fit(ds.X_train, ds.y_train)

        hooker = TabPFNHookedModel(model)
        _, cache = hooker.forward_with_cache(ds.X_test)
        per_layer = _extract_test_label_token_per_layer(cache)

        for layer_idx in range(N_LAYERS):
            pooled_per_layer[layer_idx].append(per_layer[layer_idx])

        pooled_intermediary.append(np.asarray(ds.intermediary_test, dtype=np.float32))
        pooled_final.append(np.asarray(ds.y_test, dtype=np.float32))

        elapsed_fit = time.time() - fit_start
        elapsed_total = time.time() - start_time
        print(
            f"  Dataset {ds_idx + 1:>3}/{n_datasets} done "
            f"(a={a:.3f}, b={b:.3f}, c={c:.3f}, "
            f"fit_time={elapsed_fit:.1f}s, total={elapsed_total / 60:.1f}m)"
        )

    activations_per_layer = [
        np.vstack(layer_chunks) for layer_chunks in pooled_per_layer
    ]
    intermediary_targets = np.concatenate(pooled_intermediary)
    final_targets = np.concatenate(pooled_final)

    intermediary_probe = probe_all_layers(
        activations_per_layer,
        intermediary_targets,
        complexities=COMPLEXITIES,
        random_seed=RANDOM_SEED,
    )
    final_probe = probe_all_layers(
        activations_per_layer,
        final_targets,
        complexities=COMPLEXITIES,
        random_seed=RANDOM_SEED,
    )

    return intermediary_probe, final_probe


def _print_summary(name: str, r2_curve: np.ndarray) -> None:
    best_layer = int(np.argmax(r2_curve))
    best_r2 = float(r2_curve[best_layer])
    print(f"- {name}: max R²={best_r2:.4f} at layer {best_layer}")


def main() -> None:
    results_dir = ROOT / "results" / "exp2"
    results_dir.mkdir(parents=True, exist_ok=True)

    if QUICK_RUN:
        n_datasets = 10
    else:
        n_datasets = 100

    n_train = 50
    n_test = 10

    print("=" * 72)
    print("Experiment 2: Intermediary Probing")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}")
    print(f"n_datasets={n_datasets}, n_train={n_train}, n_test={n_test}")

    intermediary_probe, final_probe = run_experiment(
        n_datasets=n_datasets,
        n_train=n_train,
        n_test=n_test,
    )

    intermediary_linear_r2 = _to_numpy(intermediary_probe["r2"])[:, 0]
    final_linear_r2 = _to_numpy(final_probe["r2"])[:, 0]

    r2_curves = np.column_stack([intermediary_linear_r2, final_linear_r2])
    fig = plot_layer_r2(
        r2_curves,
        title="Intermediary (a·b) vs Final (z) Probing — Linear (complexity=0)",
        save_path=str(results_dir / "intermediary_r2_curves.png"),
        complexity_labels=["a·b (intermediary)", "z (final answer)"],
    )
    plt.close(fig)

    results_payload = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "n_datasets": n_datasets,
        "n_train": n_train,
        "n_test": n_test,
        "complexities": COMPLEXITIES,
        "intermediary_r2": intermediary_linear_r2.tolist(),
        "final_r2": final_linear_r2.tolist(),
        "intermediary_r2_all_complexities": _to_numpy(
            intermediary_probe["r2"]
        ).tolist(),
        "final_r2_all_complexities": _to_numpy(final_probe["r2"]).tolist(),
    }
    with (results_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("\nSummary (complexity=0):")
    _print_summary("a·b (intermediary)", intermediary_linear_r2)
    _print_summary("z (final answer)", final_linear_r2)

    expected_signal = bool(np.any(intermediary_linear_r2[5:9] > 0.5))
    print(f"Expected signal check (Layer 5-8, a·b R² > 0.5): {expected_signal}")
    print(f"Saved outputs to: {results_dir}")


if __name__ == "__main__":
    main()
