# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAny=false, reportExplicitAny=false, reportImplicitStringConcatenation=false
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
N_LAYERS = 12


def _to_numpy(array_like: Any) -> np.ndarray:
    if isinstance(array_like, np.ndarray):
        return array_like
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _build_model() -> TabPFNRegressor:
    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _extract_answer_token_per_layer(cache: dict[str, Any]) -> list[np.ndarray]:
    single_eval_pos = int(cache["single_eval_pos"])
    per_layer: list[np.ndarray] = []
    for layer_idx in range(N_LAYERS):
        layer_tensor = cache["layers"][layer_idx]
        answer_tok = _to_numpy(layer_tensor[0, single_eval_pos:, -1, :])
        per_layer.append(answer_tok)
    return per_layer


def run_copy_mechanism_experiment(
    n_datasets: int,
    n_train: int = 50,
    n_test: int = 10,
) -> dict[str, np.ndarray]:
    pooled_per_layer: list[list[np.ndarray]] = [[] for _ in range(N_LAYERS)]
    pooled_a: list[np.ndarray] = []
    pooled_b: list[np.ndarray] = []
    pooled_c: list[np.ndarray] = []
    pooled_ab: list[np.ndarray] = []

    start_time = time.time()
    for dataset_idx in range(n_datasets):
        ds = generate_quadratic_data(
            n_train=n_train,
            n_test=n_test,
            random_seed=RANDOM_SEED + dataset_idx,
        )

        model = _build_model()
        model.fit(ds.X_train, ds.y_train)

        hooker = TabPFNHookedModel(model)
        _, cache = hooker.forward_with_cache(ds.X_test)
        per_layer = _extract_answer_token_per_layer(cache)

        for layer_idx in range(N_LAYERS):
            pooled_per_layer[layer_idx].append(per_layer[layer_idx])

        pooled_a.append(ds.X_test[:, 0].astype(np.float32))
        pooled_b.append(ds.X_test[:, 1].astype(np.float32))
        pooled_c.append(ds.X_test[:, 2].astype(np.float32))
        pooled_ab.append(ds.intermediary_test.astype(np.float32))

        elapsed = time.time() - start_time
        print(
            f"  Dataset {dataset_idx + 1:>3}/{n_datasets} done "
            f"(total={elapsed / 60:.1f}m)"
        )

    activations_per_layer = [
        np.vstack(layer_chunks).astype(np.float32) for layer_chunks in pooled_per_layer
    ]
    targets_a = np.concatenate(pooled_a)
    targets_b = np.concatenate(pooled_b)
    targets_c = np.concatenate(pooled_c)
    targets_ab = np.concatenate(pooled_ab)

    probe_kwargs = {"complexities": [0], "random_seed": RANDOM_SEED}
    a_probe = probe_all_layers(activations_per_layer, targets_a, **probe_kwargs)
    b_probe = probe_all_layers(activations_per_layer, targets_b, **probe_kwargs)
    c_probe = probe_all_layers(activations_per_layer, targets_c, **probe_kwargs)
    ab_probe = probe_all_layers(activations_per_layer, targets_ab, **probe_kwargs)

    return {
        "a_r2": _to_numpy(a_probe["r2"])[:, 0],
        "b_r2": _to_numpy(b_probe["r2"])[:, 0],
        "c_r2": _to_numpy(c_probe["r2"])[:, 0],
        "ab_r2": _to_numpy(ab_probe["r2"])[:, 0],
    }


def _print_peak(name: str, r2_curve: np.ndarray) -> None:
    best_layer = int(np.argmax(r2_curve))
    best_r2 = float(r2_curve[best_layer])
    print(f"- {name}: max R²={best_r2:.4f} at layer {best_layer}")


def main() -> None:
    results_dir = ROOT / "results" / "exp4"
    results_dir.mkdir(parents=True, exist_ok=True)

    n_datasets = 10 if QUICK_RUN else 100

    print("=" * 72)
    print("Experiment 4: Copy Mechanism")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}")
    print(f"n_datasets={n_datasets}, n_train=50, n_test=10")

    r2_curves = run_copy_mechanism_experiment(
        n_datasets=n_datasets,
        n_train=50,
        n_test=10,
    )

    r2_matrix = np.column_stack(
        [
            r2_curves["a_r2"],
            r2_curves["b_r2"],
            r2_curves["c_r2"],
            r2_curves["ab_r2"],
        ]
    )

    fig = plot_layer_r2(
        r2_matrix,
        title="Copy Mechanism: Recoverability from Answer Token (Linear Probe)",
        save_path=str(results_dir / "copy_mechanism_r2_curves.png"),
        complexity_labels=["a", "b", "c", "a·b"],
    )
    plt.close(fig)

    payload = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "a_r2": r2_curves["a_r2"].tolist(),
        "b_r2": r2_curves["b_r2"].tolist(),
        "c_r2": r2_curves["c_r2"].tolist(),
        "ab_r2": r2_curves["ab_r2"].tolist(),
    }
    with (results_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nPeak linear-probe R² by target:")
    _print_peak("a", r2_curves["a_r2"])
    _print_peak("b", r2_curves["b_r2"])
    _print_peak("c", r2_curves["c_r2"])
    _print_peak("a·b", r2_curves["ab_r2"])

    layer5_plus_ok = bool(
        np.all(r2_curves["a_r2"][5:] > 0.5)
        and np.all(r2_curves["b_r2"][5:] > 0.5)
        and np.all(r2_curves["c_r2"][5:] > 0.5)
        and np.all(r2_curves["ab_r2"][5:] > 0.5)
    )
    print(f"\nExpected signal check (all targets Layer 5+, R² > 0.5): {layer5_plus_ok}")
    print(f"Saved outputs to: {results_dir}")


if __name__ == "__main__":
    main()
