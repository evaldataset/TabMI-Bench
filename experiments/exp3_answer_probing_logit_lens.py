# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportExplicitAny=false, reportImplicitStringConcatenation=false
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score
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
R2_THRESHOLD = 0.5


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


def _decode_logits_to_prediction(
    model: TabPFNRegressor, logits: np.ndarray
) -> np.ndarray:
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


def _first_layer_above_threshold(r2_curve: np.ndarray, threshold: float) -> int | None:
    above = np.where(r2_curve >= threshold)[0]
    if len(above) == 0:
        return None
    return int(above[0])


def run_experiment(
    n_datasets: int, n_train: int, n_test: int
) -> tuple[np.ndarray, np.ndarray]:
    pooled_per_layer: list[list[np.ndarray]] = [[] for _ in range(N_LAYERS)]
    pooled_y_test: list[np.ndarray] = []
    pooled_logit_lens_preds: list[list[np.ndarray]] = [[] for _ in range(N_LAYERS)]

    start_time = time.time()
    for ds_idx in range(n_datasets):
        fit_start = time.time()
        ds = generate_quadratic_data(
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

            logits = hooker.apply_logit_lens(cache, layer_idx=layer_idx)
            preds = _decode_logits_to_prediction(model, logits)
            pooled_logit_lens_preds[layer_idx].append(preds.astype(np.float32))

        pooled_y_test.append(np.asarray(ds.y_test, dtype=np.float32))

        elapsed_fit = time.time() - fit_start
        elapsed_total = time.time() - start_time
        print(
            f"  Dataset {ds_idx + 1:>3}/{n_datasets} done "
            f"(fit_time={elapsed_fit:.1f}s, total={elapsed_total / 60:.1f}m)"
        )

    activations_per_layer = [
        np.vstack(layer_chunks).astype(np.float32) for layer_chunks in pooled_per_layer
    ]
    y_test_pooled = np.concatenate(pooled_y_test).astype(np.float32)

    linear_probe = probe_all_layers(
        activations_per_layer,
        y_test_pooled,
        complexities=COMPLEXITIES,
        random_seed=RANDOM_SEED,
    )
    linear_probe_r2 = _to_numpy(linear_probe["r2"])[:, 0]

    logit_lens_r2 = np.zeros(N_LAYERS, dtype=np.float32)
    for layer_idx in range(N_LAYERS):
        layer_preds = np.concatenate(pooled_logit_lens_preds[layer_idx])
        logit_lens_r2[layer_idx] = float(r2_score(y_test_pooled, layer_preds))

    return linear_probe_r2, logit_lens_r2


def main() -> None:
    results_dir = ROOT / "results" / "exp3"
    results_dir.mkdir(parents=True, exist_ok=True)

    n_datasets = 10 if QUICK_RUN else 100
    n_train = 50
    n_test = 10

    print("=" * 72)
    print("Experiment 3: Answer Probing + Logit Lens")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}")
    print(f"n_datasets={n_datasets}, n_train={n_train}, n_test={n_test}")

    linear_probe_r2, logit_lens_r2 = run_experiment(
        n_datasets=n_datasets,
        n_train=n_train,
        n_test=n_test,
    )

    r2_curves = np.column_stack([linear_probe_r2, logit_lens_r2])
    fig = plot_layer_r2(
        r2_curves,
        title="Final Answer (z): Linear Probe vs Logit Lens",
        save_path=str(results_dir / "answer_probing_vs_logit_lens.png"),
        complexity_labels=["Linear Probe (complexity=0)", "Logit Lens"],
    )
    plt.close(fig)

    results_payload = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "n_datasets": n_datasets,
        "n_train": n_train,
        "n_test": n_test,
        "linear_probe_r2": linear_probe_r2.tolist(),
        "logit_lens_r2": logit_lens_r2.tolist(),
    }
    with (results_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    linear_best_layer = int(np.argmax(linear_probe_r2))
    linear_best_r2 = float(linear_probe_r2[linear_best_layer])
    logit_best_layer = int(np.argmax(logit_lens_r2))
    logit_best_r2 = float(logit_lens_r2[logit_best_layer])

    linear_first = _first_layer_above_threshold(linear_probe_r2, R2_THRESHOLD)
    logit_first = _first_layer_above_threshold(logit_lens_r2, R2_THRESHOLD)
    gap = (
        None
        if (linear_first is None or logit_first is None)
        else logit_first - linear_first
    )

    print("\nSummary:")
    print(f"- Linear probe: max R²={linear_best_r2:.4f} at layer {linear_best_layer}")
    print(f"- Logit lens:  max R²={logit_best_r2:.4f} at layer {logit_best_layer}")
    print(
        f"- First layer with R² >= {R2_THRESHOLD:.1f}: "
        f"linear={linear_first}, logit_lens={logit_first}, gap={gap}"
    )
    print(f"Saved outputs to: {results_dir}")


if __name__ == "__main__":
    main()
