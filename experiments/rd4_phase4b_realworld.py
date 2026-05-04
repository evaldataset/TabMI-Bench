# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportImplicitStringConcatenation=false
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

from src.data.real_world_datasets import get_available_datasets  # noqa: E402
from src.hooks.tabpfn_hooker import TabPFNHookedModel  # noqa: E402
from src.probing.linear_probe import probe_all_layers  # noqa: E402
from src.probing.real_world_targets import (  # noqa: E402
    ProbingTarget,
    compute_feature_targets,
    compute_prediction_targets,
)
from src.visualization.plots import plot_layer_r2  # noqa: E402


QUICK_RUN = True
RANDOM_SEED = 42
COMPLEXITIES = [0]
LAYER_REGION_START = 5
LAYER_REGION_END = 8


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
    n_layers = len(cache["layers"])
    per_layer: list[np.ndarray] = []
    for layer_idx in range(n_layers):
        layer_tensor = cache["layers"][layer_idx]
        test_tok = _to_numpy(layer_tensor[0, single_eval_pos:, -1, :]).astype(
            np.float32
        )
        per_layer.append(test_tok)
    return per_layer


def _probe_target(
    activations_per_layer: list[np.ndarray],
    target: ProbingTarget,
) -> np.ndarray:
    probe = probe_all_layers(
        activations_per_layer,
        np.asarray(target.values, dtype=np.float32),
        complexities=COMPLEXITIES,
        random_seed=RANDOM_SEED,
    )
    return _to_numpy(probe["r2"])[:, 0]


def _safe_target_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def main() -> None:
    results_dir = ROOT / "results" / "rd4_phase4b"
    results_dir.mkdir(parents=True, exist_ok=True)

    n_train = 200 if QUICK_RUN else 500
    n_test = 50 if QUICK_RUN else 100

    datasets, skipped = get_available_datasets(
        n_train=n_train,
        n_test=n_test,
        random_seed=RANDOM_SEED,
    )

    print("=" * 80)
    print("RD-4 Phase 4B: Real-world dataset probing")
    print("=" * 80)
    print(f"QUICK_RUN={QUICK_RUN} n_train={n_train} n_test={n_test}")
    print(f"datasets={[d.name for d in datasets]}")
    if skipped:
        print(f"skipped={[s['name'] for s in skipped]}")

    results_payload: dict[str, Any] = {
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
        "n_train": n_train,
        "n_test": n_test,
        "complexities": COMPLEXITIES,
        "layer_peak_region": [LAYER_REGION_START, LAYER_REGION_END],
        "datasets": {},
    }

    datasets_peak_in_region = 0
    dataset_summaries: list[str] = []

    for ds in datasets:
        print(f"\n[DATASET] {ds.name}")
        print(
            f"- train={ds.X_train.shape}, test={ds.X_test.shape}, features={ds.n_features}"
        )

        start_time = time.time()
        model = _build_model()
        model.fit(ds.X_train, ds.y_train)

        hooker = TabPFNHookedModel(model)
        predictions, cache = hooker.forward_with_cache(ds.X_test)
        activations_per_layer = _extract_test_label_token_per_layer(cache)

        prediction_targets = compute_prediction_targets(
            y_pred=np.asarray(predictions),
            y_true=np.asarray(ds.y_test),
        )
        y_true_target = next(t for t in prediction_targets if t.name == "y_true")
        feature_targets = compute_feature_targets(ds.X_test, ds.feature_names)
        all_targets = [y_true_target] + feature_targets

        target_curves: dict[str, np.ndarray] = {}
        for target in all_targets:
            target_curves[target.name] = _probe_target(activations_per_layer, target)

        y_true_curve = target_curves["y_true"]
        best_layer = int(np.argmax(y_true_curve))
        max_r2 = float(np.max(y_true_curve))
        in_peak_region = LAYER_REGION_START <= best_layer <= LAYER_REGION_END
        if in_peak_region:
            datasets_peak_in_region += 1

        elapsed = time.time() - start_time
        print(
            f"- y_true best_layer={best_layer}, max_r2={max_r2:.4f}, "
            f"peak_layer5to8={in_peak_region}, elapsed={elapsed / 60:.1f}m"
        )

        labels = ["y_true"] + [target.name for target in feature_targets]
        matrix = np.column_stack([target_curves[label] for label in labels])
        fig = plot_layer_r2(
            matrix,
            title=f"{ds.name}: Linear Probe R2 by Layer",
            save_path=str(results_dir / f"{ds.name}_r2_curves.png"),
            complexity_labels=[_safe_target_name(label) for label in labels],
        )
        plt.close(fig)

        results_payload["datasets"][ds.name] = {
            "n_train": int(ds.n_train),
            "n_test": int(ds.n_test),
            "n_features": int(ds.n_features),
            "feature_names": [str(name) for name in ds.feature_names],
            "y_true_best_layer": best_layer,
            "y_true_max_r2": max_r2,
            "y_true_peak_layer5to8": in_peak_region,
            "r2_curves": {
                _safe_target_name(name): [float(v) for v in curve.tolist()]
                for name, curve in target_curves.items()
            },
        }

        dataset_summaries.append(
            f"{ds.name}: best_layer={best_layer}, max_r2={max_r2:.4f}, layer5to8={in_peak_region}"
        )

    success = datasets_peak_in_region >= 2
    results_payload["success_criterion"] = {
        "definition": "At least 2 datasets have y_true best layer in 5-8",
        "datasets_in_region": datasets_peak_in_region,
        "total_datasets": len(datasets),
        "passed": success,
    }

    with (results_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    summary_lines = [
        "RD-4 Phase 4B Real-world probing summary",
        f"quick_run={QUICK_RUN}",
        f"datasets_in_peak_region={datasets_peak_in_region}/{len(datasets)}",
        f"success_criterion_passed={success}",
        "",
        *dataset_summaries,
    ]
    with (results_dir / "summary.txt").open("w", encoding="utf-8") as f:
        _ = f.write("\n".join(summary_lines) + "\n")

    print("\nDone.")
    print(f"Success criterion (>=2 datasets peak at layer 5-8): {success}")
    print(f"Saved: {results_dir / 'results.json'}")
    print(f"Saved: {results_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
