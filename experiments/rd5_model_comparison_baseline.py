# pyright: reportMissingImports=false
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rd5_config import cfg
# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportImplicitStringConcatenation=false

import json
import random
import time
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def format_metric(value: float | None) -> str:
    if value is None:
        return "ERR"
    return f"{value:.3f}"


def run_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "error",
        "error": None,
        "r2": None,
        "mae": None,
        "mean_pred": None,
        "std_pred": None,
        "duration_seconds": None,
    }

    try:
        if model_name == "TabPFN":
            from tabpfn import TabPFNRegressor

            model = TabPFNRegressor(
                device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt"
            )
        elif model_name == "TabICL":
            from tabicl import TabICLRegressor

            model = TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)
        elif model_name == "iLTM":
            from iltm import iLTMRegressor

            model = iLTMRegressor(device="cpu", n_ensemble=1, seed=cfg.SEED)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        start = time.perf_counter()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        duration_seconds = time.perf_counter() - start

        preds_np = np.asarray(preds).reshape(-1)
        result.update(
            {
                "status": "ok",
                "r2": r2_score_np(y_test, preds_np),
                "mae": mae_np(y_test, preds_np),
                "mean_pred": float(np.mean(preds_np)),
                "std_pred": float(np.std(preds_np)),
                "duration_seconds": float(duration_seconds),
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {model_name} failed: {result['error']}")

    return result


def main() -> int:
    random_seed = 42
    random.seed(random_seed)

    np.random.seed(42)
    X = np.random.randn(60, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(60) * 0.1

    X_train = X[:50]
    X_test = X[50:]
    y_train = y[:50]
    y_test = y[50:]

    gt_stats = {
        "mean": float(np.mean(y_test)),
        "std": float(np.std(y_test)),
        "n": int(y_test.shape[0]),
    }

    print("Ground truth (y_test) stats:")
    print(f"- n={gt_stats['n']} mean={gt_stats['mean']:.3f} std={gt_stats['std']:.3f}")
    print()

    model_order = ["TabPFN", "TabICL", "iLTM"]
    results: dict[str, dict[str, Any]] = {}

    for name in model_order:
        print(f"Running {name}...")
        results[name] = run_model(name, X_train, y_train, X_test, y_test)

    print()
    print("| Model   | R²    | MAE   | Mean Pred | Std Pred | Time (s) |")
    print("|---------|-------|-------|-----------|----------|----------|")
    for name in model_order:
        row = results[name]
        time_str = format_metric(row["duration_seconds"])
        print(
            f"| {name:<7} | {format_metric(row['r2']):<5} | {format_metric(row['mae']):<5} | {format_metric(row['mean_pred']):<9} | {format_metric(row['std_pred']):<8} | {time_str:<8} |"
        )

    output = {
        "random_seed": random_seed,
        "data": {
            "n_samples": 60,
            "n_features": 2,
            "train_size": 50,
            "test_size": 10,
            "formula": "y = 2*x0 + 3*x1 + N(0, 0.1)",
        },
        "ground_truth": gt_stats,
        "models": results,
    }

    output_path = ROOT / "results" / "rd5" / "baseline" / "comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Saved comparison JSON to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
