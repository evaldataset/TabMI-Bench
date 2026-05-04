# pyright: reportMissingImports=false
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rd5_config import cfg

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false

import json

import matplotlib

matplotlib.use("Agg")
import numpy as np
from sklearn.metrics import r2_score
from tabicl import TabICLRegressor


def _build_model(random_seed: int) -> TabICLRegressor:
    return TabICLRegressor(device=cfg.DEVICE, random_state=random_seed)


def main() -> int:
    random_seed = cfg.SEED

    np.random.seed(random_seed)
    X = np.random.randn(60, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(60) * 0.1

    X_train = X[:50]
    X_test = X[50:]
    y_train = y[:50]
    y_test = y[50:]

    model = _build_model(random_seed=random_seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = float(r2_score(y_test, preds))
    model_name = type(model).__name__
    model_module = type(model).__module__

    print("=== TabICL Regression Smoke Test ===")
    print(f"model: {model_name} ({model_module})")
    print(f"random_seed: {random_seed}")
    print(f"train_shape: {X_train.shape}, test_shape: {X_test.shape}")
    print("predictions:", np.asarray(preds).tolist())
    print(f"r2_score: {r2:.6f}")

    output_dir = ROOT / "results" / "rd5" / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tabicl_smoke.json"

    payload = {
        "model": model_name,
        "module": model_module,
        "random_seed": random_seed,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "r2_score": r2,
        "predictions": np.asarray(preds).tolist(),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"saved_results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
