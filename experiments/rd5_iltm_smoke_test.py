# pyright: reportMissingImports=false
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rd5_config import cfg

import json

import matplotlib
import numpy as np
from iltm import iLTMRegressor
from sklearn.metrics import r2_score

matplotlib.use("Agg")


def main() -> int:
    random_seed = cfg.SEED
    np.random.seed(random_seed)

    X = np.random.randn(60, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(60) * 0.1

    X_train = X[:50]
    X_test = X[50:]
    y_train = y[:50]
    y_test = y[50:]

    model = iLTMRegressor(device="cpu", n_ensemble=1, seed=random_seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    score = r2_score(y_test, preds)

    print("=== iLTM Regression Smoke Test ===")
    print(f"random_seed: {random_seed}")
    print(f"train_shape: X={X_train.shape}, y={y_train.shape}")
    print(f"test_shape: X={X_test.shape}, y={y_test.shape}")
    print(f"predictions: {preds.tolist()}")
    print(f"r2_score: {score:.6f}")
    print(f"model_class: {model.__class__.__name__}")
    print(f"model_repr: {model}")

    output_path = ROOT / "results" / "rd5" / "baseline" / "iltm_smoke.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "random_seed": random_seed,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "predictions": preds.tolist(),
        "r2_score": float(score),
        "model_info": {
            "model_class": model.__class__.__name__,
            "model_repr": str(model),
            "device": "cpu",
        },
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved_results: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
