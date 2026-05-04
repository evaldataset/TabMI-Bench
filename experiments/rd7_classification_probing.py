# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from tabpfn import TabPFNRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rd5_config import cfg

from src.data.classification_generator import generate_linear_classification  # noqa: E402
from src.hooks.tabpfn_hooker import TabPFNHookedModel  # noqa: E402
from src.probing.linear_probe import LinearProbe, probe_layer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUICK_RUN = True
RANDOM_SEED = 42
N_LAYERS = 12
ALPHA = 2.0
BETA = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_numpy(array_like: Any) -> np.ndarray:
    """Convert tensor/array to numpy."""
    if isinstance(array_like, np.ndarray):
        return array_like
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _extract_train_label_token(cache: dict[str, Any], layer_idx: int) -> np.ndarray:
    """Extract training samples' label token activations for a given layer.

    Returns:
        np.ndarray of shape [N_train, 192]
    """
    single_eval_pos = int(cache["single_eval_pos"])
    layer_tensor = cache["layers"][layer_idx]
    # [batch=1, :N_train, last_feature_block, emsize] → [N_train, 192]
    act = layer_tensor[0, :single_eval_pos, -1, :]
    return _to_numpy(act)


# ---------------------------------------------------------------------------
# Main probing
# ---------------------------------------------------------------------------
def run_classification_probing() -> dict[str, Any]:
    """Run decision boundary probing across all 12 layers.

    Three probing targets:
        1. Class label (y ∈ {0,1}) — can we decode the true class?
        2. Decision score (α*x1 + β*x2) — can we decode the continuous score?
        3. x1 feature value — can we decode the first feature?
    """
    print("=" * 72)
    print("RD-7 M7-T2: Classification Decision Boundary Probing")
    print("=" * 72)
    print(f"QUICK_RUN={QUICK_RUN}, RANDOM_SEED={RANDOM_SEED}")
    print(f"Decision boundary: {ALPHA}*x1 + {BETA}*x2 > 0")

    # --- 1. Generate data ---
    print("\n[1] Generating linear classification data...")
    ds = generate_linear_classification(
        alpha=ALPHA,
        beta=BETA,
        n_train=100,
        n_test=20,
        random_seed=RANDOM_SEED,
    )
    print(f"    X_train: {ds.X_train.shape}, y_train: {ds.y_train.shape}")
    print(f"    Class distribution: {np.bincount(ds.y_train)}")
    print(f"    Description: {ds.description}")

    # --- 2. Fit TabPFNRegressor with binary targets as float ---
    print("\n[2] Fitting TabPFNRegressor with binary targets (y as float)...")
    y_train_float = ds.y_train.astype(float)  # {0, 1} → {0.0, 1.0}

    model = TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")
    model.fit(ds.X_train, y_train_float)

    # --- 3. Forward pass with activation caching ---
    print("\n[3] Running forward pass with activation caching...")
    hooker = TabPFNHookedModel(model)
    preds, cache = hooker.forward_with_cache(ds.X_test)
    single_eval_pos = int(cache["single_eval_pos"])
    print(f"    single_eval_pos (N_train): {single_eval_pos}")
    print(f"    Predictions shape: {preds.shape}")

    # Classification accuracy from regression output
    pred_classes = (preds > 0.5).astype(int)
    test_accuracy = float(accuracy_score(ds.y_test, pred_classes))
    print(f"    Test accuracy (pred > 0.5): {test_accuracy:.4f}")

    # --- 4. Prepare probing targets (train set) ---
    y_class_train = ds.y_train.astype(float)  # class label {0.0, 1.0}
    score_train = ALPHA * ds.X_train[:, 0] + BETA * ds.X_train[:, 1]  # decision score
    x1_train = ds.X_train[:, 0].copy()  # first feature

    targets = {
        "class_label": y_class_train,
        "decision_score": score_train,
        "x1_feature": x1_train,
    }

    print("\n[4] Probing targets (train set):")
    for name, t in targets.items():
        print(f"    {name}: mean={t.mean():.3f}, std={t.std():.3f}")

    # --- 5. Probe each layer for each target (complexity=0) ---
    print(f"\n[5] Probing all {N_LAYERS} layers (complexity=0, linear)...")
    r2_per_target: dict[str, list[float]] = {name: [] for name in targets}

    start_time = time.time()
    for layer_idx in range(N_LAYERS):
        act = _extract_train_label_token(cache, layer_idx)  # [N_train, 192]

        for target_name, target_values in targets.items():
            result = probe_layer(
                act, target_values, complexities=[0], random_seed=RANDOM_SEED
            )
            r2 = result[0]["r2"]
            r2_per_target[target_name].append(r2)

        elapsed = time.time() - start_time
        print(
            f"    Layer {layer_idx:>2}: "
            f"class={r2_per_target['class_label'][-1]:.4f}, "
            f"score={r2_per_target['decision_score'][-1]:.4f}, "
            f"x1={r2_per_target['x1_feature'][-1]:.4f} "
            f"({elapsed:.1f}s)"
        )

    # --- 6. Classification accuracy from probed class label predictions ---
    print(f"\n[6] Probed classification accuracy per layer...")
    probe_accuracy_per_layer: list[float] = []
    for layer_idx in range(N_LAYERS):
        act = _extract_train_label_token(cache, layer_idx)

        X_tr, X_te, y_tr, y_te = train_test_split(
            act, ds.y_train, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
        )
        probe = LinearProbe(complexity=0, random_seed=RANDOM_SEED)
        probe.fit(X_tr, y_tr.astype(float))
        y_pred_cont = probe.predict(X_te)
        y_pred_class = (y_pred_cont > 0.5).astype(int)
        acc = float(accuracy_score(y_te, y_pred_class))
        probe_accuracy_per_layer.append(acc)
        print(f"    Layer {layer_idx:>2}: accuracy={acc:.4f}")

    # --- 7. Results summary ---
    r2_class = np.array(r2_per_target["class_label"])
    r2_score_arr = np.array(r2_per_target["decision_score"])
    r2_feature = np.array(r2_per_target["x1_feature"])

    peak_layer_class = int(np.argmax(r2_class))
    peak_layer_score = int(np.argmax(r2_score_arr))
    peak_layer_feature = int(np.argmax(r2_feature))
    best_probe_acc_layer = int(np.argmax(probe_accuracy_per_layer))

    print(f"\n{'=' * 72}")
    print("Summary:")
    print(
        f"  Class label:    peak R²={r2_class[peak_layer_class]:.4f} "
        f"at layer {peak_layer_class}"
    )
    print(
        f"  Decision score: peak R²={r2_score_arr[peak_layer_score]:.4f} "
        f"at layer {peak_layer_score}"
    )
    print(
        f"  x1 feature:     peak R²={r2_feature[peak_layer_feature]:.4f} "
        f"at layer {peak_layer_feature}"
    )
    print(f"  Test accuracy (regression→class): {test_accuracy:.4f}")
    print(
        f"  Best probed accuracy: {max(probe_accuracy_per_layer):.4f} "
        f"at layer {best_probe_acc_layer}"
    )

    return {
        "r2_class_per_layer": r2_class.tolist(),
        "r2_score_per_layer": r2_score_arr.tolist(),
        "r2_feature_per_layer": r2_feature.tolist(),
        "peak_layer_class": peak_layer_class,
        "peak_layer_score": peak_layer_score,
        "peak_layer_feature": peak_layer_feature,
        "probe_accuracy_per_layer": probe_accuracy_per_layer,
        "test_accuracy": test_accuracy,
        "alpha": ALPHA,
        "beta": BETA,
        "quick_run": QUICK_RUN,
        "random_seed": RANDOM_SEED,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_probing_results(results: dict[str, Any], save_path: Path) -> None:
    """Plot R² curves for all three probing targets on a single figure."""
    layers = np.arange(N_LAYERS)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        layers,
        results["r2_class_per_layer"],
        "o-",
        label="Class label (y ∈ {0,1})",
        color="#e74c3c",
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        layers,
        results["r2_score_per_layer"],
        "s-",
        label=f"Decision score ({ALPHA}x₁+{BETA}x₂)",
        color="#2ecc71",
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        layers,
        results["r2_feature_per_layer"],
        "^-",
        label="x₁ feature value",
        color="#3498db",
        linewidth=2,
        markersize=6,
    )

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("R² (linear probe, complexity=0)", fontsize=13)
    ax.set_title(
        "RD-7: Classification Decision Boundary Probing\n"
        f"(α={ALPHA}, β={BETA}, binary classification via TabPFNRegressor)",
        fontsize=14,
    )
    ax.set_xticks(layers)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    results_dir = ROOT / "results" / "rd7" / "probing"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = run_classification_probing()

    # Save plot
    plot_path = results_dir / "classification_probing.png"
    plot_probing_results(results, plot_path)

    # Save results JSON
    json_path = results_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
