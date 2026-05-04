# pyright: reportMissingImports=false
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from tabpfn import TabPFNRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rd5_config import cfg

from src.data.classification_generator import generate_linear_classification  # noqa: E402
from src.hooks.tabpfn_hooker import TabPFNHookedModel  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_TRAIN = 40
N_TEST = 20
N_LAYERS = 12
ALPHA = 2.0
BETA = 3.0
TSNE_LAYERS = [0, 4, 8, 11]


# ---------------------------------------------------------------------------
# CKA computation
# ---------------------------------------------------------------------------
def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered Kernel Alignment between X [n,d1] and Y [n,d2]."""
    n = X.shape[0]
    K_X = X @ X.T  # [n, n]
    K_Y = Y @ Y.T  # [n, n]
    H = np.eye(n) - np.ones((n, n)) / n  # centering matrix
    K_X_c = H @ K_X @ H
    K_Y_c = H @ K_Y @ H
    hsic_xy = np.trace(K_X_c @ K_Y_c) / (n - 1) ** 2
    hsic_xx = np.trace(K_X_c @ K_X_c) / (n - 1) ** 2
    hsic_yy = np.trace(K_Y_c @ K_Y_c) / (n - 1) ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 1e-10 else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_numpy(array_like: Any) -> np.ndarray:
    """Convert tensor/array to numpy."""
    import torch

    if isinstance(array_like, np.ndarray):
        return array_like
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


# ---------------------------------------------------------------------------
# Extract label-token activations for all layers
# ---------------------------------------------------------------------------
def extract_all_layer_activations(
    cache: dict[str, Any],
    single_eval_pos: int,
) -> list[np.ndarray]:
    """Extract train-sample label-token activations for all 12 layers.

    Args:
        cache: Cache dict from forward_with_cache.
        single_eval_pos: Number of training samples (= N_train).

    Returns:
        List of 12 arrays, each [N_train, 192].
    """
    layer_acts: list[np.ndarray] = []
    for i in range(N_LAYERS):
        act = cache["layers"][i]  # [1, N_train+N_test, fb+1, 192]
        train_act = act[0, :single_eval_pos, -1, :]  # [N_train, 192]
        layer_acts.append(_to_numpy(train_act))
    return layer_acts


# ---------------------------------------------------------------------------
# t-SNE visualization
# ---------------------------------------------------------------------------
def plot_tsne(
    layer_acts: list[np.ndarray],
    labels: np.ndarray,
    tsne_layers: list[int],
    out_dir: Path,
) -> dict[str, Any]:
    """Run t-SNE on selected layers and save scatter plots.

    Args:
        layer_acts: All 12 layer activations, each [N_train, 192].
        labels: Integer class labels [N_train].
        tsne_layers: Layer indices to visualize.
        out_dir: Output directory for PNG files.

    Returns:
        Dict with t-SNE metadata per layer.
    """
    perplexity = min(15, N_TRAIN // 3)
    tsne_meta: dict[str, Any] = {}

    print(f"\n[t-SNE] perplexity={perplexity}, layers={tsne_layers}")

    for layer_idx in tsne_layers:
        act = layer_acts[layer_idx]  # [N_train, 192]

        tsne = TSNE(
            n_components=2,
            random_state=RANDOM_SEED,
            perplexity=perplexity,
        )
        emb = tsne.fit_transform(act)  # [N_train, 2]

        # Plot scatter colored by class label (0=blue, 1=red)
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ["#3498db", "#e74c3c"]
        class_names = ["Class 0", "Class 1"]

        for cls_val in [0, 1]:
            mask = labels == cls_val
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                c=colors[cls_val],
                label=class_names[cls_val],
                s=60,
                alpha=0.75,
                edgecolors="white",
                linewidth=0.5,
            )

        ax.set_xlabel("t-SNE dim 1", fontsize=12)
        ax.set_ylabel("t-SNE dim 2", fontsize=12)
        ax.set_title(
            f"Layer {layer_idx} — Label Token t-SNE\n"
            f"(N={N_TRAIN}, perplexity={perplexity})",
            fontsize=13,
        )
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        save_path = out_dir / f"tsne_layer_{layer_idx}.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {save_path.name}")

        tsne_meta[f"layer_{layer_idx}"] = {
            "kl_divergence": float(tsne.kl_divergence_),
            "n_iter": int(tsne.n_iter_),
            "perplexity": perplexity,
        }

    return tsne_meta


# ---------------------------------------------------------------------------
# CKA matrix computation & visualization
# ---------------------------------------------------------------------------
def compute_cka_matrix(layer_acts: list[np.ndarray]) -> np.ndarray:
    """Build 12×12 CKA similarity matrix.

    Args:
        layer_acts: All 12 layer activations, each [N_train, 192].

    Returns:
        np.ndarray of shape [12, 12].
    """
    cka_matrix = np.zeros((N_LAYERS, N_LAYERS))
    for i in range(N_LAYERS):
        for j in range(N_LAYERS):
            cka_matrix[i, j] = compute_cka(layer_acts[i], layer_acts[j])
    return cka_matrix


def plot_cka_heatmap(cka_matrix: np.ndarray, out_dir: Path) -> None:
    """Plot CKA heatmap and save to disk.

    Uses seaborn if available, falls back to matplotlib imshow.

    Args:
        cka_matrix: [12, 12] CKA similarity matrix.
        out_dir: Output directory for PNG.
    """
    tick_labels = [f"L{i}" for i in range(N_LAYERS)]

    try:
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            cka_matrix,
            ax=ax,
            vmin=0,
            vmax=1,
            cmap="viridis",
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
        )
    except ImportError:
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cka_matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(N_LAYERS))
        ax.set_yticks(range(N_LAYERS))
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        plt.colorbar(im, ax=ax)
        # Add text annotations
        for i in range(N_LAYERS):
            for j in range(N_LAYERS):
                ax.text(
                    j,
                    i,
                    f"{cka_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if cka_matrix[i, j] < 0.5 else "black",
                )

    ax.set_title("CKA Similarity Matrix (12×12 Layers)", fontsize=14)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)

    fig.tight_layout()
    save_path = out_dir / "cka_matrix.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_representation_geometry() -> dict[str, Any]:
    """Run representation geometry analysis: t-SNE + CKA.

    Returns:
        Results dict with CKA matrix, t-SNE metadata, and key findings.
    """
    print("=" * 72)
    print("RD-7 M7-T3: Representation Geometry (t-SNE + CKA)")
    print("=" * 72)
    print(f"RANDOM_SEED={RANDOM_SEED}, N_TRAIN={N_TRAIN}, N_TEST={N_TEST}")
    print(f"Decision boundary: {ALPHA}*x1 + {BETA}*x2 > 0")

    # --- 1. Generate data ---
    print("\n[1] Generating linear classification data...")
    dataset = generate_linear_classification(
        alpha=ALPHA,
        beta=BETA,
        n_train=N_TRAIN,
        n_test=N_TEST,
        random_seed=RANDOM_SEED,
    )
    print(f"    X_train: {dataset.X_train.shape}, y_train: {dataset.y_train.shape}")
    print(f"    Class distribution: {np.bincount(dataset.y_train)}")
    print(f"    Description: {dataset.description}")

    # --- 2. Fit TabPFNRegressor with binary targets as float ---
    print("\n[2] Fitting TabPFNRegressor with binary targets (y as float)...")
    y_train_float = dataset.y_train.astype(float)  # {0, 1} → {0.0, 1.0}

    model = TabPFNRegressor(
        device=cfg.DEVICE,
        model_path=str(ROOT / "tabpfn-v2-regressor.ckpt"),
    )
    model.fit(dataset.X_train, y_train_float)

    # --- 3. Forward pass with activation caching ---
    print("\n[3] Running forward pass with activation caching...")
    hooker = TabPFNHookedModel(model)
    preds, cache = hooker.forward_with_cache(dataset.X_test)
    single_eval_pos = int(cache["single_eval_pos"])
    print(f"    single_eval_pos (N_train): {single_eval_pos}")
    print(f"    Predictions shape: {preds.shape}")

    # --- 4. Extract label-token activations for all 12 layers ---
    print("\n[4] Extracting label-token activations (train samples)...")
    start_time = time.time()
    layer_acts = extract_all_layer_activations(cache, single_eval_pos)
    labels = dataset.y_train  # [N_train] — integer class labels
    for i in [0, 5, 11]:
        print(f"    Layer {i}: shape={layer_acts[i].shape}")
    print(f"    Extraction time: {time.time() - start_time:.2f}s")

    # --- 5. Output directory ---
    out_dir = ROOT / "results" / "rd7" / "geometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 6. t-SNE for layers [0, 4, 8, 11] ---
    print("\n[5] Running t-SNE...")
    start_time = time.time()
    tsne_meta = plot_tsne(layer_acts, labels, TSNE_LAYERS, out_dir)
    print(f"    t-SNE time: {time.time() - start_time:.2f}s")

    # --- 7. CKA matrix [12×12] ---
    print("\n[6] Computing CKA matrix...")
    start_time = time.time()
    cka_matrix = compute_cka_matrix(layer_acts)
    print(f"    CKA computation time: {time.time() - start_time:.2f}s")
    print(f"    Diagonal (self-similarity): {np.diag(cka_matrix).tolist()}")

    # --- 8. CKA heatmap ---
    print("\n[7] Plotting CKA heatmap...")
    plot_cka_heatmap(cka_matrix, out_dir)

    # --- 9. Key findings ---
    early_late_cka = float(cka_matrix[0, 11])
    mid_block_cka = float(np.mean(cka_matrix[4:9, 4:9]))
    adjacent_cka = float(np.mean([cka_matrix[i, i + 1] for i in range(N_LAYERS - 1)]))

    print(f"\n{'=' * 72}")
    print("Key Findings:")
    print(f"  CKA(L0, L11) — early vs late:     {early_late_cka:.4f}")
    print(f"  CKA(L4-L8) — mid block avg:       {mid_block_cka:.4f}")
    print(f"  CKA(adjacent layers) — avg:        {adjacent_cka:.4f}")
    print(
        f"  CKA diagonal — all 1.0:            {np.allclose(np.diag(cka_matrix), 1.0)}"
    )

    # --- 10. Build results ---
    results: dict[str, Any] = {
        "cka_matrix": cka_matrix.tolist(),
        "tsne_layers": TSNE_LAYERS,
        "tsne_metadata": tsne_meta,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "random_seed": RANDOM_SEED,
        "dataset": "linear_classification_alpha2_beta3",
        "key_findings": {
            "early_late_cka": early_late_cka,
            "mid_block_cka": mid_block_cka,
            "adjacent_cka_avg": adjacent_cka,
        },
    }

    return results


# ---------------------------------------------------------------------------
# Notepad append
# ---------------------------------------------------------------------------
def append_notepad(results: dict[str, Any]) -> None:
    """Append findings to learnings.md notepad (never overwrite)."""
    notepad_path = ROOT / ".sisyphus" / "notepads" / "tfmi-phase2" / "learnings.md"
    perplexity = min(15, N_TRAIN // 3)
    findings = results["key_findings"]

    with notepad_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## RD-7 M7-T3: rd7_representation_geometry.py\n")
        f.write(f"- t-SNE perplexity: {perplexity} (min(15, N_train//3))\n")
        f.write(f"- CKA matrix diagonal = 1.0 (self-similarity)\n")
        f.write(f"- early_late_cka (L0 vs L11): {findings['early_late_cka']:.3f}\n")
        f.write(f"- mid_block_cka (L4-L8 avg): {findings['mid_block_cka']:.3f}\n")
        f.write(f"- adjacent_cka_avg: {findings['adjacent_cka_avg']:.3f}\n")
    print(f"\nNotepad appended: {notepad_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    out_dir = ROOT / "results" / "rd7" / "geometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_representation_geometry()

    # Save results JSON
    json_path = out_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Append to notepad
    append_notepad(results)

    print(f"\n{'=' * 72}")
    print("RD-7 M7-T3: COMPLETE")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
