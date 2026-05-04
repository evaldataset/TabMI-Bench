# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportImplicitStringConcatenation=false
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from iltm import iLTMRegressor
from sklearn.cross_decomposition import CCA
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rd5_config import cfg  # noqa: E402
from src.data.synthetic_generator import generate_linear_data  # noqa: E402
from src.hooks.iltm_hooker import iLTMHookedModel  # noqa: E402
from src.hooks.tabicl_hooker import TabICLHookedModel  # noqa: E402
from src.hooks.tabpfn_hooker import TabPFNHookedModel  # noqa: E402
from src.sae.sparse_autoencoder import SAETrainer, TabPFNSparseAutoencoder  # noqa: E402


RESULTS_DIR = ROOT / "results" / "rd5" / "sae_feature_similarity"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
MODEL_ORDER = ["tabpfn", "tabicl", "iltm"]
MODEL_TITLE = {"tabpfn": "TabPFN", "tabicl": "TabICL", "iltm": "iLTM"}

CORE_LAYER = {"tabpfn": 6, "tabicl": 5, "iltm": 1}
HIDDEN_DIM = {"tabpfn": 192, "tabicl": 512, "iltm": 512}

EXPANSION_FACTOR = 16
SAE_EPOCHS = cfg.sae_epochs
SAE_BATCH_SIZE = cfg.SAE_BATCH_SIZE
SAE_LR = cfg.SAE_LR


@dataclass(frozen=True)
class PairMetric:
    """Container for pairwise similarity metrics.

    Attributes:
        cka: Linear CKA similarity.
        cca: Mean canonical correlation from CCA.
    """

    cka: float
    cca: float


def _build_tabpfn() -> TabPFNRegressor:
    """Build TabPFN regressor from shared config.

    Returns:
        Initialized TabPFNRegressor.
    """

    return TabPFNRegressor(device=cfg.DEVICE, model_path="tabpfn-v2-regressor.ckpt")


def _build_tabicl() -> TabICLRegressor:
    """Build TabICL regressor from shared config.

    Returns:
        Initialized TabICLRegressor.
    """

    return TabICLRegressor(device=cfg.DEVICE, random_state=cfg.SEED)


def _build_iltm() -> iLTMRegressor:
    """Build iLTM regressor from shared config.

    Returns:
        Initialized iLTMRegressor.
    """

    return iLTMRegressor(device="cpu", n_ensemble=1, seed=cfg.SEED)


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representations.

    Args:
        X: Matrix of shape [n_samples, d_x].
        Y: Matrix of shape [n_samples, d_y].

    Returns:
        Linear CKA in [0, 1] (numerically clipped).
    """

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"CKA requires equal n_samples, got {X.shape[0]} vs {Y.shape[0]}"
        )

    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)
    n = x.shape[0]
    if n <= 1:
        return 0.0

    kx = x @ x.T
    ky = y @ y.T
    h = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / float(n)
    kx_c = h @ kx @ h
    ky_c = h @ ky @ h

    hsic_xy = np.trace(kx_c @ ky_c) / (n - 1) ** 2
    hsic_xx = np.trace(kx_c @ kx_c) / (n - 1) ** 2
    hsic_yy = np.trace(ky_c @ ky_c) / (n - 1) ** 2
    denom = np.sqrt(max(hsic_xx, 0.0) * max(hsic_yy, 0.0))
    if denom <= 1e-12:
        return 0.0
    score = float(hsic_xy / denom)
    return float(np.clip(score, 0.0, 1.0))


def _cca_similarity(X: np.ndarray, Y: np.ndarray, max_components: int = 32) -> float:
    """Compute average canonical correlation between matrices.

    Args:
        X: Matrix [n_samples, d_x].
        Y: Matrix [n_samples, d_y].
        max_components: Maximum number of CCA components.

    Returns:
        Mean canonical correlation, clipped to [0, 1].
    """

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"CCA requires equal n_samples, got {X.shape[0]} vs {Y.shape[0]}"
        )

    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)
    if x.shape[0] < 3:
        return 0.0

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    n_comp = min(max_components, x.shape[0] - 1, x.shape[1], y.shape[1])
    if n_comp < 1:
        return 0.0

    try:
        cca = CCA(n_components=n_comp, max_iter=2000)
        x_c, y_c = cca.fit_transform(x, y)
    except Exception:
        return 0.0

    corr_values: list[float] = []
    for i in range(n_comp):
        xv = x_c[:, i]
        yv = y_c[:, i]
        sx = float(np.std(xv))
        sy = float(np.std(yv))
        if sx < 1e-12 or sy < 1e-12:
            continue
        corr = float(np.corrcoef(xv, yv)[0, 1])
        if np.isfinite(corr):
            corr_values.append(abs(corr))

    if not corr_values:
        return 0.0
    return float(np.clip(np.mean(corr_values), 0.0, 1.0))


def _extract_core_activations(
    model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """Fit one model and extract core-layer activations for test points.

    Args:
        model_name: One of tabpfn/tabicl/iltm.
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.

    Returns:
        Activation matrix [n_test, hidden_dim].
    """

    layer_idx = CORE_LAYER[model_name]
    expected_dim = HIDDEN_DIM[model_name]

    if model_name == "tabpfn":
        model = _build_tabpfn()
        model.fit(X_train, y_train)
        hooker = TabPFNHookedModel(model)
        _, cache = hooker.forward_with_cache(X_test)
        acts = hooker.get_test_label_token(cache, layer_idx)
    elif model_name == "tabicl":
        model = _build_tabicl()
        model.fit(X_train, y_train)
        hooker = TabICLHookedModel(model)
        _, cache = hooker.forward_with_cache(X_test)
        acts = hooker.get_layer_activations(cache, layer_idx)
    elif model_name == "iltm":
        model = _build_iltm()
        model.fit(X_train, y_train)
        hooker = iLTMHookedModel(model)
        _, cache = hooker.forward_with_cache(X_test)
        acts = hooker.get_layer_activations(cache, layer_idx)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    acts_np = np.asarray(acts, dtype=np.float32)
    if acts_np.ndim != 2 or acts_np.shape[1] != expected_dim:
        raise ValueError(
            f"Unexpected shape for {model_name}: {acts_np.shape}, expected [n_samples, {expected_dim}]"
        )
    return acts_np


def _build_pairs() -> list[tuple[int, int]]:
    """Build synthetic coefficient pairs for shared data generation.

    Returns:
        List of (alpha, beta) pairs.
    """

    return cfg.coef_pairs("tabpfn")


def _collect_training_activations(
    model_name: str, pairs: list[tuple[int, int]]
) -> np.ndarray:
    """Collect pooled core-layer activations for SAE training.

    Args:
        model_name: One of tabpfn/tabicl/iltm.
        pairs: Synthetic coefficient pairs.

    Returns:
        Concatenated activations [n_total, hidden_dim].
    """

    pooled: list[np.ndarray] = []
    n_pairs = len(pairs)
    for idx, (alpha, beta) in enumerate(pairs):
        ds = generate_linear_data(
            alpha=float(alpha),
            beta=float(beta),
            n_train=cfg.N_TRAIN,
            n_test=cfg.N_TEST,
            random_seed=cfg.SEED + idx,
        )
        pooled.append(
            _extract_core_activations(model_name, ds.X_train, ds.y_train, ds.X_test)
        )
        if model_name == "iltm" and (idx + 1) % 5 == 0:
            print(f"  [iltm] pair {idx + 1}/{n_pairs} done", flush=True)
    return np.concatenate(pooled, axis=0)


def _checkpoint_path(model_name: str) -> Path:
    """Resolve SAE checkpoint path for a model.

    Args:
        model_name: One of tabpfn/tabicl/iltm.

    Returns:
        Filesystem path for the checkpoint.
    """

    return CHECKPOINT_DIR / f"{model_name}_sae_expansion{EXPANSION_FACTOR}.pt"


def _new_sae(model_name: str) -> TabPFNSparseAutoencoder:
    """Instantiate SAE with model-specific input dimension.

    Args:
        model_name: One of tabpfn/tabicl/iltm.

    Returns:
        SAE instance.
    """

    return TabPFNSparseAutoencoder(
        input_dim=HIDDEN_DIM[model_name],
        expansion_factor=EXPANSION_FACTOR,
    )


def _load_or_train_sae(
    model_name: str, train_acts: np.ndarray
) -> TabPFNSparseAutoencoder:
    """Load a pre-trained SAE; if absent, train and save one first.

    Args:
        model_name: One of tabpfn/tabicl/iltm.
        train_acts: Training activations [n_total, hidden_dim].

    Returns:
        Loaded SAE on cfg.DEVICE.
    """

    device = torch.device(cfg.DEVICE)
    ckpt_path = _checkpoint_path(model_name)
    sae = _new_sae(model_name).to(device)

    if not ckpt_path.exists():
        trainer = SAETrainer(sae, lr=SAE_LR, l1_coeff=1e-3)
        trainer.train(
            activations=np.asarray(train_acts, dtype=np.float32),
            epochs=SAE_EPOCHS,
            batch_size=SAE_BATCH_SIZE,
            verbose=False,
        )
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(sae.state_dict(), ckpt_path)

    loaded = _new_sae(model_name).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    loaded.load_state_dict(state_dict)
    loaded.eval()
    return loaded


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    """L2-normalize rows with numerical stability.

    Args:
        mat: Input matrix [n_rows, n_cols].

    Returns:
        Row-normalized matrix with same shape.
    """

    x = np.asarray(mat, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _align_rows_for_pair(
    A: np.ndarray, B: np.ndarray, pair_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample rows to equal counts for cross-model comparison.

    Args:
        A: Matrix [n_a, d_a].
        B: Matrix [n_b, d_b].
        pair_seed: Seed used for deterministic row sampling.

    Returns:
        Tuple of aligned matrices [n_min, d_a] and [n_min, d_b].
    """

    n_min = min(A.shape[0], B.shape[0])
    if A.shape[0] == n_min:
        a_idx = np.arange(n_min)
    else:
        a_idx = np.random.default_rng(pair_seed).choice(
            A.shape[0], size=n_min, replace=False
        )
    if B.shape[0] == n_min:
        b_idx = np.arange(n_min)
    else:
        b_idx = np.random.default_rng(pair_seed + 11).choice(
            B.shape[0], size=n_min, replace=False
        )
    return A[a_idx], B[b_idx]


def _pair_seed(name_a: str, name_b: str) -> int:
    """Build deterministic pair seed.

    Args:
        name_a: First model name.
        name_b: Second model name.

    Returns:
        Integer seed.
    """

    mapping = {"tabpfn": 101, "tabicl": 202, "iltm": 303}
    return cfg.SEED + mapping[name_a] + mapping[name_b]


def _pair_metrics(A: np.ndarray, B: np.ndarray, name_a: str, name_b: str) -> PairMetric:
    """Compute CKA and CCA for one model pair.

    Args:
        A: Representation matrix for model A.
        B: Representation matrix for model B.
        name_a: Model A name.
        name_b: Model B name.

    Returns:
        PairMetric with CKA and CCA.
    """

    aligned_a, aligned_b = _align_rows_for_pair(A, B, _pair_seed(name_a, name_b))
    return PairMetric(
        cka=_linear_cka(aligned_a, aligned_b),
        cca=_cca_similarity(aligned_a, aligned_b),
    )


def _compute_pairwise(reps: dict[str, np.ndarray]) -> dict[str, PairMetric]:
    """Compute pairwise metrics across all model pairs.

    Args:
        reps: Map model_name -> representation matrix.

    Returns:
        Pair key map: "a__b" -> PairMetric.
    """

    out: dict[str, PairMetric] = {}
    for name_a, name_b in combinations(MODEL_ORDER, 2):
        out[f"{name_a}__{name_b}"] = _pair_metrics(
            reps[name_a], reps[name_b], name_a, name_b
        )
    return out


def _matrix_from_pairwise(
    pairwise: dict[str, PairMetric], metric_key: str
) -> np.ndarray:
    """Build symmetric 3x3 matrix from pairwise metrics.

    Args:
        pairwise: Pairwise metric map.
        metric_key: Either "cka" or "cca".

    Returns:
        Symmetric matrix in MODEL_ORDER.
    """

    mat = np.eye(len(MODEL_ORDER), dtype=np.float64)
    idx = {name: i for i, name in enumerate(MODEL_ORDER)}
    for pair_key, metric in pairwise.items():
        left, right = pair_key.split("__")
        v = metric.cka if metric_key == "cka" else metric.cca
        i, j = idx[left], idx[right]
        mat[i, j] = v
        mat[j, i] = v
    return mat


def _collect_sae_features(
    saes: dict[str, TabPFNSparseAutoencoder],
    train_acts: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Encode pre-collected activations through SAEs.

    Args:
        saes: Loaded SAE map model_name -> SAE.
        train_acts: Pre-collected activations from Phase 1 (reuse to avoid
            expensive re-extraction, especially for iLTM).

    Returns:
        Map model_name -> pooled SAE activation matrix.
    """

    device = torch.device(cfg.DEVICE)
    out: dict[str, np.ndarray] = {}

    for model_name in MODEL_ORDER:
        with torch.no_grad():
            feat = saes[model_name].encode(
                torch.from_numpy(
                    train_acts[model_name].astype(np.float32, copy=False)
                ).to(device)
            )
        out[model_name] = feat.detach().cpu().numpy()

    return out


def _plot_heatmaps(
    dict_cka: np.ndarray,
    dict_cca: np.ndarray,
    act_cka: np.ndarray,
    act_cca: np.ndarray,
    save_path: Path,
) -> None:
    """Plot similarity heatmaps and save as PNG.

    Args:
        dict_cka: SAE dictionary CKA matrix.
        dict_cca: SAE dictionary CCA matrix.
        act_cka: SAE activation CKA matrix.
        act_cca: SAE activation CCA matrix.
        save_path: Output image path.
    """

    labels = [MODEL_TITLE[m] for m in MODEL_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        (dict_cka, "Dictionary Similarity (CKA)"),
        (dict_cca, "Dictionary Similarity (CCA)"),
        (act_cka, "SAE Activation Similarity (CKA)"),
        (act_cca, "SAE Activation Similarity (CCA)"),
    ]

    for ax, (mat, title) in zip(axes.ravel(), panels, strict=True):
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="magma")
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(
                    j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white"
                )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Run cross-model SAE feature similarity experiment.

    Returns:
        Process exit code (0 for success).
    """

    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = _build_pairs()
    print("=" * 80)
    print("RD5 SAE Feature Similarity: TabPFN vs TabICL vs iLTM")
    print("=" * 80)
    print(
        f"QUICK_RUN={cfg.QUICK_RUN}, seed={cfg.SEED}, device={cfg.DEVICE}, "
        f"pairs={len(pairs)}, n_train={cfg.N_TRAIN}, n_test={cfg.N_TEST}"
    )

    train_acts: dict[str, np.ndarray] = {}
    for model_name in MODEL_ORDER:
        print(
            f"[{model_name}] extracting core activations ({len(pairs)} pairs)...",
            flush=True,
        )
        train_acts[model_name] = _collect_training_activations(model_name, pairs)
        print(
            f"[{model_name}] pooled core activations: {train_acts[model_name].shape}",
            flush=True,
        )

    saes: dict[str, TabPFNSparseAutoencoder] = {}
    dictionaries: dict[str, np.ndarray] = {}
    for model_name in MODEL_ORDER:
        sae = _load_or_train_sae(model_name, train_acts[model_name])
        saes[model_name] = sae
        dictionary = sae.encoder.weight.detach().cpu().numpy()
        dictionaries[model_name] = _normalize_rows(dictionary)
        print(
            f"[{model_name}] SAE loaded: dict_shape={dictionary.shape}, "
            f"checkpoint={_checkpoint_path(model_name)}"
        )

    dict_pairwise = _compute_pairwise(dictionaries)
    dict_cka = _matrix_from_pairwise(dict_pairwise, metric_key="cka")
    dict_cca = _matrix_from_pairwise(dict_pairwise, metric_key="cca")

    sae_features = _collect_sae_features(saes, train_acts)
    for model_name in MODEL_ORDER:
        print(
            f"[{model_name}] SAE features on shared data: {sae_features[model_name].shape}"
        )

    act_pairwise = _compute_pairwise(sae_features)
    act_cka = _matrix_from_pairwise(act_pairwise, metric_key="cka")
    act_cca = _matrix_from_pairwise(act_pairwise, metric_key="cca")

    heatmap_path = RESULTS_DIR / "similarity_heatmaps.png"
    _plot_heatmaps(dict_cka, dict_cca, act_cka, act_cca, heatmap_path)

    result_payload = {
        "config": {
            "quick_run": cfg.QUICK_RUN,
            "seed": cfg.SEED,
            "device": cfg.DEVICE,
            "n_train": cfg.N_TRAIN,
            "n_test": cfg.N_TEST,
            "pairs": pairs,
            "core_layer": CORE_LAYER,
            "hidden_dim": HIDDEN_DIM,
            "expansion_factor": EXPANSION_FACTOR,
            "sae_epochs": SAE_EPOCHS,
            "sae_batch_size": SAE_BATCH_SIZE,
            "sae_lr": SAE_LR,
        },
        "dictionary_pairwise": {
            k: {"cka": v.cka, "cca": v.cca} for k, v in dict_pairwise.items()
        },
        "activation_pairwise": {
            k: {"cka": v.cka, "cca": v.cca} for k, v in act_pairwise.items()
        },
        "dictionary_matrix": {
            "order": MODEL_ORDER,
            "cka": dict_cka.tolist(),
            "cca": dict_cca.tolist(),
        },
        "activation_matrix": {
            "order": MODEL_ORDER,
            "cka": act_cka.tolist(),
            "cca": act_cca.tolist(),
        },
        "artifacts": {
            "heatmap_path": str(heatmap_path),
            "checkpoints": {m: str(_checkpoint_path(m)) for m in MODEL_ORDER},
        },
    }

    json_path = RESULTS_DIR / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    print("\nPairwise dictionary similarity:")
    for pair_key, metric in dict_pairwise.items():
        print(f"  {pair_key}: CKA={metric.cka:.4f}, CCA={metric.cca:.4f}")

    print("\nPairwise SAE activation similarity (same synthetic inputs):")
    for pair_key, metric in act_pairwise.items():
        print(f"  {pair_key}: CKA={metric.cka:.4f}, CCA={metric.cca:.4f}")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {heatmap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
