# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportImplicitStringConcatenation=false, reportUntypedBaseClass=false, reportUnannotatedClassAttribute=false
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from iltm import iLTMRegressor
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rd5_config import cfg
from src.data.synthetic_generator import generate_linear_data
from src.hooks.iltm_hooker import iLTMHookedModel
from src.hooks.tabicl_hooker import TabICLHookedModel
from src.hooks.tabpfn_hooker import TabPFNHookedModel

from typing import Any

from torch import nn
from torch.nn import functional as F


RESULTS_DIR = ROOT / "results" / "rd5" / "usae_cross_model"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

MODEL_ORDER = ["tabpfn", "tabicl", "iltm"]
MODEL_TITLE = {"tabpfn": "TabPFN", "tabicl": "TabICL", "iltm": "iLTM"}
CORE_LAYER = {"tabpfn": 6, "tabicl": 5, "iltm": 1}
HIDDEN_DIM = {"tabpfn": 192, "tabicl": 512, "iltm": 512}

N_CONCEPTS = 4096
TOPK = 32
AUX_K = N_CONCEPTS // 2
AUX_PENALTY = 0.1
SAE_LR = 3e-4


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
        Initialized iLTMRegressor on CPU.
    """

    return iLTMRegressor(device="cpu", n_ensemble=1, seed=cfg.SEED)


def _topk_sparse(codes: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only top-k values per row.

    Args:
        codes: Dense activation tensor of shape [batch, n_concepts].
        k: Number of non-zero entries to keep per row.

    Returns:
        Sparse tensor with same shape as ``codes``.
    """

    if k <= 0:
        return torch.zeros_like(codes)
    k_eff = min(k, int(codes.shape[1]))
    values, indices = torch.topk(codes, k=k_eff, dim=1)
    out = torch.zeros_like(codes)
    out.scatter_(1, indices, values)
    return out


class USAEModel(nn.Module):
    """USAE branch for one model with private encoder/decoder.

    Encoder: Linear(input_dim -> n_concepts) + BatchNorm1d
    Activation: ReLU + TopK
    Decoder: Linear(n_concepts -> input_dim) with column-normalized weights
    """

    def __init__(self, input_dim: int, n_concepts: int, topk: int) -> None:
        """Initialize model-specific USAE branch.

        Args:
            input_dim: Model hidden size.
            n_concepts: Shared concept count.
            topk: Number of active concepts per sample.
        """

        super().__init__()
        self.input_dim = input_dim
        self.n_concepts = n_concepts
        self.topk = topk
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_concepts),
            nn.BatchNorm1d(n_concepts),
        )
        self.decoder = nn.Linear(n_concepts, input_dim)

    def _decoder_weight_normalized(self) -> torch.Tensor:
        """Return decoder weights with L2-normalized columns.

        Returns:
            Weight tensor of shape [input_dim, n_concepts].
        """

        weight = self.decoder.weight
        norms = torch.linalg.norm(weight, dim=0, keepdim=True).clamp_min(1e-12)
        return weight / norms

    def encode_with_pretopk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode activations and expose pre-TopK codes.

        Args:
            x: Input activations of shape [batch, input_dim].

        Returns:
            Tuple of (pre_topk_relu, topk_codes), each [batch, n_concepts].
        """

        pre_topk = F.relu(self.encoder(x))
        topk_codes = _topk_sparse(pre_topk, self.topk)
        return pre_topk, topk_codes

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse concept codes to activation space.

        Args:
            z: Sparse codes of shape [batch, n_concepts].

        Returns:
            Reconstructed activations of shape [batch, input_dim].
        """

        return F.linear(z, self._decoder_weight_normalized(), self.decoder.bias)

    def dictionary(self) -> torch.Tensor:
        """Return concept-to-activation matrix.

        Returns:
            Matrix of shape [n_concepts, input_dim].
        """

        return self._decoder_weight_normalized().T


def _extract_core_activations(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Fit model and extract target core-layer activations.

    Args:
        model_name: One of tabpfn/tabicl/iltm.
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.

    Returns:
        Activation array of shape [n_test, hidden_dim].
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


def _collect_shared_activations() -> tuple[
    dict[str, np.ndarray], list[tuple[int, int]]
]:
    """Collect pooled activations on shared synthetic coefficient pairs.

    Returns:
        Tuple of:
            - model_name -> pooled activations [n_total, hidden_dim]
            - coefficient pair list used for generation
    """

    pairs = cfg.coef_pairs("tabpfn")
    pooled: dict[str, list[np.ndarray]] = {name: [] for name in MODEL_ORDER}
    n_pairs = len(pairs)

    for idx, (alpha, beta) in enumerate(pairs):
        ds = generate_linear_data(
            alpha=float(alpha),
            beta=float(beta),
            n_train=cfg.N_TRAIN,
            n_test=cfg.N_TEST,
            random_seed=cfg.SEED + idx,
        )
        for model_name in MODEL_ORDER:
            acts = _extract_core_activations(
                model_name=model_name,
                X_train=ds.X_train,
                y_train=ds.y_train,
                X_test=ds.X_test,
            )
            pooled[model_name].append(acts)

            if model_name == "iltm" and (idx + 1) % 5 == 0:
                print(f"  [iltm] pair {idx + 1}/{n_pairs} done", flush=True)

    return {k: np.concatenate(v, axis=0) for k, v in pooled.items()}, pairs


def _compute_loss_for_target(
    target_x: torch.Tensor,
    codes_topk: torch.Tensor,
    pre_topk: torch.Tensor,
    decoder_model: USAEModel,
) -> torch.Tensor:
    """Compute USAE reconstruction + auxiliary revival loss for one decoder.

    Args:
        target_x: Target activations [batch, input_dim_j].
        codes_topk: Source top-k concept codes [batch, n_concepts].
        pre_topk: Source pre-top-k ReLU codes [batch, n_concepts].
        decoder_model: Decoder owner model branch.

    Returns:
        Scalar total loss.
    """

    x_hat = decoder_model.decode(codes_topk)
    residual = target_x - x_hat
    main_loss = residual.abs().mean()

    aux_codes = F.relu(pre_topk) - codes_topk
    aux_topk = _topk_sparse(aux_codes, AUX_K)
    aux_hat = aux_topk @ decoder_model.dictionary()
    aux_loss = (residual - aux_hat).abs().mean()
    return main_loss + AUX_PENALTY * aux_loss


def _train_usae(activations: dict[str, np.ndarray]) -> dict[str, USAEModel]:
    """Train USAE with private encoders/decoders and shared concept space.

    Args:
        activations: model_name -> activations [n_total, hidden_dim].

    Returns:
        Trained USAE branches per model.
    """

    device = torch.device(cfg.DEVICE)
    n_samples = activations[MODEL_ORDER[0]].shape[0]

    tensors = {
        name: torch.from_numpy(activations[name].astype(np.float32, copy=False)).to(
            device
        )
        for name in MODEL_ORDER
    }

    usaes = {
        name: USAEModel(HIDDEN_DIM[name], N_CONCEPTS, TOPK).to(device)
        for name in MODEL_ORDER
    }
    optimizers = {
        name: torch.optim.Adam(usaes[name].parameters(), lr=SAE_LR)
        for name in MODEL_ORDER
    }

    n_batches = max(1, int(np.ceil(n_samples / float(cfg.SAE_BATCH_SIZE))))
    rng = np.random.default_rng(cfg.SEED)

    for epoch in range(cfg.sae_epochs):
        for model in usaes.values():
            model.train()

        order = rng.permutation(n_samples)
        for batch_idx in range(n_batches):
            start = batch_idx * cfg.SAE_BATCH_SIZE
            end = min((batch_idx + 1) * cfg.SAE_BATCH_SIZE, n_samples)
            idx = order[start:end]
            if idx.size == 0:
                continue

            source_name = MODEL_ORDER[int(rng.integers(low=0, high=len(MODEL_ORDER)))]
            source_batch = tensors[source_name][idx]

            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)

            pre_topk, codes_topk = usaes[source_name].encode_with_pretopk(source_batch)

            total_loss = torch.zeros((), device=device)
            for target_name in MODEL_ORDER:
                target_batch = tensors[target_name][idx]
                total_loss = total_loss + _compute_loss_for_target(
                    target_x=target_batch,
                    codes_topk=codes_topk,
                    pre_topk=pre_topk,
                    decoder_model=usaes[target_name],
                )

            total_loss.backward()
            for opt in optimizers.values():
                opt.step()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.sae_epochs:
            print(
                f"[train] epoch {epoch + 1}/{cfg.sae_epochs} done",
                flush=True,
            )

    for model in usaes.values():
        model.eval()
    return usaes


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute global R-squared with variance baseline.

    Args:
        y_true: Ground-truth matrix [n_samples, dim].
        y_pred: Predicted matrix [n_samples, dim].

    Returns:
        R^2 score.
    """

    mse = float(np.mean((y_true - y_pred) ** 2))
    var = float(np.var(y_true))
    if var < 1e-12:
        return 0.0
    return float(1.0 - (mse / var))


def _evaluate_cross_reconstruction(
    usaes: dict[str, USAEModel], activations: dict[str, np.ndarray]
) -> np.ndarray:
    """Compute 3x3 cross-model reconstruction R^2 matrix.

    Args:
        usaes: Trained USAE branches.
        activations: model_name -> activations [n_total, hidden_dim].

    Returns:
        Matrix in MODEL_ORDER where (i, j)=R^2(encode i -> decode j).
    """

    device = torch.device(cfg.DEVICE)
    mat = np.zeros((len(MODEL_ORDER), len(MODEL_ORDER)), dtype=np.float64)

    with torch.no_grad():
        for i, source_name in enumerate(MODEL_ORDER):
            source = torch.from_numpy(
                activations[source_name].astype(np.float32, copy=False)
            ).to(device)
            _, codes_topk = usaes[source_name].encode_with_pretopk(source)

            for j, target_name in enumerate(MODEL_ORDER):
                pred = usaes[target_name].decode(codes_topk).detach().cpu().numpy()
                target = activations[target_name]
                mat[i, j] = _r2_score(target, pred)

    return mat


def _compute_sparsity_and_dead_concepts(
    usaes: dict[str, USAEModel], activations: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute TopK sparsity and dead concept statistics.

    Args:
        usaes: Trained USAE branches.
        activations: model_name -> activations [n_total, hidden_dim].

    Returns:
        Tuple of (sparsity_payload, dead_concepts_payload).
    """

    device = torch.device(cfg.DEVICE)
    per_model_sparsity: dict[str, float] = {}
    per_model_dead: dict[str, float] = {}
    total_zero = 0
    total_count = 0
    total_dead = 0.0

    with torch.no_grad():
        for model_name in MODEL_ORDER:
            x = torch.from_numpy(
                activations[model_name].astype(np.float32, copy=False)
            ).to(device)
            _, codes_topk = usaes[model_name].encode_with_pretopk(x)
            codes_np = codes_topk.detach().cpu().numpy()

            zero_count = int(np.sum(codes_np == 0.0))
            count = int(codes_np.size)
            sparsity = float(zero_count / max(count, 1))

            fire_mask = np.any(codes_np > 0.0, axis=0)
            dead_fraction = float(1.0 - (np.sum(fire_mask) / float(codes_np.shape[1])))

            per_model_sparsity[model_name] = sparsity
            per_model_dead[model_name] = dead_fraction
            total_zero += zero_count
            total_count += count
            total_dead += dead_fraction

    overall_sparsity = float(total_zero / max(total_count, 1))
    overall_dead = float(total_dead / len(MODEL_ORDER))

    return (
        {"overall": overall_sparsity, "per_model": per_model_sparsity},
        {"overall": overall_dead, "per_model": per_model_dead},
    )


def _save_checkpoints(usaes: dict[str, USAEModel]) -> dict[str, str]:
    """Persist USAE branches to checkpoint directory.

    Args:
        usaes: Trained USAE branches.

    Returns:
        model_name -> checkpoint path (string).
    """

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_map: dict[str, str] = {}
    for model_name in MODEL_ORDER:
        path = CHECKPOINT_DIR / f"{model_name}_usae.pt"
        torch.save(usaes[model_name].state_dict(), path)
        ckpt_map[model_name] = str(path)
    return ckpt_map


def _plot_heatmap(matrix: np.ndarray, save_path: Path) -> None:
    """Save cross-reconstruction heatmap figure.

    Args:
        matrix: R^2 matrix [3, 3] in MODEL_ORDER.
        save_path: Output path.
    """

    labels = [MODEL_TITLE[m] for m in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title("USAE Cross-Reconstruction R$^2$")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="white",
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Run RD5 USAE shared-latent cross-model experiment.

    Returns:
        Process exit code.
    """

    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("RD5 USAE Cross-Model: TabPFN vs TabICL vs iLTM", flush=True)
    print("=" * 80, flush=True)
    print(
        f"QUICK_RUN={cfg.QUICK_RUN}, seed={cfg.SEED}, device={cfg.DEVICE}, "
        f"n_train={cfg.N_TRAIN}, n_test={cfg.N_TEST}, epochs={cfg.sae_epochs}",
        flush=True,
    )

    activations, pairs = _collect_shared_activations()
    for model_name in MODEL_ORDER:
        print(
            f"[{model_name}] pooled activations: {activations[model_name].shape}",
            flush=True,
        )

    usaes = _train_usae(activations)
    matrix = _evaluate_cross_reconstruction(usaes, activations)
    sparsity, dead_concepts = _compute_sparsity_and_dead_concepts(usaes, activations)
    checkpoints = _save_checkpoints(usaes)

    heatmap_path = RESULTS_DIR / "cross_reconstruction_heatmap.png"
    _plot_heatmap(matrix, heatmap_path)

    payload: dict[str, Any] = {
        "config": {
            "quick_run": cfg.QUICK_RUN,
            "seed": cfg.SEED,
            "device": cfg.DEVICE,
            "n_train": cfg.N_TRAIN,
            "n_test": cfg.N_TEST,
            "pairs": pairs,
            "model_order": MODEL_ORDER,
            "core_layer": CORE_LAYER,
            "hidden_dim": HIDDEN_DIM,
            "n_concepts": N_CONCEPTS,
            "topk": TOPK,
            "aux_penalty": AUX_PENALTY,
            "sae_epochs": cfg.sae_epochs,
            "sae_batch_size": cfg.SAE_BATCH_SIZE,
            "sae_lr": SAE_LR,
        },
        "reconstruction_matrix": {
            "order": MODEL_ORDER,
            "r2": matrix.tolist(),
        },
        "sparsity": sparsity,
        "dead_concepts": dead_concepts,
        "artifacts": {
            "heatmap_path": str(heatmap_path),
            "checkpoints": checkpoints,
        },
    }

    json_path = RESULTS_DIR / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nCross-reconstruction R^2 matrix (rows=encode, cols=decode):", flush=True)
    for i, source_name in enumerate(MODEL_ORDER):
        row_vals = " ".join(f"{matrix[i, j]:.4f}" for j in range(len(MODEL_ORDER)))
        print(f"  {source_name}: {row_vals}", flush=True)

    print(
        f"\nSelf mean={np.mean(np.diag(matrix)):.4f}, cross mean={np.mean(matrix[np.where(~np.eye(3, dtype=bool))]):.4f}",
        flush=True,
    )
    print(f"Sparsity overall={sparsity['overall']:.4f}", flush=True)
    print(f"Dead concepts overall={dead_concepts['overall']:.4f}", flush=True)
    print(f"Saved: {json_path}", flush=True)
    print(f"Saved: {heatmap_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
