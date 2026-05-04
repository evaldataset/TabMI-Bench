# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUntypedBaseClass=false, reportUnknownParameterType=false, reportUnknownArgumentType=false
"""Sparse Autoencoder (SAE) for TabPFN feature decomposition.

Trains a sparse autoencoder on TabPFN's hidden state activations to
decompose polysemantic neurons into monosemantic features. This enables
richer interpretability than linear probing alone.

Architecture:
    encoder: Linear(input_dim → hidden_dim) + ReLU
    decoder: Linear(hidden_dim → input_dim)
    Loss = MSE(h, h_reconstructed) + l1_coeff * L1(encoded)

Reference:
    - Bricken et al. "Towards Monosemanticity" (Anthropic, 2023)
    - Chanin et al. "A is for Absorption" (2024) arXiv:2409.14507
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SmoothJumpReLU(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        init_theta: float = 0.01,
        bandwidth: float = 100.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive, got {bandwidth}")

        self.bandwidth = bandwidth
        self.theta = nn.Parameter(torch.full((hidden_dim,), init_theta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.bandwidth * (x - self.theta))
        return F.relu(x) * gate


class TopKActivation(nn.Module):
    def __init__(self, hidden_dim: int, k: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if k > hidden_dim:
            raise ValueError(
                f"k must be <= hidden_dim, got k={k}, hidden_dim={hidden_dim}"
            )
        self.hidden_dim = hidden_dim
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pos = F.relu(x)
        if self.k >= x_pos.shape[-1]:
            return x_pos

        values, indices = torch.topk(x_pos, k=self.k, dim=-1)
        out = torch.zeros_like(x_pos)
        out.scatter_(-1, indices, values)
        return out


class TabPFNSparseAutoencoder(nn.Module):
    """Sparse Autoencoder for TabPFN hidden state decomposition.

    Args:
        input_dim: Dimension of input activations (default: 192 for TabPFN)
        expansion_factor: Hidden dim = input_dim * expansion_factor (default: 4)

    Usage:
        sae = TabPFNSparseAutoencoder(input_dim=192, expansion_factor=4)
        encoded = sae.encode(h)       # [batch, 768] sparse features
        decoded = sae.decode(encoded) # [batch, 192] reconstruction
        decoded, encoded = sae(h)     # forward pass returns both
    """

    def __init__(
        self,
        input_dim: int = 192,
        expansion_factor: int = 4,
        activation: str = "relu",
        jumprelu_bandwidth: float = 100.0,
        topk_k: int | None = None,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if expansion_factor <= 0:
            raise ValueError(
                f"expansion_factor must be positive, got {expansion_factor}"
            )
        if activation not in {"relu", "jumprelu", "topk"}:
            raise ValueError(
                f"activation must be 'relu', 'jumprelu', or 'topk', got {activation}"
            )

        hidden_dim = input_dim * expansion_factor
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.expansion_factor = expansion_factor
        self.activation_name = activation

        self.encoder = nn.Linear(input_dim, hidden_dim)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "jumprelu":
            self.activation = SmoothJumpReLU(
                hidden_dim=hidden_dim,
                init_theta=0.01,
                bandwidth=jumprelu_bandwidth,
            )
        else:
            k = topk_k if topk_k is not None else max(1, hidden_dim // 16)
            self.activation = TopKActivation(hidden_dim=hidden_dim, k=k)

        # Decoder: Linear (no activation)
        self.decoder = nn.Linear(hidden_dim, input_dim)

        # Initialize with Xavier
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse features. Returns [batch, hidden_dim]."""
        if h.ndim != 2:
            raise ValueError(
                f"Expected 2D tensor [batch, input_dim], got shape {h.shape}"
            )
        if h.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {h.shape[-1]}")
        return self.activation(self.encoder(h))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activation space. Returns [batch, input_dim]."""
        if features.ndim != 2:
            raise ValueError(
                f"Expected 2D tensor [batch, hidden_dim], got shape {features.shape}"
            )
        if features.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden dim {self.hidden_dim}, got {features.shape[-1]}"
            )
        return self.decoder(features)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (reconstructed, encoded)."""
        encoded = self.encode(h)
        reconstructed = self.decode(encoded)
        return reconstructed, encoded


class SAETrainer:
    """Training pipeline for TabPFN SAE.

    Usage:
        trainer = SAETrainer(sae, lr=1e-3, l1_coeff=1e-3)

        # Collect activations from model
        activations = trainer.collect_activations(
            model, datasets, layer=6, token_idx=-1
        )

        # Train SAE
        history = trainer.train(activations, epochs=100, batch_size=64)
    """

    def __init__(
        self,
        sae: TabPFNSparseAutoencoder,
        lr: float = 1e-3,
        l1_coeff: float = 1e-3,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if l1_coeff < 0:
            raise ValueError(f"l1_coeff must be non-negative, got {l1_coeff}")
        self.sae = sae
        self.l1_coeff = l1_coeff
        self.optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    @property
    def device(self) -> torch.device:
        """Current model device."""
        return next(self.sae.parameters()).device

    def collect_activations(
        self,
        model: Any,
        datasets: list[dict[str, Any]],
        layer: int = 6,
        token_idx: int = -1,
    ) -> np.ndarray:
        """Collect label-token activations from multiple datasets.

        For each dataset:
            1. model.fit(X_train, y_train)
            2. hooker.forward_with_cache(X_test)
            3. Extract act[0, :single_eval_pos, token_idx, :] (train tokens)

        Concatenate all → [total_samples, 192]

        Args:
            model: TabPFNRegressor instance
            datasets: List of dicts with X_train, y_train, X_test keys
            layer: Which layer to extract from (default 6)
            token_idx: Which token (-1 = label token)

        Returns:
            np.ndarray of shape [total_samples, input_dim]
        """
        from src.hooks.tabpfn_hooker import TabPFNHookedModel

        if not datasets:
            raise ValueError("datasets must contain at least one dataset")

        all_activations: list[np.ndarray] = []

        for ds_idx, ds in enumerate(datasets):
            self._validate_dataset(ds, ds_idx)

            model.fit(ds["X_train"], ds["y_train"])
            hooker = TabPFNHookedModel(model)
            _preds, cache = hooker.forward_with_cache(ds["X_test"])

            if "layers" not in cache or "single_eval_pos" not in cache:
                raise KeyError("cache must contain 'layers' and 'single_eval_pos' keys")

            single_eval_pos = int(cache["single_eval_pos"])
            layer_acts = cache["layers"]

            if not 0 <= layer < len(layer_acts):
                raise ValueError(
                    f"Requested layer {layer}, but cache has {len(layer_acts)} layers"
                )

            act = layer_acts[layer]
            train_act = act[0, :single_eval_pos, token_idx, :].detach().cpu().numpy()
            if train_act.shape[-1] != self.sae.input_dim:
                raise ValueError(
                    f"Activation dim mismatch: expected {self.sae.input_dim}, got {train_act.shape[-1]}"
                )
            all_activations.append(train_act.astype(np.float32, copy=False))

        return np.concatenate(all_activations, axis=0)

    def train(
        self,
        activations: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train the SAE on collected activations.

        Returns training history:
            {
                'total_loss': [...],
                'mse_loss': [...],
                'l1_loss': [...],
                'sparsity': [...],  # fraction of zero features per sample
                'reconstruction_r2': [...],  # R² of reconstruction
            }
        """
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        act_tensor = self._to_tensor(activations)
        n_samples = act_tensor.shape[0]
        if n_samples == 0:
            raise ValueError("activations cannot be empty")

        self.sae.train()
        history: dict[str, list[float]] = {
            "total_loss": [],
            "mse_loss": [],
            "l1_loss": [],
            "sparsity": [],
            "reconstruction_r2": [],
        }

        for epoch in range(epochs):
            perm = torch.randperm(n_samples, device=act_tensor.device)
            epoch_mse = 0.0
            epoch_l1 = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                batch_idx = perm[start : start + batch_size]
                batch = act_tensor[batch_idx]

                reconstructed, encoded = self.sae(batch)
                mse_loss = F.mse_loss(reconstructed, batch)
                l1_loss = encoded.abs().mean()
                total_loss = mse_loss + self.l1_coeff * l1_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_mse += float(mse_loss.item())
                epoch_l1 += float(l1_loss.item())
                n_batches += 1

            avg_mse = epoch_mse / max(n_batches, 1)
            avg_l1 = epoch_l1 / max(n_batches, 1)

            self.sae.eval()
            with torch.no_grad():
                rec_full, enc_full = self.sae(act_tensor)
                sparsity = float((enc_full == 0).float().mean().item())
                r2 = self._compute_reconstruction_r2(act_tensor, rec_full)
            self.sae.train()

            history["total_loss"].append(avg_mse + self.l1_coeff * avg_l1)
            history["mse_loss"].append(avg_mse)
            history["l1_loss"].append(avg_l1)
            history["sparsity"].append(sparsity)
            history["reconstruction_r2"].append(r2)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(
                    f"  Epoch {epoch:3d}/{epochs}: MSE={avg_mse:.6f}, L1={avg_l1:.4f}, R²={r2:.4f}, sparsity={sparsity:.3f}"
                )

        return history

    def _to_tensor(self, activations: np.ndarray) -> torch.Tensor:
        """Convert activation array to float tensor on current model device."""
        array = np.asarray(activations, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(
                f"Expected 2D activations [n_samples, input_dim], got shape {array.shape}"
            )
        if array.shape[1] != self.sae.input_dim:
            raise ValueError(
                f"Expected activations with dim {self.sae.input_dim}, got {array.shape[1]}"
            )
        return torch.from_numpy(array).to(self.device)

    @staticmethod
    def _compute_reconstruction_r2(
        inputs: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> float:
        """Compute reconstruction R² = 1 - SS_res / SS_tot."""
        ss_res = ((inputs - reconstructed) ** 2).sum().item()
        ss_tot = ((inputs - inputs.mean(dim=0)) ** 2).sum().item()
        return float(1.0 - ss_res / max(ss_tot, 1e-10))

    @staticmethod
    def _validate_dataset(ds: dict[str, Any], ds_idx: int) -> None:
        """Validate dataset item for activation collection."""
        required_keys = ("X_train", "y_train", "X_test")
        missing = [key for key in required_keys if key not in ds]
        if missing:
            raise KeyError(f"Dataset index {ds_idx} missing keys: {', '.join(missing)}")


def generate_diverse_datasets(
    n_datasets: int = 50,
    n_train: int = 50,
    n_test: int = 20,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate diverse linear regression datasets for activation collection.

    Varies α ∈ [0.5, 5.0], β ∈ [0.5, 5.0] uniformly across datasets.
    Each dataset: z = α*x + β*y

    Returns list of dicts with X_train, y_train, X_test, y_test, alpha, beta.
    """
    if n_datasets <= 0:
        raise ValueError(f"n_datasets must be positive, got {n_datasets}")
    if n_train <= 0:
        raise ValueError(f"n_train must be positive, got {n_train}")
    if n_test <= 0:
        raise ValueError(f"n_test must be positive, got {n_test}")

    rng = np.random.default_rng(random_seed)
    datasets: list[dict[str, Any]] = []

    for _ in range(n_datasets):
        alpha = float(rng.uniform(0.5, 5.0))
        beta = float(rng.uniform(0.5, 5.0))

        n_total = n_train + n_test
        X = rng.standard_normal((n_total, 2), dtype=np.float32)
        y = alpha * X[:, 0] + beta * X[:, 1]

        datasets.append(
            {
                "X_train": X[:n_train],
                "y_train": y[:n_train],
                "X_test": X[n_train:],
                "y_test": y[n_train:],
                "alpha": alpha,
                "beta": beta,
            }
        )

    return datasets
