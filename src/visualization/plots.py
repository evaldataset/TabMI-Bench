"""Visualization utilities for TabPFN mechanistic interpretability experiments.

Provides publication-ready plots for:
- Layer-wise R² curves (probing results)
- Probe complexity vs R² (linearity evidence)
- Attention weight heatmaps (Sample/Feature attention)
- Representation geometry (t-SNE/UMAP)
- Multi-condition R² comparison
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import publication style constants
from src.visualization.styles import (
    ERROR_ALPHA,
    FIGURE_SIZES,
    FONT_SIZES,
    LINEWIDTH,
    MARKERSIZE,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
    PUBLICATION_DPI,
    apply_publication_style,
    save_fig,
)

# Apply publication defaults
apply_publication_style()
sns.set_theme(style="whitegrid", palette="muted")

# Color palette for complexity levels
_COMPLEXITY_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]
_COMPLEXITY_STYLES = ["-", "--", "-.", ":"]


def _ensure_dir(path: Optional[str]) -> None:
    """Create parent directories for save_path if needed."""
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_layer_r2(
    r2_scores: np.ndarray,
    title: str = "R² across layers",
    save_path: Optional[str] = None,
    complexity_labels: Optional[list[str]] = None,
) -> plt.Figure:
    """Plot R² scores across transformer layers.

    Args:
        r2_scores: [n_layers, n_complexities] or [n_layers] array.
        title: Plot title.
        save_path: If provided, save figure to this path (PNG).
        complexity_labels: Labels for each complexity level.
            Defaults to ["Linear (0)", "MLP-1", "MLP-2", "MLP-3"].

    Returns:
        matplotlib Figure.
    """
    r2 = np.asarray(r2_scores)
    if r2.ndim == 1:
        r2 = r2[:, np.newaxis]

    n_layers, n_complexities = r2.shape
    layers = list(range(n_layers))

    if complexity_labels is None:
        default_labels = ["Linear (0)", "MLP-1", "MLP-2", "MLP-3"]
        complexity_labels = default_labels[:n_complexities]

    fig, ax = plt.subplots(figsize=(9, 5))

    for c_idx in range(n_complexities):
        color = _COMPLEXITY_COLORS[c_idx % len(_COMPLEXITY_COLORS)]
        style = _COMPLEXITY_STYLES[c_idx % len(_COMPLEXITY_STYLES)]
        label = (
            complexity_labels[c_idx]
            if c_idx < len(complexity_labels)
            else f"complexity={c_idx}"
        )
        ax.plot(
            layers,
            r2[:, c_idx],
            marker="o",
            markersize=5,
            color=color,
            linestyle=style,
            linewidth=2,
            label=label,
        )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold")
    ax.set_xticks(layers)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    _ensure_dir(save_path)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_probe_complexity(
    complexities: list[int],
    r2_scores: list[float],
    layer: int,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot R² vs probe complexity for a specific layer.

    A monotonically decreasing curve indicates linear encoding.

    Args:
        complexities: List of complexity values, e.g. [0, 1, 2, 3].
        r2_scores: Corresponding R² values.
        layer: Layer index (used in default title).
        title: Override title.
        save_path: If provided, save figure.

    Returns:
        matplotlib Figure.
    """
    if title is None:
        title = f"Probe Complexity vs R² (Layer {layer})"

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        complexities,
        r2_scores,
        marker="s",
        markersize=8,
        color="#2196F3",
        linewidth=2.5,
        label="R²",
    )
    ax.fill_between(complexities, r2_scores, alpha=0.15, color="#2196F3")

    ax.set_xlabel("Probe Complexity (hidden layers)", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold")
    ax.set_xticks(complexities)
    ax.set_xticklabels([f"complexity={c}" for c in complexities], rotation=15)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.4)

    # Annotate monotonic decrease
    if len(r2_scores) > 1 and all(
        r2_scores[i] >= r2_scores[i + 1] for i in range(len(r2_scores) - 1)
    ):
        ax.text(
            0.98,
            0.05,
            "↓ Monotonic decrease\n(linear encoding evidence)",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#4CAF50",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    fig.tight_layout()
    _ensure_dir(save_path)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_attention_heatmap(
    attn_matrix: np.ndarray,
    layer: int,
    head: int,
    attn_type: str = "feature",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    x_labels: Optional[list[str]] = None,
    y_labels: Optional[list[str]] = None,
) -> plt.Figure:
    """Plot attention weight matrix as heatmap.

    Args:
        attn_matrix: [seq_q, seq_k] attention weights (softmaxed, rows sum to 1).
        layer: Layer index.
        head: Head index.
        attn_type: "feature" or "sample".
        title: Override title.
        save_path: If provided, save figure.
        x_labels, y_labels: Axis tick labels.

    Returns:
        matplotlib Figure.
    """
    if title is None:
        title = f"{attn_type.capitalize()} Attention — Layer {layer}, Head {head}"

    seq_q, seq_k = attn_matrix.shape
    figsize = (max(6, seq_k * 0.4 + 2), max(5, seq_q * 0.4 + 2))

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        attn_matrix,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        annot=(seq_q <= 15 and seq_k <= 15),  # annotate only for small matrices
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"label": "Attention weight"},
        xticklabels=x_labels if x_labels else "auto",
        yticklabels=y_labels if y_labels else "auto",
    )

    ax.set_xlabel("Key (attended to)", fontsize=11)
    ax.set_ylabel("Query (attending from)", fontsize=11)
    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold")

    fig.tight_layout()
    _ensure_dir(save_path)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_representation_geometry(
    activations: np.ndarray,
    labels: np.ndarray,
    layer: int,
    method: str = "tsne",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    n_components: int = 2,
) -> plt.Figure:
    """Visualize representation space geometry using t-SNE or UMAP.

    Args:
        activations: [n_samples, n_features] hidden states.
        labels: [n_samples] color labels (continuous or categorical).
        layer: Layer index (for title).
        method: "tsne" or "umap".
        title: Override title.
        save_path: If provided, save figure.
        n_components: 2 for 2D visualization.

    Returns:
        matplotlib Figure.
    """
    if title is None:
        title = f"Representation Geometry — Layer {layer} ({method.upper()})"

    # Dimensionality reduction
    if method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=n_components,
            random_state=42,
            perplexity=min(30, max(5, len(activations) // 4)),
            max_iter=500,
        )
        embedding = reducer.fit_transform(activations)
    elif method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=n_components, random_state=42)
            embedding = reducer.fit_transform(activations)
        except ImportError:
            # Fallback to t-SNE if umap not available
            from sklearn.manifold import TSNE

            reducer = TSNE(n_components=n_components, random_state=42, max_iter=500)
            embedding = reducer.fit_transform(activations)
            method = "tsne (umap unavailable)"
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'.")

    fig, ax = plt.subplots(figsize=(7, 6))

    is_continuous = np.issubdtype(labels.dtype, np.floating)
    if is_continuous:
        sc = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap="viridis",
            alpha=0.7,
            s=30,
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label="Target value")
    else:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for lbl, color in zip(unique_labels, colors):
            mask = labels == lbl
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[color],
                label=str(lbl),
                alpha=0.7,
                s=30,
                edgecolors="none",
            )
        ax.legend(loc="best", fontsize=9, markerscale=1.5)

    ax.set_xlabel(f"{method.upper()} dim 1", fontsize=11)
    ax.set_ylabel(f"{method.upper()} dim 2", fontsize=11)
    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _ensure_dir(save_path)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_layer_comparison(
    r2_dict: dict[str, np.ndarray],
    title: str = "R² comparison across conditions",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare R² curves across multiple conditions on one plot.

    Args:
        r2_dict: {label: r2_array_of_shape_[n_layers]} mapping.
        title: Plot title.
        save_path: If provided, save figure.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(r2_dict)))
    for (label, r2_arr), color in zip(r2_dict.items(), colors):
        r2_arr = np.asarray(r2_arr)
        layers = list(range(len(r2_arr)))
        ax.plot(
            layers,
            r2_arr,
            marker="o",
            markersize=5,
            color=color,
            linewidth=2,
            label=label,
        )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold")
    n_layers = len(next(iter(r2_dict.values())))
    ax.set_xticks(range(n_layers))
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    _ensure_dir(save_path)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig

# ---------------------------------------------------------------------------
# Publication-quality multi-model plots (Phase 7)
# ---------------------------------------------------------------------------


def plot_multi_model_r2(
    model_data: dict[str, dict[str, list[float] | np.ndarray]],
    title: str = "R² across layers",
    ylabel: str = "R²",
    save_path: str | None = None,
    figsize: tuple[float, float] | None = None,
    ylim: tuple[float, float] = (-0.05, 1.05),
) -> plt.Figure:
    """Plot R² layer profiles for multiple models with error bars.

    Args:
        model_data: {model_key: {"mean": [...], "std": [...]}} mapping.
            model_key must be a key in MODEL_COLORS.
        title: Plot title.
        ylabel: Y-axis label.
        save_path: Base path WITHOUT extension (saves PNG + PDF).
        figsize: Override figure size.
        ylim: Y-axis limits.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize or FIGURE_SIZES["full"])

    for model_key, data in model_data.items():
        mean = np.asarray(data["mean"])
        std = np.asarray(data.get("std", np.zeros_like(mean)))
        layers = np.arange(len(mean))
        color = MODEL_COLORS.get(model_key, "#333333")
        label = MODEL_LABELS.get(model_key, model_key)
        marker = MODEL_MARKERS.get(model_key, "o")

        ax.plot(
            layers, mean,
            marker=marker, markersize=MARKERSIZE,
            color=color, linewidth=LINEWIDTH,
            label=label, zorder=3,
        )
        ax.fill_between(
            layers, mean - std, mean + std,
            alpha=ERROR_ALPHA, color=color, zorder=2,
        )

    ax.set_xlabel("Layer", fontsize=FONT_SIZES["label"])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES["label"])
    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold")
    ax.set_ylim(*ylim)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend(loc="best", fontsize=FONT_SIZES["legend"], framealpha=0.9)

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path, close=False)

    return fig


def plot_cka_heatmaps(
    cka_matrices: dict[str, np.ndarray],
    title: str = "CKA Similarity",
    save_path: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot CKA heatmaps for multiple models side-by-side.

    Args:
        cka_matrices: {model_key: cka_matrix_2d} mapping.
        title: Super-title.
        save_path: Base path WITHOUT extension.
        figsize: Override figure size.

    Returns:
        matplotlib Figure.
    """
    n_models = len(cka_matrices)
    if figsize is None:
        figsize = (3.0 * n_models + 0.5, 3.0)

    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    im = None  # will be set in loop
    for ax, (model_key, matrix) in zip(axes, cka_matrices.items()):
        n = matrix.shape[0]
        label = MODEL_LABELS.get(model_key, model_key)
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="equal")
        ax.set_title(label, fontsize=FONT_SIZES["label"], fontweight="bold")
        ax.set_xlabel("Layer", fontsize=FONT_SIZES["tick"])
        ax.set_ylabel("Layer", fontsize=FONT_SIZES["tick"])
        ax.set_xticks(range(0, n, max(1, n // 6)))
        ax.set_yticks(range(0, n, max(1, n // 6)))

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("CKA", fontsize=FONT_SIZES["tick"])

    fig.suptitle(title, fontsize=FONT_SIZES["title"], fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path, close=False)

    return fig


def plot_sensitivity_profiles(
    model_data: dict[str, dict[str, list[float] | np.ndarray]],
    title: str = "Causal Tracing Sensitivity",
    save_path: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot noising-based causal tracing sensitivity profiles.

    Args:
        model_data: {model_key: {"mean": [...], "std": [...]}} mapping.
        title: Plot title.
        save_path: Base path WITHOUT extension.
        figsize: Override figure size.

    Returns:
        matplotlib Figure.
    """
    return plot_multi_model_r2(
        model_data,
        title=title,
        ylabel="Sensitivity",
        save_path=save_path,
        figsize=figsize,
        ylim=(-0.05, 1.1),
    )


def plot_steering_scatter(
    layers: list[int],
    model_data: dict[str, dict[str, list[float] | np.ndarray]],
    title: str = "Steering Effectiveness",
    save_path: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot steering |r| across layers for multiple models.

    Args:
        layers: List of layer indices probed.
        model_data: {model_key: {"mean": [...], "std": [...]}} mapping.
        title: Plot title.
        save_path: Base path WITHOUT extension.
        figsize: Override figure size.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize or FIGURE_SIZES["full"])

    for model_key, data in model_data.items():
        mean = np.asarray(data["mean"])
        std = np.asarray(data.get("std", np.zeros_like(mean)))
        color = MODEL_COLORS.get(model_key, "#333333")
        label = MODEL_LABELS.get(model_key, model_key)
        marker = MODEL_MARKERS.get(model_key, "o")

        ax.errorbar(
            layers, mean, yerr=std,
            marker=marker, markersize=MARKERSIZE + 1,
            color=color, linewidth=LINEWIDTH,
            label=label, capsize=3, capthick=1.0,
            zorder=3,
        )

    ax.set_xlabel("Layer", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("|r| (Pearson)", fontsize=FONT_SIZES["label"])
    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold")
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.legend(loc="best", fontsize=FONT_SIZES["legend"], framealpha=0.9)

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path, close=False)

    return fig


def plot_sae_grouped_bar(
    data: dict[str, dict[str, dict[str, float]]],
    metric: str = "max_alpha_corr",
    title: str = "SAE Feature Correlation",
    ylabel: str = "Max |r_α|",
    save_path: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot grouped bar chart for SAE comparison.

    Args:
        data: {variant: {model: {"mean": float, "std": float}}}.
        metric: Metric name (for labeling).
        title: Plot title.
        ylabel: Y-axis label.
        save_path: Base path WITHOUT extension.
        figsize: Override figure size.

    Returns:
        matplotlib Figure.
    """
    from src.visualization.styles import SAE_COLORS, SAE_HATCHES, SAE_LABELS

    variants = list(data.keys())
    models = list(next(iter(data.values())).keys())
    n_variants = len(variants)
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.8 / n_variants

    fig, ax = plt.subplots(figsize=figsize or FIGURE_SIZES["full"])

    for i, variant in enumerate(variants):
        means = [data[variant][m]["mean"] for m in models]
        stds = [data[variant][m]["std"] for m in models]
        offset = (i - (n_variants - 1) / 2) * width
        color = SAE_COLORS.get(variant, f"C{i}")
        hatch = SAE_HATCHES.get(variant, "")
        label = SAE_LABELS.get(variant, variant)

        ax.bar(
            x + offset, means, width * 0.9,
            yerr=stds, label=label,
            color=color, alpha=0.8, hatch=hatch,
            capsize=3, edgecolor="white", linewidth=0.5,
        )

    ax.set_xlabel("Model", fontsize=FONT_SIZES["label"])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES["label"])
    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.legend(fontsize=FONT_SIZES["legend"], framealpha=0.9)

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path, close=False)

    return fig

# ---------------------------------------------------------------------------
# Verification script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    os.makedirs("results/test_plots", exist_ok=True)

    # 1. plot_layer_r2
    r2 = np.random.rand(12, 4) * 0.8 + 0.1
    fig = plot_layer_r2(
        r2, title="Test R² plot", save_path="results/test_plots/layer_r2.png"
    )
    assert os.path.exists("results/test_plots/layer_r2.png")
    plt.close(fig)
    print("plot_layer_r2: OK")

    # 2. plot_probe_complexity
    fig = plot_probe_complexity(
        [0, 1, 2, 3],
        [0.9, 0.85, 0.8, 0.75],
        layer=6,
        save_path="results/test_plots/probe_complexity.png",
    )
    assert os.path.exists("results/test_plots/probe_complexity.png")
    plt.close(fig)
    print("plot_probe_complexity: OK")

    # 3. plot_attention_heatmap
    attn = np.random.rand(8, 8)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    fig = plot_attention_heatmap(
        attn,
        layer=5,
        head=0,
        attn_type="sample",
        save_path="results/test_plots/attn_heatmap.png",
    )
    assert os.path.exists("results/test_plots/attn_heatmap.png")
    plt.close(fig)
    print("plot_attention_heatmap: OK")

    # 4. plot_representation_geometry (t-SNE, small data for speed)
    acts = np.random.randn(40, 20)
    lbls = np.random.randn(40)
    fig = plot_representation_geometry(
        acts,
        lbls,
        layer=5,
        method="tsne",
        save_path="results/test_plots/repr_geometry.png",
    )
    assert os.path.exists("results/test_plots/repr_geometry.png")
    plt.close(fig)
    print("plot_representation_geometry (tsne): OK")

    # 5. plot_layer_comparison
    r2_dict = {"alpha": np.random.rand(12), "beta": np.random.rand(12)}
    fig = plot_layer_comparison(
        r2_dict, save_path="results/test_plots/layer_comparison.png"
    )
    assert os.path.exists("results/test_plots/layer_comparison.png")
    plt.close(fig)
    print("plot_layer_comparison: OK")

    print("ALL CHECKS PASSED")
