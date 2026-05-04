"""Publication-quality style constants for TFMI paper figures.

Defines shared constants for DPI, fonts, colors, figure sizes, and
matplotlib rcParams to ensure visual consistency across all figures.

Target venues: NeurIPS 2026, ICML, TMLR.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# DPI & Export
# ---------------------------------------------------------------------------
PUBLICATION_DPI: int = 300
SCREEN_DPI: int = 150  # for quick preview

# ---------------------------------------------------------------------------
# Font Sizes (following NeurIPS / ICML conventions)
# ---------------------------------------------------------------------------
FONT_SIZES: dict[str, int] = {
    "title": 10,
    "label": 12,
    "tick": 10,
    "legend": 9,
    "annotation": 8,
}

# ---------------------------------------------------------------------------
# Figure Sizes (inches) — NeurIPS column: 5.5 in, full: 6.75 in text width
# ---------------------------------------------------------------------------
FIGURE_SIZES: dict[str, tuple[float, float]] = {
    "single": (3.25, 2.5),  # single column
    "half": (5.5, 3.0),  # half page width
    "full": (6.75, 3.0),  # full text width
    "full_tall": (6.75, 4.5),  # full width, taller
    "wide": (7.0, 3.5),  # slightly wider (appendix)
}

# ---------------------------------------------------------------------------
# Model Colors — 4-model palette (colorblind-friendly)
# ---------------------------------------------------------------------------
MODEL_COLORS: dict[str, str] = {
    "tabpfn": "#1f77b4",  # blue
    "tabpfn25": "#9467bd",  # purple
    "tabicl": "#ff7f0e",  # orange
    "iltm": "#2ca02c",  # green
}

MODEL_LABELS: dict[str, str] = {
    "tabpfn": "TabPFN v2",
    "tabpfn25": "TabPFN v2.5",
    "tabicl": "TabICL v2",
    "iltm": "iLTM",
}

MODEL_MARKERS: dict[str, str] = {
    "tabpfn": "o",
    "tabpfn25": "D",
    "tabicl": "s",
    "iltm": "^",
}

# ---------------------------------------------------------------------------
# Line & Marker Defaults
# ---------------------------------------------------------------------------
LINEWIDTH: float = 1.5
MARKERSIZE: int = 5
ERROR_ALPHA: float = 0.2
ERROR_CAPSIZE: int = 2

# ---------------------------------------------------------------------------
# CKA Heatmap Colormap
# ---------------------------------------------------------------------------
CKA_CMAP: str = "viridis"
ATTENTION_CMAP: str = "Blues"

# ---------------------------------------------------------------------------
# SAE Variant Styling
# ---------------------------------------------------------------------------
SAE_COLORS: dict[str, str] = {
    "relu_16x": "#1f77b4",
    "jumprelu_16x": "#ff7f0e",
    "topk_16x_6p25": "#2ca02c",
}

SAE_LABELS: dict[str, str] = {
    "relu_16x": "ReLU (16×)",
    "jumprelu_16x": "JumpReLU (16×)",
    "topk_16x_6p25": "TopK (16×)",
}

SAE_HATCHES: dict[str, str] = {
    "relu_16x": "",
    "jumprelu_16x": "//",
    "topk_16x_6p25": "xx",
}

# ---------------------------------------------------------------------------
# Export Formats
# ---------------------------------------------------------------------------
EXPORT_FORMATS: list[str] = ["png", "pdf"]


# ---------------------------------------------------------------------------
# Apply publication rcParams
# ---------------------------------------------------------------------------
def apply_publication_style() -> None:
    """Set matplotlib rcParams for publication-quality figures.

    Call once at the top of any figure-generation script.
    """
    matplotlib.use("Agg")
    rc: dict[str, Any] = {
        # Figure
        "figure.dpi": PUBLICATION_DPI,
        "savefig.dpi": PUBLICATION_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Font
        "font.size": FONT_SIZES["tick"],
        "axes.titlesize": FONT_SIZES["title"],
        "axes.labelsize": FONT_SIZES["label"],
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "legend.fontsize": FONT_SIZES["legend"],
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        # Lines
        "lines.linewidth": LINEWIDTH,
        "lines.markersize": MARKERSIZE,
        # Axes
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        # Ticks
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # Legend
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        # PDF
        "pdf.fonttype": 42,  # TrueType (editable text in PDF)
        "ps.fonttype": 42,
    }
    plt.rcParams.update(rc)


def save_fig(
    fig: plt.Figure,
    path: str,
    *,
    dpi: int = PUBLICATION_DPI,
    formats: list[str] | None = None,
    close: bool = True,
) -> list[str]:
    """Save figure in multiple formats (PNG + PDF by default).

    Args:
        fig: Matplotlib Figure object.
        path: Base path WITHOUT extension (e.g., 'paper/figures/fig1').
        dpi: Resolution for raster formats.
        formats: List of extensions. Defaults to EXPORT_FORMATS.
        close: Whether to close the figure after saving.

    Returns:
        List of saved file paths.
    """
    from pathlib import Path

    if formats is None:
        formats = EXPORT_FORMATS

    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for fmt in formats:
        out = str(base.with_suffix(f".{fmt}"))
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        saved.append(out)

    if close:
        plt.close(fig)

    return saved
