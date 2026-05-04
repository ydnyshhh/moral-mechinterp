"""Reusable figure style helpers for static research plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from moral_mechinterp.io import ensure_dir


def apply_paper_style(*, font_family: str = "serif") -> None:
    """Apply a custom empirical-science plotting style.

    ``font_family="serif"`` gives a more publication-like STIX/Times look.
    ``font_family="sans"`` keeps a DejaVu Sans style for slides or reports.
    """

    try:
        import scienceplots  # noqa: F401
    except ImportError:
        pass

    if font_family == "serif":
        font_config = {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "Times", "STIXGeneral"],
            "mathtext.fontset": "stix",
        }
    elif font_family == "sans":
        font_config = {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
        }
    else:
        raise ValueError("font_family must be either 'serif' or 'sans'")

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": "#2A2A2A",
            "axes.linewidth": 0.85,
            "xtick.color": "#2A2A2A",
            "ytick.color": "#2A2A2A",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
            **font_config,
        }
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = 0.03,
    y: float = 0.94,
    ha: str = "left",
    va: str = "top",
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=10,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2},
    )


def _trim_svg_trailing_whitespace(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    trimmed = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    path.write_text(trimmed, encoding="utf-8")


def save_figure(fig: plt.Figure, output_base: str | Path) -> list[Path]:
    base = Path(output_base)
    ensure_dir(base.parent)
    if base.suffix:
        base = base.with_suffix("")

    paths: list[Path] = []
    for suffix in (".png", ".pdf", ".svg"):
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight")
        if suffix == ".svg":
            _trim_svg_trailing_whitespace(path)
        paths.append(path)
    plt.close(fig)
    return paths
