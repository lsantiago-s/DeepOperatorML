from __future__ import annotations

import logging
import shutil

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_HAS_LATEX = shutil.which("latex") is not None
_WARNED_MISSING_LATEX = False


def configure_matplotlib_text(
    *,
    font_family: str = "serif",
    font_size: float,
    axes_labelsize: float,
    legend_fontsize: float,
    cmap_name: str | None = None,
) -> None:
    global _WARNED_MISSING_LATEX

    if not _HAS_LATEX and not _WARNED_MISSING_LATEX:
        logger.warning(
            "LaTeX executable not found; falling back to Matplotlib mathtext for plots."
        )
        _WARNED_MISSING_LATEX = True

    plt.rc("font", family=font_family, size=font_size)
    plt.rc("text", usetex=_HAS_LATEX)
    plt.rc("axes", labelsize=axes_labelsize)
    plt.rc("legend", fontsize=legend_fontsize)
    if cmap_name is not None:
        plt.rc("image", cmap=cmap_name)
