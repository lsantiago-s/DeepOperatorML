import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from src.problems.plotting_compat import configure_matplotlib_text

configure_matplotlib_text(
    font_size=15,
    axes_labelsize=15,
    legend_fontsize=12,
    cmap_name='Spectral',
)
# matplotlib.rcParams['text.latex.preamble'] = r'\math'


def plot_basis(basis: np.ndarray,
               index: int,
               target_labels: list[str]) -> Figure:

    n1, n2 = basis.shape
    n_channels = 2
    # ensure we have a safe label mapping even if target_labels is None or shorter
    label_mapping = target_labels if target_labels and len(target_labels) >= n_channels \
        else [str(i) for i in range(n_channels)]

    ncols = n_channels

    labels = {0: r'$\Re(\cdot)$', 1: r'$\Im(\cdot)$'}

    fig, axs = plt.subplots(1, ncols, figsize=(ncols * 4, 4), squeeze=False)

    # human-friendly suffix/title (define once)
    suffix = 'st' if index == 1 else 'nd' if index == 2 else 'rd' if index == 3 else 'th'
    sup_title = f"{index}{suffix} vector"

    field_ch = basis
    contour = axs[0, 0].contourf(field_ch.real)
    # contour = axs[0, 0, ch].imshow(np.flipud(field_ch.T))
    axs[0, 0].invert_yaxis()
    axs[0, 0].set_xlabel('i', fontsize=12)
    axs[0, 0].set_ylabel('j', fontsize=12)
    if n_channels > 1:
        axs[0, 0].set_title(f"{index}{suffix} vector for {labels[0]}")

    fig.colorbar(contour, ax=axs[0, 0])

    contour = axs[0, 1].contourf(field_ch.imag)
    # contour = axs[0, 1].imshow(np.flipud(field_ch.T))
    axs[0, 1].invert_yaxis()
    axs[0, 1].set_xlabel('i', fontsize=12)
    axs[0, 1].set_ylabel('j', fontsize=12)
    if n_channels > 1:
        axs[0, 1].set_title(f"{index}{suffix} vector for {labels[1]}")

    fig.colorbar(contour, ax=axs[0, 1])

    if n_channels == 1:
        fig.suptitle(sup_title)

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    return fig
