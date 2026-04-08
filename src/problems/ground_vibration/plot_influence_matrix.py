from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.figure import Figure

def _format_param_map(param_map: dict[str, float]) -> str:
    return ", ".join([f"{k}={v:.2E}" for k, v in param_map.items()])

plt.rc('font', family='serif', size=18)
plt.rc('text', usetex=True)
plt.rc('axes', labelsize=18)
plt.rc('legend', fontsize=12)
plt.rc('image', cmap=plt.get_cmap('RdBu').name)

def plot_influence_matrix(
        U_true: np.ndarray,
        U_pred: np.ndarray,
        param_map: dict[str, float],
        target_labels: list[str],
    ) -> Figure:
    """
    Plot operator-level comparison per channel.

    Args:
        U_true/U_pred: (N, N, C) complex influence channels.
    """
    if U_true.shape != U_pred.shape:
        raise ValueError(f"U_true and U_pred must have same shape. Got {U_true.shape} and {U_pred.shape}")
    if U_true.ndim != 3:
        raise ValueError(f"Expected (N, N, C) arrays. Got shape {U_true.shape}")

    n_channels = U_true.shape[2]
    if n_channels != len(target_labels):
        labels = [f"ch_{i}" for i in range(n_channels)]
    else:
        labels = target_labels

    fig, ax = plt.subplots(n_channels, 3, figsize=(12, max(3 * n_channels, 8)), sharex=True, sharey=True)
    if n_channels == 1:
        ax = np.asarray(ax)[None, :]  # type: ignore[assignment]

    param_str = _format_param_map(param_map=param_map)
    for ch in range(n_channels):
        true_abs = np.abs(U_true[..., ch])
        pred_abs = np.abs(U_pred[..., ch])
        rel_err = np.abs(U_pred[..., ch] - U_true[..., ch]) / (np.abs(U_true[..., ch]) + 1e-14)

        vmin = float(min(np.min(true_abs), np.min(pred_abs)))
        vmax = float(max(np.max(true_abs), np.max(pred_abs)))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        im0 = ax[ch, 0].imshow(true_abs, origin='lower', norm=norm, cmap='magma')
        ax[ch, 0].set_title(f"True |{labels[ch]}|")
        fig.colorbar(im0, ax=ax[ch, 0], fraction=0.046)

        im1 = ax[ch, 1].imshow(pred_abs, origin='lower', norm=norm, cmap='magma')
        ax[ch, 1].set_title(f"Pred |{labels[ch]}|")
        fig.colorbar(im1, ax=ax[ch, 1], fraction=0.046)

        im2 = ax[ch, 2].imshow(rel_err, origin='lower', cmap='viridis')
        ax[ch, 2].set_title(f"RelErr {labels[ch]}")
        fig.colorbar(im2, ax=ax[ch, 2], fraction=0.046)

        for col in range(3):
            ax[ch, col].set_xlabel('source index j')
            ax[ch, col].set_ylabel('field index i')

    fig.suptitle(f"Influence Operator Comparison ({param_str})", fontsize=14)
    fig.tight_layout()

    return fig
