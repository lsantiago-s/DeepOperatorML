from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import PercentFormatter
from matplotlib.figure import Figure

def format_param(param: list, param_keys: list[str]):
    if not isinstance(param, float) and len(param_keys) == len(param):
        items = [f"{k}={v:.1E}" for k, v in zip(param_keys, param)]
        return "(" + ", ".join(items) + ")"
    else:
        return f"{param:.0E}"

plt.rc('font', family='serif', size=18)
plt.rc('text', usetex=True)
plt.rc('axes', labelsize=18)
plt.rc('legend', fontsize=12)
cmap = plt.get_cmap('RdBu') # # tried: 'RdBu'
plt.rc('image', cmap=cmap.name)
# matplotlib.rcParams['text.latex.preamble'] = r'\math'

def plot_influence_matrix(
        U_true: np.ndarray,
        U_pred: np.ndarray,
        input_function_labels: list[str],
        input_function_value: list[float],
        target_labels: list[str]
    ) -> Figure:

    param_str = format_param(
        param=input_function_value,
        param_keys=input_function_labels
    )
    labels = target_labels

    fig, ax = plt.subplots(4, 2, figsize=(
        6, 12), sharex=True, sharey=True)
    
    for index, _ in enumerate(ax):
        norm_real = colors.Normalize(vmin=min(np.min(U_pred[index].real), np.min(U_true[index].real)),
                                    vmax=max(np.max(U_pred[index].real), np.max(U_true[index].real)))
        norm_imag = colors.Normalize(vmin=min(np.min(U_pred[index].imag), np.min(U_true[index].imag)),
                                    vmax=max(np.max(U_pred[index].imag), np.max(U_true[index].imag)))
        c0 = ax[index, 0].contourf(
            (U_true[index].real - U_pred[index].real)/(U_true[index].real + 1e-8), norm=norm_real)
        ax[index, 0].set_title(
            f"Predicted {labels[0]} {input_function_labels[0]}={param_str}")
        ax[index, 0].set_xlabel('i')
        ax[index, 0].set_ylabel('j')

        c1 = ax[index, 1].contourf(
            (U_true[index].imag - U_pred[index].imag)/(U_true[index].imag + 1e-8), norm=norm_imag
        )
        ax[index, 1].set_title(
            f"True {labels[0]} {input_function_labels[0]}={param_str}")
        ax[index, 1].set_xlabel('i')
        ax[index, 1].set_ylabel('j')

        c_bar_0 = fig.colorbar(c0, ax=ax[index, 0])
        c_bar_1 = fig.colorbar(c1, ax=ax[index, 1])
        # c_bar_2.ax.yaxis.set_major_formatter(PercentFormatter(1))
        # c_bar_2.set_label('%')

    # ax[0, 0].invert_yaxis()
    fig.tight_layout()

    return fig
