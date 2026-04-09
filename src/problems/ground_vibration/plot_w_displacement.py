import numpy as np
import matplotlib.pyplot as plt
from typing import Any

def plot_w_displacement(
        w_integration: np.ndarray[tuple[int], Any],
        w_model: np.ndarray[tuple[int], Any],
        coordinates: np.ndarray[tuple[int], Any],
        **kwargs
    ) -> plt.Figure:
    fig, ax = plt.subplots()

    p0 = kwargs.get('p0', None)
    ax.plot(coordinates, w_integration[1::2].real, label=f'$\\Re(ω_{{s}})$ integration', color='cyan', linestyle='-', lw=1)
    ax.plot(coordinates, w_integration[1::2].imag, label=f'$\\Im(ω_{{s}})$ integration', color='magenta', linestyle='-', lw=1)
    ax.plot(coordinates, w_model[1::2].real, label=f'$\\Re(ω_{{s}})$ model', color='blue', linestyle=':', marker='x', lw=2)
    ax.plot(coordinates, w_model[1::2].imag, label=f'$\\Im(ω_{{s}})$ model', color='red', linestyle=':', marker='x', lw=2)
    ax.set_xlabel('x')
    ax.set_ylabel(f'$ω_{{s, z}}^{{0}}$')
    ax.set_title(f'$ω_{{s, z}}^{{0}}$ for {kwargs.get("type", "")} load with $p_0 = {p0:.1E}$')
    ax.legend()
    return fig