import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plot_w_displacement import plot_w_displacement

def compute_uniform_surface_traction_vector(
    x: np.ndarray,
    p0: float,
    complex_dtype: np.dtype = np.complex128
) -> np.ndarray:
    """
    Uniform vertical traction p0 applied at all nodes (q_x = 0, q_z = p0).
    """
    Ns = len(x)
    q = np.zeros((2 * Ns, 1), dtype=complex_dtype)
    q[1::2, 0] = p0  # vertical tractions only
    return q

def compute_harmonic_surface_traction_vector(
    x: np.ndarray,
    p0: float,
    k: float,
    complex_dtype: np.dtype = np.complex128
) -> np.ndarray:
    """
    Harmonic vertical traction of form q_z = p0 * exp(-1j * k * x).
    Represents a standing or traveling harmonic load.
    """
    Ns = len(x)
    q = np.zeros((2 * Ns, 1), dtype=complex_dtype)
    q[1::2, 0] = p0 * np.exp(-1j * k * x)  # vertical tractions
    return q

def compute_patch_surface_traction_vector(
    x: np.ndarray,
    p0: float,
    patch_mask: np.ndarray,
    normalize: bool = True,
    complex_dtype: np.dtype = np.complex128
) -> np.ndarray:

    Ns = len(x)
    q = np.zeros((2 * Ns, 1), dtype=complex_dtype)

    mask = patch_mask.astype(float)
    if normalize and mask.sum() > 0:
        mask /= mask.sum()  # ensures total force consistency

    q[1::2, 0] = p0 * mask
    return q

if __name__ == "__main__":
    U_path = '/Users/ls/workspace/DeepOperatorML/output/ground_vibration/2025-11-10_14-23-36/aux/U_matrices.npz'
    output_path = '/Users/ls/workspace/DeepOperatorML/output/ground_vibration/2025-11-10_14-23-36/aux/output_data.npz'
    U_data = np.load(U_path, allow_pickle=True)
    output_data = np.load(output_path, allow_pickle=True)
    raw_data = np.load('/Users/ls/workspace/DeepOperatorML/data/raw/ground_vibration/ground_vibration_v1.npz')
    
    coords = output_data['xt']
    x = np.unique(coords[:, 0])
    print(f"x = {x}")
    L = x.max() - x.min()
    print(f"L = {L}")

    p0 = -1e3
    U_true = U_data['U_true']
    U_pred = U_data['U_pred']

    Ns = len(x)
    patch_mask = (x >= -0.25 * L) & (x <= 0.25 * L)


    q_uniform = compute_uniform_surface_traction_vector(x=x, p0=p0)
    q_harmonic = compute_harmonic_surface_traction_vector(x=x, p0=p0, k=50.0)
    q_patch = compute_patch_surface_traction_vector(x=x, p0=p0, patch_mask=patch_mask)

    # Compute displacements
    w_true_uniform = U_true @ q_uniform
    w_pred_uniform = U_pred @ q_uniform

    w_true_harmonic = U_true @ q_harmonic
    w_pred_harmonic = U_pred @ q_harmonic

    w_true_patch = U_true @ q_patch
    w_pred_patch = U_pred @ q_patch

    # Create output directory if it doesn't exist
    output_dir = Path('/Users/ls/workspace/DeepOperatorML/output/ground_vibration/2025-11-10_14-23-36/plots/displacement_plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample in range(len(w_true_uniform)):
        fig_1 = plot_w_displacement(w_integration=w_true_uniform[sample, :, 0], w_model=w_pred_uniform[sample, :, 0], coordinates=x, type='uniform', p0=p0)
        plt.savefig(output_dir / f'uniform_sample_{sample}.png', dpi=300)
        plt.close(fig_1)

        fig_2 = plot_w_displacement(w_integration=w_true_harmonic[sample, :, 0], w_model=w_pred_harmonic[sample, :, 0], coordinates=x, type='harmonic', p0=p0)
        plt.savefig(output_dir / f'harmonic_sample_{sample}.png', dpi=300)
        plt.close(fig_2)

        fig_3 = plot_w_displacement(w_integration=w_true_patch[sample, :, 0], w_model=w_pred_patch[sample, :, 0], coordinates=x, type='patch', p0=p0)
        plt.savefig(output_dir / f'patch_sample_{sample}.png', dpi=300)
        plt.close(fig_3)