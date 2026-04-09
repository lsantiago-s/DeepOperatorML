import yaml
import json
import logging
import numpy as np
import time
from typing import Any
from pathlib import Path
from src.problems.base_generator import BaseProblemGenerator

logger = logging.getLogger(__name__)

class GroundVibrationProblemGenerator(BaseProblemGenerator):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def load_config(self) -> dict[str, Any]:
        return self.config
    
    def _get_input_functions(self, data: np.ndarray) -> np.ndarray[Any, Any]:
        """Generate input functions to the operator (input to the branch).

        Args:
            data (np.ndarray): PDE params is (N_samples, 6) array, where each row is
                (c11, c13, c33, c44, ρ, ω).

        Returns:
            pde_sample: Initial condition vector. Vector in space (x0, y0, z0) corresponding to the trajectorie's initial condition. 
        """
        pde_sample = data
        return pde_sample # (N, 6)
    
    def _get_coordinates(self, data: np.ndarray) -> np.ndarray[Any, Any]:
        """Return 1D surface-node coordinates."""
        coords = np.asarray(data, dtype=float).reshape(-1)
        return coords

    def _influence_matrix(self, real_matrix_data: np.ndarray, imag_matrix_data: np.ndarray, pde_samples: np.ndarray, mesh_points: np.ndarray) -> np.ndarray[Any, Any]:
        """Decode flattened MATLAB matrices into (samples, N*N, 4) complex channels.

        Expected channel order is:
        [u_xx, u_xz, u_zx, u_zz], where each channel is indexed by (field_i, source_j).
        """
        num_samples = len(pde_samples)
        n_nodes = len(mesh_points)
        ndof = 2 * n_nodes
        expected_flat = ndof * ndof

        real_matrix_data = np.asarray(real_matrix_data, dtype=float)
        imag_matrix_data = np.asarray(imag_matrix_data, dtype=float)

        if real_matrix_data.shape != imag_matrix_data.shape:
            raise ValueError(
                "Real/imag matrix CSV shapes must match. "
                f"Got {real_matrix_data.shape} vs {imag_matrix_data.shape}."
            )
        if real_matrix_data.shape[0] != num_samples:
            raise ValueError(
                "Number of matrix rows must match number of parameter samples. "
                f"Got matrix rows={real_matrix_data.shape[0]} and samples={num_samples}."
            )
        if real_matrix_data.shape[1] != expected_flat:
            raise ValueError(
                "Unexpected flattened matrix size. "
                f"Expected {expected_flat} (for 2N x 2N with N={n_nodes}), got {real_matrix_data.shape[1]}."
            )

        matrix_data_flat = real_matrix_data + 1j * imag_matrix_data
        full_matrices = np.empty((num_samples, ndof, ndof), dtype=np.complex128)

        # Each CSV row was generated from MATLAB linear indexing (column-major).
        for sample_idx in range(num_samples):
            full_matrices[sample_idx] = matrix_data_flat[sample_idx].reshape((ndof, ndof), order="F")

        # (sample, i, j, 2, 2) block for node pair (field i, source j)
        blocks = full_matrices.reshape(num_samples, n_nodes, 2, n_nodes, 2).transpose(0, 1, 3, 2, 4)

        channels = np.stack(
            [
                blocks[..., 0, 0],  # u_xx
                blocks[..., 0, 1],  # u_xz
                blocks[..., 1, 0],  # u_zx
                blocks[..., 1, 1],  # u_zz
            ],
            axis=-1,
        )

        return channels.reshape(num_samples, n_nodes * n_nodes, 4)
    
    def generate(self):
        start = time.perf_counter()
        pde_params_data = np.loadtxt(self.config['pde_params_data_path'], delimiter=',')
        mesh_params_data = json.load(open(self.config['mesh_params_data_path'], 'r'))['x_positions']
        real_matrix_data = np.loadtxt(self.config['real_influence_matrix_data_path'], delimiter=',')
        imag_matrix_data = np.loadtxt(self.config['imag_influence_matrix_data_path'],delimiter=',')

        logger.info(f"Formatting...")
        pde_samples = self._get_input_functions(pde_params_data)
        # params_array columns from MATLAB generator:
        # [c11, c13, c33, c44, rho, omega]
        c11 = pde_samples[:, 0]
        c13 = pde_samples[:, 1]
        c33 = pde_samples[:, 2]
        c44 = pde_samples[:, 3]
        rho = pde_samples[:, 4]
        omega = pde_samples[:, 5]

        mesh_points = self._get_coordinates(mesh_params_data)
        influence_matrix = self._influence_matrix(real_matrix_data=real_matrix_data, imag_matrix_data=imag_matrix_data, pde_samples=pde_samples, mesh_points=mesh_points)

        logger.info(
            f"\nData shapes:\nSample (c11, c13, c33, c44, ρ, ω): {pde_samples.shape},\n"
            f"min: ({c11.min(axis=0):.2f}, {c13.min(axis=0):.2f}, {c33.min(axis=0):.2f}, {c44.min(axis=0):.2f})\n"
            f"max: ({c11.max(axis=0):.2f}, {c13.max(axis=0):.2f}, {c33.max(axis=0):.2f}, {c44.max(axis=0):.2f})\n"
            f"mean:({c11.mean(axis=0):.2f}, {c13.mean(axis=0):.2f}, {c33.mean(axis=0):.2f}, {c44.mean(axis=0):.2f})\n"
            f"std: ({c11.std(axis=0):.2f}, {c13.std(axis=0):.2f}, {c33.std(axis=0):.2f}, {c44.std(axis=0):.2f})\n"
            f"x=[{mesh_points.min()}, {mesh_points.max()}], {mesh_points.shape}, \n"
            f"Influence matrix: {influence_matrix.shape}.")

        influence_tensor = influence_matrix.reshape(len(pde_samples), len(mesh_points), len(mesh_points), 4)
        reciprocity_error = np.linalg.norm(
            influence_tensor - influence_tensor.transpose(0, 2, 1, 3)[..., [0, 2, 1, 3]]
        ) / (np.linalg.norm(influence_tensor) + 1e-30)
        
        duration = time.perf_counter() - start
        metadata = {
            "runtime_s": float(duration),
            "runtime_ms": float(duration * 1e3),
            "pde_samples": {
                "c11": {
                    "shape": len(pde_samples),
                    "min":  f"{c11.min(axis=0):.2f}",
                    "max":  f"{c11.max(axis=0):.2f}",
                    "mean": f"{c11.mean(axis=0):.2f}",
                    "std":  f"{c11.std(axis=0):.2f},"
                },
                "c13": {
                    "shape": len(pde_samples),
                    "min":  f"{c13.min(axis=0):.2f}",
                    "max":  f"{c13.max(axis=0):.2f}",
                    "mean": f"{c13.mean(axis=0):.2f}",
                    "std":  f"{c13.std(axis=0):.2f},"
                },

                "c33": {
                    "shape": len(pde_samples),
                    "min":  f"{c33.min(axis=0):.2f}",
                    "max":  f"{c33.max(axis=0):.2f}",
                    "mean": f"{c33.mean(axis=0):.2f}",
                    "std":  f"{c33.std(axis=0):.2f},"
                },

                "c44": {
                    "shape": len(pde_samples),
                    "min":  f"{c44.min(axis=0):.2f}",
                    "max":  f"{c44.max(axis=0):.2f}",
                    "mean": f"{c44.mean(axis=0):.2f}",
                    "std":  f"{c44.std(axis=0):.2f},"
                },

                "ρ": {
                    "shape": len(pde_samples),
                    "min":  f"{rho.min(axis=0):.2f}",
                    "max":  f"{rho.max(axis=0):.2f}",
                    "mean": f"{rho.mean(axis=0):.2f}",
                    "std":  f"{rho.std(axis=0):.2f},"
                },

                "ω": {
                    "shape": len(pde_samples),
                    "min":  f"{omega.min(axis=0):.2f}",
                    "max":  f"{omega.max(axis=0):.2f}",
                    "mean": f"{omega.mean(axis=0):.2f}",
                    "std":  f"{omega.std(axis=0):.2f},"
                },

            },
            "mesh_points": {
                "x": {
                    "shape": len(mesh_points),
                    "min":  f"{mesh_points.min():.2f}",
                    "max":  f"{mesh_points.max():.2f}",
                    "mean": f"{mesh_points.mean():.2f}",
                    "std":  f"{mesh_points.std():.2f}"
                },
            },
            "influence_matrix": {
                "g_u": {
                    "shape": [i for i in influence_matrix.shape],
                    "min":  ', '.join([f'{i:.2E}' for i in influence_matrix.min(axis=(0, 1))]),
                    "max":  ', '.join([f'{i:.2E}' for i in influence_matrix.max(axis=(0, 1))]),
                    "mean": ', '.join([f'{i:.2E}' for i in influence_matrix.mean(axis=(0, 1))]),
                    "std":  ', '.join([f'{i:.2E}' for i in influence_matrix.std(axis=(0, 1))])
                },
                "layout": "flattened (field_i, source_j) x [u_xx, u_xz, u_zx, u_zz]",
                "reciprocity_relative_error": float(reciprocity_error),
            }
        }
        path = Path(self.config["data_filename"])
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path, 
            c11=c11,
            c13=c13,
            c33=c33,
            c44=c44,
            ρ=rho,
            ω=omega,
            x=mesh_points,
            g_u=influence_matrix
        )

        metadata_path = path.with_suffix('.yaml')

        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info(f"Saved data at {path}")
        logger.info(f"Saved metadata at {metadata_path}")
