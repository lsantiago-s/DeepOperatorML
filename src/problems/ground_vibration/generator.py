import yaml
import json
import logging
import numpy as np
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
            data (np.ndarray): PDE params is (N_samples, 6) array, where each row is (c44, c11, c33, c13, ρ, ω).

        Returns:
            pde_sample: Initial condition vector. Vector in space (x0, y0, z0) corresponding to the trajectorie's initial condition. 
        """
        pde_sample = data
        return pde_sample # (N, 6)
    
    def _get_coordinates(self, data: np.ndarray) -> np.ndarray[Any, Any]:
        "Generate timesteps array from t=t0 to t=tf, in seconds."
        return np.meshgrid(data, data)[1] # (N_points, n_points)

    def _influence_matrix(self, real_matrix_data: np.ndarray, imag_matrix_data: np.ndarray, pde_samples: np.ndarray, mesh_points: np.ndarray) -> np.ndarray[Any, Any]:
        "Generate influence matrix."
        matrix_data = (real_matrix_data + 1j * imag_matrix_data).reshape(len(pde_samples), len(mesh_points)**2,  -1)
        return matrix_data
    
    def generate(self):
        pde_params_data = np.loadtxt(self.config['pde_params_data_path'], delimiter=',')
        mesh_params_data = json.load(open(self.config['mesh_params_data_path'], 'r'))['x_positions']
        real_matrix_data = np.loadtxt(self.config['real_influence_matrix_data_path'], delimiter=',')
        imag_matrix_data = np.loadtxt(self.config['imag_influence_matrix_data_path'],delimiter=',')

        logger.info(f"Formatting...")
        pde_samples = self._get_input_functions(pde_params_data)
        mesh_points = self._get_coordinates(mesh_params_data)
        influence_matrix = self._influence_matrix(real_matrix_data=real_matrix_data, imag_matrix_data=imag_matrix_data, pde_samples=pde_samples, mesh_points=mesh_points)

        logger.info(
            f"\nData shapes:\nSample (c44, c11, c33, c13, ρ, ω): {pde_samples.shape},\nmin: ({pde_samples[:, 0].min(axis=0):.2f}, {pde_samples[:, 1].min(axis=0):.2f}, {pde_samples[:, 2].min(axis=0):.2f})\nmax: ({pde_samples[:, 0].max(axis=0):.2f}, {pde_samples[:, 1].max(axis=0):.2f}, {pde_samples[:, 2].max(axis=0):.2f})\nmean:({pde_samples[:, 0].mean(axis=0):.2f}, {pde_samples[:, 1].mean(axis=0):.2f}, {pde_samples[:, 2].mean(axis=0):.2f})\nstd: ({pde_samples[:, 0].std(axis=0):.2f}, {pde_samples[:, 1].std(axis=0):.2f}, {pde_samples[:, 2].std(axis=0):.2f})\nx=[{mesh_points.min()}, {mesh_points.max()}], {mesh_points.shape}, \nInfluence matrix: {influence_matrix.shape}.")
        
        metadata = {
            "pde_samples": {
                "c44": {
                    "shape": len(pde_samples),
                    "min":  f"{pde_samples[:, 0].min(axis=0):.2f}",
                    "max":  f"{pde_samples[:, 0].max(axis=0):.2f}",
                    "mean": f"{pde_samples[:, 0].mean(axis=0):.2f}",
                    "std":  f"{pde_samples[:, 0].std(axis=0):.2f},"
                },
                "c33": {
                    "shape": len(pde_samples),
                    "min":  f"{pde_samples[:, 1].min(axis=0):.2f}",
                    "max":  f"{pde_samples[:, 1].max(axis=0):.2f}",
                    "mean": f"{pde_samples[:, 1].mean(axis=0):.2f}",
                    "std":  f"{pde_samples[:, 1].std(axis=0):.2f},"
                },

                "c11": {
                    "shape": len(pde_samples),
                    "min":  f"{pde_samples[:, 2].min(axis=0):.2f}",
                    "max":  f"{pde_samples[:, 2].max(axis=0):.2f}",
                    "mean": f"{pde_samples[:, 2].mean(axis=0):.2f}",
                    "std":  f"{pde_samples[:, 2].std(axis=0):.2f},"
                },

                "c13": {
                    "shape": len(pde_samples),
                    "min":  f"{pde_samples[:, 3].min(axis=0):.2f}",
                    "max":  f"{pde_samples[:, 3].max(axis=0):.2f}",
                    "mean": f"{pde_samples[:, 3].mean(axis=0):.2f}",
                    "std":  f"{pde_samples[:, 3].std(axis=0):.2f},"
                },

                "ρ": {
                    "shape": len(pde_samples),
                    "min":  f"{pde_samples[:, 4].min(axis=0):.2f}",
                    "max":  f"{pde_samples[:, 4].max(axis=0):.2f}",
                    "mean": f"{pde_samples[:, 4].mean(axis=0):.2f}",
                    "std":  f"{pde_samples[:, 4].std(axis=0):.2f},"
                },

                "ω": {
                    "shape": len(pde_samples),
                    "min":  f"{pde_samples[:, 5].min(axis=0):.2f}",
                    "max":  f"{pde_samples[:, 5].max(axis=0):.2f}",
                    "mean": f"{pde_samples[:, 5].mean(axis=0):.2f}",
                    "std":  f"{pde_samples[:, 5].std(axis=0):.2f},"
                },

            },
            "mesh_points": {
                "x": {
                    "shape": len(mesh_points.flatten()),
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
                }
            }
        }
        path = Path(self.config["data_filename"])
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path, 
            c44=pde_samples[:, 0], 
            c33=pde_samples[:, 1], 
            c11=pde_samples[:, 2], 
            c13=pde_samples[:, 3], 
            ρ=pde_samples[:, 4], 
            ω=pde_samples[:, 5], 
            x=mesh_points,
            g_u=influence_matrix
        )

        metadata_path = path.with_suffix('.yaml')

        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info(f"Saved data at {path}")
        logger.info(f"Saved metadata at {metadata_path}")
