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
        self.config_path = Path(config_path).resolve()
        with open(self.config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)

    def load_config(self) -> dict[str, Any]:
        return self.config

    def _resolve_config_path(self, key: str) -> Path:
        raw_path = self.config[key]
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate

        repo_root = Path(__file__).resolve().parents[3]
        config_relative = (self.config_path.parent / candidate).resolve()
        repo_relative = (repo_root / candidate).resolve()

        if config_relative.exists():
            return config_relative
        if repo_relative.exists():
            return repo_relative
        return repo_relative

    def _require_existing_path(self, key: str) -> Path:
        path = self._resolve_config_path(key)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required ground_vibration input '{key}': {path}\n"
                "This problem depends on externally provided CSV/JSON artifacts. "
                "Place the dataset at the configured path or update the datagen YAML."
            )
        return path

    @staticmethod
    def _as_2d_rows(data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Expected params array to be 1D or 2D, got shape {arr.shape}.")
        return arr

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}.")
        return payload

    def _normalize_raw_inputs(
        self,
        pde_params_data: np.ndarray,
        mesh_metadata: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        params = self._as_2d_rows(pde_params_data)
        layout = [str(name).lower() for name in self.config.get(
            "params_array_layout",
            ["c11", "c13", "c33", "c44", "rho", "omega"],
        )]
        if params.shape[1] != len(layout):
            raise ValueError(
                "params_array column count does not match params_array_layout. "
                f"Got shape {params.shape} and layout {layout}."
            )

        columns = {name: params[:, idx] for idx, name in enumerate(layout)}
        required = ["c11", "c13", "c33", "c44", "rho"]
        missing_required = [name for name in required if name not in columns]
        if missing_required:
            raise KeyError(f"params_array is missing required columns: {missing_required}")

        c44 = np.asarray(columns["c44"], dtype=float)
        rho = np.asarray(columns["rho"], dtype=float)
        if np.any(c44 <= 0.0) or np.any(rho <= 0.0):
            raise ValueError("c44 and rho must be positive to compute shear-wave speed and a0.")

        eta = np.asarray(
            columns.get(
                "eta",
                np.full(params.shape[0], float(self.config.get("default_eta", 0.0)), dtype=float),
            ),
            dtype=float,
        )

        if "a0" in columns:
            a0 = np.asarray(columns["a0"], dtype=float)
            omega = np.asarray(columns.get("omega", np.full_like(a0, np.nan)), dtype=float)
        elif "omega" in columns:
            omega = np.asarray(columns["omega"], dtype=float)
            strip_half_width = self.config.get("strip_half_width", mesh_metadata.get("strip_half_width", None))
            if strip_half_width is None:
                raise KeyError(
                    "ground_vibration datagen requires 'strip_half_width' in config or metadata "
                    "to convert omega to normalized frequency a0."
                )
            b_value = float(strip_half_width)
            c_s = np.sqrt(c44 / rho)
            a0 = omega * b_value / np.maximum(c_s, 1e-14)
        else:
            raise KeyError("params_array must include either 'a0' or 'omega'.")

        if "omega" not in columns:
            strip_half_width = self.config.get("strip_half_width", mesh_metadata.get("strip_half_width", None))
            if strip_half_width is None:
                raise KeyError(
                    "ground_vibration datagen requires 'strip_half_width' in config or metadata "
                    "to convert a0 back to omega for bookkeeping."
                )
            b_value = float(strip_half_width)
            c_s = np.sqrt(c44 / rho)
            omega = a0 * c_s / max(b_value, 1e-14)

        xb = np.column_stack([
            np.asarray(columns["c11"], dtype=float),
            np.asarray(columns["c13"], dtype=float),
            np.asarray(columns["c33"], dtype=float),
            c44,
            rho,
            eta,
            a0,
        ])

        return {
            "xb": xb,
            "c11": np.asarray(columns["c11"], dtype=float),
            "c13": np.asarray(columns["c13"], dtype=float),
            "c33": np.asarray(columns["c33"], dtype=float),
            "c44": c44,
            "rho": rho,
            "eta": eta,
            "a0": a0,
            "omega": omega,
        }
    
    def _get_input_functions(self, data: np.ndarray) -> np.ndarray[Any, Any]:
        """Generate branch inputs for the full influence matrix operator.

        Args:
            data (np.ndarray): Branch input array of shape (N_samples, 7) with
                rows (c11, c13, c33, c44, rho, eta, a0).

        Returns:
            np.ndarray: Branch input vectors for the operator surrogate.
        """
        pde_sample = np.asarray(data, dtype=float)
        return pde_sample
    
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
        pde_params_path = self._require_existing_path('pde_params_data_path')
        mesh_params_path = self._require_existing_path('mesh_params_data_path')
        real_matrix_path = self._require_existing_path('real_influence_matrix_data_path')
        imag_matrix_path = self._require_existing_path('imag_influence_matrix_data_path')

        pde_params_data = np.loadtxt(pde_params_path, delimiter=',')
        mesh_metadata = self._load_json(mesh_params_path)
        mesh_params_data = mesh_metadata['x_positions']
        real_matrix_data = np.loadtxt(real_matrix_path, delimiter=',')
        imag_matrix_data = np.loadtxt(imag_matrix_path, delimiter=',')

        normalized = self._normalize_raw_inputs(
            pde_params_data=pde_params_data,
            mesh_metadata=mesh_metadata,
        )

        logger.info(f"Formatting...")
        pde_samples = self._get_input_functions(normalized["xb"])
        c11 = normalized["c11"]
        c13 = normalized["c13"]
        c33 = normalized["c33"]
        c44 = normalized["c44"]
        rho = normalized["rho"]
        eta = normalized["eta"]
        a0 = normalized["a0"]
        omega = normalized["omega"]

        mesh_points = self._get_coordinates(mesh_params_data)
        influence_matrix = self._influence_matrix(real_matrix_data=real_matrix_data, imag_matrix_data=imag_matrix_data, pde_samples=pde_samples, mesh_points=mesh_points)

        logger.info(
            f"\nData shapes:\nSample (c11, c13, c33, c44, rho, eta, a0): {pde_samples.shape},\n"
            f"min: ({c11.min(axis=0):.2f}, {c13.min(axis=0):.2f}, {c33.min(axis=0):.2f}, {c44.min(axis=0):.2f}, {rho.min(axis=0):.2f}, {eta.min(axis=0):.4f}, {a0.min(axis=0):.2f})\n"
            f"max: ({c11.max(axis=0):.2f}, {c13.max(axis=0):.2f}, {c33.max(axis=0):.2f}, {c44.max(axis=0):.2f}, {rho.max(axis=0):.2f}, {eta.max(axis=0):.4f}, {a0.max(axis=0):.2f})\n"
            f"mean:({c11.mean(axis=0):.2f}, {c13.mean(axis=0):.2f}, {c33.mean(axis=0):.2f}, {c44.mean(axis=0):.2f}, {rho.mean(axis=0):.2f}, {eta.mean(axis=0):.4f}, {a0.mean(axis=0):.2f})\n"
            f"std: ({c11.std(axis=0):.2f}, {c13.std(axis=0):.2f}, {c33.std(axis=0):.2f}, {c44.std(axis=0):.2f}, {rho.std(axis=0):.2f}, {eta.std(axis=0):.4f}, {a0.std(axis=0):.2f})\n"
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
                "η": {
                    "shape": len(pde_samples),
                    "min":  f"{eta.min(axis=0):.4f}",
                    "max":  f"{eta.max(axis=0):.4f}",
                    "mean": f"{eta.mean(axis=0):.4f}",
                    "std":  f"{eta.std(axis=0):.4f},"
                },
                "a0": {
                    "shape": len(pde_samples),
                    "min":  f"{a0.min(axis=0):.2f}",
                    "max":  f"{a0.max(axis=0):.2f}",
                    "mean": f"{a0.mean(axis=0):.2f}",
                    "std":  f"{a0.std(axis=0):.2f},"
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
            },
            "formulation": {
                "operator_input": ["c11", "c13", "c33", "c44", "rho", "eta", "a0"],
                "source_params_layout": self.config.get("params_array_layout", ["c11", "c13", "c33", "c44", "rho", "omega"]),
                "a0_definition": "a0 = omega * b / cS, cS = sqrt(c44 / rho)",
                "strip_half_width": float(self.config.get("strip_half_width", mesh_metadata.get("strip_half_width", 1.0))),
                "notes": [
                    "eta enters the viscoelastic constitutive law through c_ij* = c_ij (1 + i eta).",
                    "If source params provide omega instead of a0, the generator derives a0 from strip_half_width.",
                ],
            },
        }
        path = self._resolve_config_path("data_filename")
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path, 
            xb=pde_samples,
            c11=c11,
            c13=c13,
            c33=c33,
            c44=c44,
            ρ=rho,
            η=eta,
            a0=a0,
            ω=omega,
            x=mesh_points,
            g_u=influence_matrix
        )

        metadata_path = path.with_suffix('.yaml')

        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info(f"Saved data at {path}")
        logger.info(f"Saved metadata at {metadata_path}")
