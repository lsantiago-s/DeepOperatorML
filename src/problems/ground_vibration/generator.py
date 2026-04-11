from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.problems.base_generator import BaseProblemGenerator

logger = logging.getLogger(__name__)


class GroundVibrationProblemGenerator(BaseProblemGenerator):
    """Build raw NPZ for homogeneous-soil full influence matrix learning.

    Raw output contract:
    - branch input xb: (n_samples, 7) = [c11, c13, c33, c44, rho, eta, a0]
    - trunk query geometry (saved as vectors): x, s1, s2 with length N each
    - operator target g_u: (n_samples, N*N, 4) complex channels [u_xx, u_xz, u_zx, u_zz]
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.config_path = Path(config_path).resolve()
        with open(self.config_path, "r", encoding="utf-8") as file:
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
        with open(path, "r", encoding="utf-8") as f:
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
        layout = [
            str(name).lower()
            for name in self.config.get(
                "params_array_layout",
                ["c11", "c13", "c33", "c44", "rho", "eta", "a0", "omega"],
            )
        ]
        if params.shape[1] != len(layout):
            raise ValueError(
                "params_array column count does not match params_array_layout. "
                f"Got shape {params.shape} and layout {layout}."
            )

        columns = {name: np.asarray(params[:, idx], dtype=float) for idx, name in enumerate(layout)}
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

        strip_half_width = self.config.get("strip_half_width", mesh_metadata.get("strip_half_width", None))
        if strip_half_width is None:
            strip_half_width = mesh_metadata.get("source_half_width", None)
        if strip_half_width is None:
            raise KeyError(
                "ground_vibration datagen requires strip_half_width/source_half_width in config or metadata "
                "to convert between a0 and omega."
            )
        b_value = float(strip_half_width)

        if "a0" in columns:
            a0 = np.asarray(columns["a0"], dtype=float)
            omega = np.asarray(columns.get("omega", np.full_like(a0, np.nan)), dtype=float)
        elif "omega" in columns:
            omega = np.asarray(columns["omega"], dtype=float)
            c_s = np.sqrt(c44 / rho)
            a0 = omega * b_value / np.maximum(c_s, 1e-14)
        else:
            raise KeyError("params_array must include either 'a0' or 'omega'.")

        if "omega" not in columns:
            c_s = np.sqrt(c44 / rho)
            omega = a0 * c_s / max(b_value, 1e-14)

        xb = np.column_stack(
            [
                np.asarray(columns["c11"], dtype=float),
                np.asarray(columns["c13"], dtype=float),
                np.asarray(columns["c33"], dtype=float),
                c44,
                rho,
                eta,
                a0,
            ]
        )

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
            "strip_half_width": np.asarray(b_value, dtype=float),
        }

    def _extract_geometry(self, mesh_metadata: dict[str, Any]) -> dict[str, np.ndarray]:
        if "x_positions" not in mesh_metadata:
            raise KeyError("mesh metadata must contain 'x_positions'.")

        x = np.asarray(mesh_metadata["x_positions"], dtype=float).reshape(-1)
        n_nodes = x.shape[0]
        if n_nodes == 0:
            raise ValueError("x_positions is empty.")

        if "source_element_s1" in mesh_metadata and "source_element_s2" in mesh_metadata:
            s1 = np.asarray(mesh_metadata["source_element_s1"], dtype=float).reshape(-1)
            s2 = np.asarray(mesh_metadata["source_element_s2"], dtype=float).reshape(-1)
        else:
            source_half_width = mesh_metadata.get("source_half_width", self.config.get("strip_half_width", None))
            if source_half_width is None:
                dx = mesh_metadata.get("dx", None)
                if dx is not None:
                    source_half_width = 0.5 * float(dx)
            if source_half_width is None:
                raise KeyError(
                    "mesh metadata must provide source_element_s1/source_element_s2 or source_half_width/dx."
                )
            b = float(source_half_width)
            s1 = x - b
            s2 = x + b

        if s1.shape[0] != n_nodes or s2.shape[0] != n_nodes:
            raise ValueError(
                "Geometry vectors must have same length. "
                f"Got len(x)={n_nodes}, len(s1)={s1.shape[0]}, len(s2)={s2.shape[0]}."
            )

        return {"x": x, "s1": s1, "s2": s2}

    @staticmethod
    def _build_query_geometry(geometry: dict[str, np.ndarray]) -> np.ndarray:
        x = geometry["x"]
        s1 = geometry["s1"]
        s2 = geometry["s2"]
        n_nodes = x.shape[0]

        # Flatten order matches g_u reshape from (field_i, source_j) with source index varying fastest.
        return np.column_stack(
            [
                np.repeat(x, n_nodes),
                np.tile(s1, n_nodes),
                np.tile(s2, n_nodes),
            ]
        )

    def _influence_matrix(
        self,
        real_matrix_data: np.ndarray,
        imag_matrix_data: np.ndarray,
        pde_samples: np.ndarray,
        mesh_points: np.ndarray,
    ) -> np.ndarray[Any, Any]:
        """Decode flattened full matrices into (samples, N*N, 4) channels."""
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
                blocks[..., 0, 0],
                blocks[..., 0, 1],
                blocks[..., 1, 0],
                blocks[..., 1, 1],
            ],
            axis=-1,
        )

        return channels.reshape(num_samples, n_nodes * n_nodes, 4)

    @staticmethod
    def _summary_stats(values: np.ndarray) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    def generate(self):
        start = time.perf_counter()
        pde_params_path = self._require_existing_path("pde_params_data_path")
        mesh_params_path = self._require_existing_path("mesh_params_data_path")
        real_matrix_path = self._require_existing_path("real_influence_matrix_data_path")
        imag_matrix_path = self._require_existing_path("imag_influence_matrix_data_path")

        pde_params_data = np.loadtxt(pde_params_path, delimiter=",")
        mesh_metadata = self._load_json(mesh_params_path)
        real_matrix_data = np.loadtxt(real_matrix_path, delimiter=",")
        imag_matrix_data = np.loadtxt(imag_matrix_path, delimiter=",")

        normalized = self._normalize_raw_inputs(
            pde_params_data=pde_params_data,
            mesh_metadata=mesh_metadata,
        )
        geometry = self._extract_geometry(mesh_metadata)
        xt_geometry = self._build_query_geometry(geometry)

        pde_samples = np.asarray(normalized["xb"], dtype=float)
        c11 = normalized["c11"]
        c13 = normalized["c13"]
        c33 = normalized["c33"]
        c44 = normalized["c44"]
        rho = normalized["rho"]
        eta = normalized["eta"]
        a0 = normalized["a0"]
        omega = normalized["omega"]

        x = geometry["x"]
        s1 = geometry["s1"]
        s2 = geometry["s2"]

        influence_matrix = self._influence_matrix(
            real_matrix_data=real_matrix_data,
            imag_matrix_data=imag_matrix_data,
            pde_samples=pde_samples,
            mesh_points=x,
        )

        n_nodes = x.shape[0]
        influence_tensor = influence_matrix.reshape(len(pde_samples), n_nodes, n_nodes, 4)
        reciprocity_error = np.linalg.norm(
            influence_tensor - influence_tensor.transpose(0, 2, 1, 3)[..., [0, 2, 1, 3]]
        ) / (np.linalg.norm(influence_tensor) + 1e-30)

        duration = time.perf_counter() - start

        external_solver_total = mesh_metadata.get("total_time_s", None)
        external_solver_per_sample = mesh_metadata.get("sample_times_mean_s", None)

        logger.info(
            "Ground-vibration data formatted: xb=%s, xt=%s, g_u=%s",
            pde_samples.shape,
            xt_geometry.shape,
            influence_matrix.shape,
        )

        metadata = {
            "runtime_s": float(duration),
            "runtime_ms": float(duration * 1e3),
            "reference_solver_total_s": None if external_solver_total is None else float(external_solver_total),
            "reference_solver_per_sample_s": (
                None if external_solver_per_sample is None else float(external_solver_per_sample)
            ),
            "pde_samples": {
                "shape": int(len(pde_samples)),
                "c11": self._summary_stats(c11),
                "c13": self._summary_stats(c13),
                "c33": self._summary_stats(c33),
                "c44": self._summary_stats(c44),
                "rho": self._summary_stats(rho),
                "eta": self._summary_stats(eta),
                "a0": self._summary_stats(a0),
                "omega": self._summary_stats(omega),
            },
            "geometry": {
                "x": {
                    "shape": int(n_nodes),
                    "min": float(np.min(x)),
                    "max": float(np.max(x)),
                    "mean": float(np.mean(x)),
                    "std": float(np.std(x)),
                },
                "s1": {
                    "shape": int(n_nodes),
                    "min": float(np.min(s1)),
                    "max": float(np.max(s1)),
                    "mean": float(np.mean(s1)),
                    "std": float(np.std(s1)),
                },
                "s2": {
                    "shape": int(n_nodes),
                    "min": float(np.min(s2)),
                    "max": float(np.max(s2)),
                    "mean": float(np.mean(s2)),
                    "std": float(np.std(s2)),
                },
                "query_layout": ["x_field", "s1_source", "s2_source"],
                "xt_shape": [int(v) for v in xt_geometry.shape],
            },
            "influence_matrix": {
                "g_u": {
                    "shape": [int(v) for v in influence_matrix.shape],
                    "real_min": [float(v) for v in np.min(influence_matrix.real, axis=(0, 1))],
                    "real_max": [float(v) for v in np.max(influence_matrix.real, axis=(0, 1))],
                    "imag_min": [float(v) for v in np.min(influence_matrix.imag, axis=(0, 1))],
                    "imag_max": [float(v) for v in np.max(influence_matrix.imag, axis=(0, 1))],
                },
                "channel_layout": ["u_xx", "u_xz", "u_zx", "u_zz"],
                "flatten_layout": "(field_i, source_j)",
                "reciprocity_relative_error": float(reciprocity_error),
            },
            "formulation": {
                "operator_input": ["c11", "c13", "c33", "c44", "rho", "eta", "a0"],
                "operator_query": ["x_m", "s1_n", "s2_n"],
                "output": "u_ij^(mn) with channels [u_xx, u_xz, u_zx, u_zz]",
                "params_array_layout": self.config.get(
                    "params_array_layout", ["c11", "c13", "c33", "c44", "rho", "eta", "a0", "omega"]
                ),
                "a0_definition": "a0 = omega * b / cS, cS = sqrt(c44 / rho)",
                "strip_half_width": float(normalized["strip_half_width"]),
                "notes": [
                    "eta enters viscoelastic constitutive law as c_ij* = c_ij (1 + i eta).",
                    "If source params provide omega instead of a0, the generator derives a0 from strip_half_width.",
                    "Geometry query is kept explicit as (x_m, s1_n, s2_n) for operator-learning trunk inputs.",
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
            rho=rho,
            eta=eta,
            a0=a0,
            omega=omega,
            # Keep unicode aliases for backward compatibility with older configs/scripts.
            **{
                "ρ": rho,
                "η": eta,
                "ω": omega,
            },
            x=x,
            s1=s1,
            s2=s2,
            xt_geometry=xt_geometry,
            g_u=influence_matrix,
        )

        metadata_path = path.with_suffix(".yaml")
        with open(metadata_path, "w", encoding="utf-8") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info("Saved ground_vibration raw data at %s", path)
        logger.info("Saved ground_vibration metadata at %s", metadata_path)
