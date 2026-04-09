from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.problems.base_generator import BaseProblemGenerator
from src.problems.kelvin.sampling_functions import mesh_rescaling

logger = logging.getLogger(__name__)


class KelvinProblemGenerator(BaseProblemGenerator):
    def __init__(self, config: str | dict[str, Any]):
        super().__init__(config=config)
        if isinstance(config, (str, Path)):
            self.config_path = config
            self.config = self.load_config()
        else:
            self.config_path = None
            self.config = config
        self.seed = self.config.get("seed")
        self.rng = np.random.default_rng(seed=self.seed)
        self.operator_mode = bool(self.config.get("operator_mode", False))

    def load_config(self):
        if self.config_path:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return self.config

    def _get_input_functions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample classic Kelvin branch variables (F, mu, nu)."""
        log_mu_samples = self.config["mu_min"] + self.rng.random(self.config["N"]) * (
            self.config["mu_max"] - self.config["mu_min"]
        )
        f_sample = -float(10**self.config["F"])
        mu_samples = 10**log_mu_samples
        nu_samples = self.rng.random(self.config["N"]) * (
            self.config["nu_max"] - self.config["nu_min"]
        )
        return np.array([f_sample]), mu_samples, nu_samples

    def _get_operator_materials(self) -> tuple[np.ndarray, np.ndarray]:
        n_samples = int(self.config["N"])
        mu_fixed = 10 ** float(self.config["mu_fixed_log10"])
        nu_fixed = float(self.config["nu_fixed"])
        mu_samples = np.full(shape=(n_samples,), fill_value=mu_fixed, dtype=np.float64)
        nu_samples = np.full(shape=(n_samples,), fill_value=nu_fixed, dtype=np.float64)
        return mu_samples, nu_samples

    def _source_spacing(self, x_source: np.ndarray) -> float:
        if x_source.size <= 1:
            return 1.0
        return float((x_source[-1] - x_source[0]) / max(x_source.size - 1, 1))

    def _sample_operator_load_profiles(self, n_samples: int, n_sources: int) -> np.ndarray:
        """Sample piecewise-constant distributed loads q(x)."""
        q = np.zeros((n_samples, n_sources), dtype=np.float64)

        q_log10_min = float(self.config.get("q_log10_min", 3.0))
        q_log10_max = float(self.config.get("q_log10_max", 6.0))
        blocks_min = int(self.config.get("q_num_blocks_min", 2))
        blocks_max = int(self.config.get("q_num_blocks_max", 6))
        allow_tension = bool(self.config.get("q_allow_tension", False))
        smooth_window = int(self.config.get("q_smooth_window", 1))

        for i in range(n_samples):
            n_blocks = int(self.rng.integers(low=blocks_min, high=blocks_max + 1))
            if n_blocks <= 1 or n_sources <= 2:
                split_points = np.array([], dtype=int)
            else:
                split_points = np.sort(
                    self.rng.choice(np.arange(1, n_sources), size=n_blocks - 1, replace=False)
                )

            starts = np.concatenate(([0], split_points))
            ends = np.concatenate((split_points, [n_sources]))
            for start, end in zip(starts, ends):
                magnitude = 10 ** self.rng.uniform(q_log10_min, q_log10_max)
                sign = float(self.rng.choice([-1.0, 1.0])) if allow_tension else -1.0
                q[i, start:end] = sign * magnitude

        q_noise_std = float(self.config.get("q_noise_std", 0.0))
        if q_noise_std > 0:
            q *= (1.0 + self.rng.normal(loc=0.0, scale=q_noise_std, size=q.shape))

        if smooth_window > 1:
            kernel = np.ones(smooth_window, dtype=np.float64) / float(smooth_window)
            for i in range(n_samples):
                q[i] = np.convolve(q[i], kernel, mode="same")

        target_total_abs_load = self.config.get("target_total_abs_load")
        if target_total_abs_load is not None:
            target_total_abs_load = float(target_total_abs_load)
            x_source = np.linspace(
                float(self.config["x_source_min"]),
                float(self.config["x_source_max"]),
                int(self.config["Ns"]),
            )
            dx = self._source_spacing(x_source=x_source)
            total_abs_load = np.sum(np.abs(q), axis=1) * dx
            scale = target_total_abs_load / np.maximum(total_abs_load, 1e-14)
            q *= scale[:, None]

        return q

    def _get_operator_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_samples = int(self.config["N"])
        n_sources = int(self.config["Ns"])
        x_source = np.linspace(
            float(self.config["x_source_min"]),
            float(self.config["x_source_max"]),
            n_sources,
        )
        q = self._sample_operator_load_profiles(n_samples=n_samples, n_sources=n_sources)
        mu_samples, nu_samples = self._get_operator_materials()
        return q, x_source, mu_samples, nu_samples

    def _get_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_field = np.linspace(self.config["x_min"] + 1e-6, self.config["x_max"] - 1e-6, self.config["N_x"])
        y_field = np.linspace(self.config["y_min"] + 1e-6, self.config["y_max"] - 1e-6, self.config["N_y"])
        z_field = np.linspace(self.config["z_min"] + 1e-6, self.config["z_max"] - 1e-6, self.config["N_z"])
        return x_field, y_field, z_field

    def _load_direction_index(self) -> int:
        if self.config["load_direction"] == "x":
            return 0
        if self.config["load_direction"] == "y":
            return 1
        if self.config["load_direction"] == "z":
            return 2
        raise ValueError("Invalid load direction. Must be 'x', 'y', or 'z'.")

    def _influencefunc(
        self,
        f: np.ndarray,
        mu: np.ndarray,
        nu: np.ndarray,
        x_field: np.ndarray,
        y_field: np.ndarray,
        z_field: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Classic Kelvin closed-form field with sampled (mu, nu) and fixed point load F."""
        start = time.perf_counter_ns()
        d = self._load_direction_index()

        x, y, z = np.meshgrid(x_field, y_field, z_field, indexing="ij")
        coords = np.stack([x, y, z], axis=-1)

        r_vals = np.linalg.norm(coords, axis=-1)
        r_b = r_vals[None, ...]

        const = f / (16 * np.pi * mu * (1 - nu))
        const = const[:, None, None, None]

        factor = (3 - 4 * nu)
        factor = factor[:, None, None, None]

        coords_b = coords[None, ...]

        delta = np.zeros(3)
        delta[d] = 1
        delta = delta.reshape(1, 1, 1, 1, 3)

        r_inv = 1 / r_b
        r_inv3 = 1 / (r_b**3)

        term1 = (factor[..., None] * delta) * r_inv[..., None]

        coord_d = coords_b[..., d : d + 1]
        term2 = (coords_b * coord_d) * r_inv3[..., None]

        u = const[..., None] * (term1 + term2)

        end = time.perf_counter_ns()
        duration = (end - start) / 1e9
        return u, duration

    def _influencefunc_operator(
        self,
        q: np.ndarray,
        x_source: np.ndarray,
        mu: np.ndarray,
        nu: np.ndarray,
        x_field: np.ndarray,
        y_field: np.ndarray,
        z_field: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Kelvin field for distributed line load q(x), using discrete superposition."""
        start = time.perf_counter_ns()

        d = self._load_direction_index()

        source_axis = str(self.config.get("source_axis", "x"))
        source_axis_map = {"x": 0, "y": 1, "z": 2}
        if source_axis not in source_axis_map:
            raise ValueError("Invalid source_axis. Must be 'x', 'y', or 'z'.")
        source_idx = source_axis_map[source_axis]

        fixed_source = np.array(
            [
                float(self.config.get("source_x", 0.0)),
                float(self.config.get("source_y", 0.0)),
                float(self.config.get("source_z", 0.0)),
            ],
            dtype=np.float64,
        )
        source_points = np.repeat(fixed_source[None, :], repeats=x_source.size, axis=0)
        source_points[:, source_idx] = x_source

        x, y, z = np.meshgrid(x_field, y_field, z_field, indexing="ij")
        eval_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        rel = eval_points[None, :, :] - source_points[:, None, :]
        r = np.linalg.norm(rel, axis=-1) + 1e-12

        mu0 = float(mu[0])
        nu0 = float(nu[0])
        if not np.allclose(mu, mu0) or not np.allclose(nu, nu0):
            raise ValueError(
                "Operator mode currently assumes fixed material properties across samples."
            )
        const = 1.0 / (16.0 * np.pi * mu0 * (1.0 - nu0))
        factor = 3.0 - 4.0 * nu0

        delta = np.zeros(3, dtype=np.float64)
        delta[d] = 1.0

        r_inv = 1.0 / r
        r_inv3 = 1.0 / (r**3)
        term1 = factor * delta[None, None, :] * r_inv[..., None]
        rel_d = rel[..., d : d + 1]
        term2 = rel * rel_d * r_inv3[..., None]

        dx = self._source_spacing(x_source=x_source)
        kernel = const * (term1 + term2) * dx

        u_flat = np.einsum("ns,spc->npc", q, kernel, optimize=True)
        u = u_flat.reshape(
            q.shape[0],
            x_field.shape[0],
            y_field.shape[0],
            z_field.shape[0],
            3,
        )

        end = time.perf_counter_ns()
        duration = (end - start) / 1e9
        return u, duration

    def generate(self):
        if self.operator_mode:
            q, x_source, mu, nu = self._get_operator_inputs()
            f = None
        else:
            f, mu, nu = self._get_input_functions()
            q = None
            x_source = None

        x_field_transformed, y_field_transformed, z_field_transformed = self._get_coordinates()

        x_field = mesh_rescaling(x_field_transformed, self.config["scaler"])
        y_field = mesh_rescaling(y_field_transformed, self.config["scaler"])
        z_field = mesh_rescaling(z_field_transformed, self.config["scaler"])

        logger.info("Generating...")
        if self.operator_mode:
            if q is None or x_source is None:
                raise RuntimeError("Operator mode is enabled, but q(x) inputs are missing.")
            displacements, duration = self._influencefunc_operator(
                q=q,
                x_source=x_source,
                mu=mu,
                nu=nu,
                x_field=x_field,
                y_field=y_field,
                z_field=z_field,
            )
            load_description = "distributed line load q(x)"
        else:
            if f is None:
                raise RuntimeError("Classic mode expects scalar force F.")
            displacements, duration = self._influencefunc(
                f=f,
                mu=mu,
                nu=nu,
                x_field=x_field,
                y_field=y_field,
                z_field=z_field,
            )
            load_description = f"point load magnitude {float(f[0]):.3E} N"

        scaler_parameter = self.config["scaler"]

        logger.info(
            f"Runtime for computing Kelvin solution: {duration*1e3:.3f} ms\n"
            f"Data shapes:\n"
            f"mu: {mu.shape}, nu: {nu.shape}\n"
            f"Displacements u: {displacements.shape}\n"
            f"x: {x_field.shape}, y: {y_field.shape}, z: {z_field.shape}\n"
            f"Load setup: {load_description}\n"
            f"Shear modulus min = {mu.min():.3E}, max = {mu.max():.3E}\n"
            f"Poisson's ratio min = {nu.min():.3f}, max = {nu.max():.3f}\n"
            f"g_u: min = {displacements.min():.3f}, max = {displacements.max():.3f}\n"
            f"g_u: mean = {displacements.mean():.3f}, std = {displacements.std():.3f}\n"
            f"scaling parameter = {scaler_parameter:.3f}"
        )

        metadata: dict[str, Any] = {
            "runtime_s": float(duration),
            "runtime_ms": float(duration * 1e3),
            "timing_breakdown": {
                "direct_solver_total_s": float(duration),
                "direct_solver_per_sample_s": float(duration / max(self.config["N"], 1)),
                "solver_kind": "kelvin_closed_form",
                "grid_points_per_sample": int(self.config["N_x"] * self.config["N_y"] * self.config["N_z"]),
            },
            "operator_mode": self.operator_mode,
            "parameters": {
                "shear_modulus": {
                    "shape": [i for i in mu.shape],
                    "min": f"{mu.min():.3E}",
                    "max": f"{mu.max():.3E}",
                    "mean": f"{mu.mean():.3E}",
                    "std": f"{mu.std():.3E}",
                },
                "poissons_ratio": {
                    "shape": [i for i in nu.shape],
                    "min": f"{nu.min():.3f}",
                    "max": f"{nu.max():.3f}",
                    "mean": f"{nu.mean():.3f}",
                    "std": f"{nu.std():.3f}",
                },
                "scaling_parameter": f"{scaler_parameter:.3f}",
            },
            "coordinate_statistics": {
                "x": {
                    "shape": [i for i in x_field.shape],
                    "min": f"{x_field.min():.3f}",
                    "max": f"{x_field.max():.3f}",
                    "mean": f"{x_field.mean():.3f}",
                    "std": f"{x_field.std():.3f}",
                },
                "y": {
                    "shape": [i for i in y_field.shape],
                    "min": f"{y_field.min():.3f}",
                    "max": f"{y_field.max():.3f}",
                    "mean": f"{y_field.mean():.3f}",
                    "std": f"{y_field.std():.3f}",
                },
                "z": {
                    "shape": [i for i in z_field.shape],
                    "min": f"{z_field.min():.3f}",
                    "max": f"{z_field.max():.3f}",
                    "mean": f"{z_field.mean():.3f}",
                    "std": f"{z_field.std():.3f}",
                },
            },
            "displacement_statistics": {
                "g_u": {
                    "shape": [i for i in displacements.shape],
                    "min": ", ".join([f"{i:.4E}" for i in displacements.min(axis=(0, 1, 2, 3))]),
                    "max": ", ".join([f"{i:.4E}" for i in displacements.max(axis=(0, 1, 2, 3))]),
                    "mean": ", ".join([f"{i:.4E}" for i in displacements.mean(axis=(0, 1, 2, 3))]),
                    "std": ", ".join([f"{i:.4E}" for i in displacements.std(axis=(0, 1, 2, 3))]),
                }
            },
            "paper_alignment": {
                "reference": "Kelvin fundamental solution for infinite isotropic elastic medium",
                "formulation_mode": (
                    "operator mode with distributed line load q(x)"
                    if self.operator_mode
                    else "closed-form displacement field with varying (mu, nu)"
                ),
                "load_setup": {
                    "load_direction": str(self.config["load_direction"]),
                },
            },
        }

        path = Path(self.config["data_filename"])
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        if self.operator_mode:
            if q is None or x_source is None:
                raise RuntimeError("Operator mode save requested without q(x) samples.")
            metadata["parameters"]["distributed_load"] = {
                "q_shape": [i for i in q.shape],
                "q_min": f"{q.min():.3E}",
                "q_max": f"{q.max():.3E}",
                "q_mean": f"{q.mean():.3E}",
                "q_std": f"{q.std():.3E}",
                "Ns": int(q.shape[1]),
                "source_axis": str(self.config.get("source_axis", "x")),
                "source_range": [float(x_source.min()), float(x_source.max())],
            }
            np.savez(
                path,
                q=q,
                q_x=x_source,
                mu=mu,
                nu=nu,
                x=x_field,
                y=y_field,
                z=z_field,
                g_u=displacements,
                c=scaler_parameter,
            )
        else:
            if f is None:
                raise RuntimeError("Classic mode save requested without scalar force F.")
            metadata["parameters"]["load_magnitude"] = f"{float(f[0]):.3E}"
            metadata["paper_alignment"]["load_setup"]["load_magnitude_N"] = float(f[0])
            np.savez(
                path,
                F=f,
                mu=mu,
                nu=nu,
                x=x_field,
                y=y_field,
                z=z_field,
                g_u=displacements,
                c=scaler_parameter,
            )

        metadata_path = path.with_suffix(".yaml")
        with open(metadata_path, "w", encoding="utf-8") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info(f"Saved data at {path}")
        logger.info(f"Saved metadata at {metadata_path}")
