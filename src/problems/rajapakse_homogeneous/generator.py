from __future__ import annotations
import time
import yaml
import logging
import numpy as np
from typing import Any
from pathlib import Path
from tqdm.auto import tqdm
from src.problems.base_generator import BaseProblemGenerator
from src.problems.rajapakse_homogeneous.influence import _ensure_library_available, influence

logger = logging.getLogger(__name__)


class RajapakseHomogeneousGenerator(BaseProblemGenerator):
    """Generator for homogeneous half-space Green functions (Rajapakse & Wang style)."""

    def __init__(self, config: str | dict[str, Any]):
        super().__init__(config)
        if isinstance(config, (str, Path)):
            self.config_path = config
            self.config = self.load_config()
        else:
            self.config_path = None
            self.config = config
        self._normalize_config_aliases()

    def load_config(self) -> dict[str, Any]:
        if self.config_path:
            with open(self.config_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        return self.config

    def _normalize_config_aliases(self) -> None:
        aliases = {
            "N_R": "N_r",
            "N_Z": "N_z",
            "dens_min": "rho_min",
            "dens_max": "rho_max",
        }
        for old, new in aliases.items():
            if old in self.config and new not in self.config:
                self.config[new] = self.config[old]

    def _sample_isotropic_parameters(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        n = int(self.config["N"])
        e_min = float(self.config["E_min"])
        e_max = float(self.config["E_max"])
        nu_min = float(self.config["nu_min"])
        nu_max = float(self.config["nu_max"])
        rho_min = float(self.config["rho_min"])
        rho_max = float(self.config["rho_max"])
        omega_min = float(self.config["omega_min"])
        omega_max = float(self.config["omega_max"])
        r_source = float(self.config["r_source"])

        log_e = rng.uniform(np.log10(e_min), np.log10(e_max), size=n)
        e = np.power(10.0, log_e)
        nu = rng.uniform(nu_min, nu_max, size=n)
        rho = rng.uniform(rho_min, rho_max, size=n)
        omega = rng.uniform(omega_min, omega_max, size=n)

        e1 = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
        c11 = e1 * (1.0 - nu)
        c12 = e1 * nu
        c13 = e1 * nu
        c33 = e1 * (1.0 - nu)
        c44 = e1 * (1.0 - 2.0 * nu) / 2.0

        c11n = c11 / c44
        c12n = c12 / c44
        c13n = c13 / c44
        c33n = c33 / c44
        c44n = np.ones_like(c11n)

        delta = omega * r_source * np.sqrt(rho / c44)
        dens_n = np.ones_like(delta)
        omega_n = delta.copy()

        return {
            "E": e,
            "nu": nu,
            "rho": rho,
            "omega": omega,
            "c11_over_c44": c11n,
            "c12_over_c44": c12n,
            "c13_over_c44": c13n,
            "c33_over_c44": c33n,
            "c44_over_c44": c44n,
            "rho_over_rho0": dens_n,
            "delta": delta,
            "omega_for_solver": omega_n,
            "sampling_mode": np.array(["isotropic_from_E_nu"] * n, dtype=object),
        }

    @staticmethod
    def _is_stable_ti(c11: float, c12: float, c13: float, c33: float, c44: float) -> bool:
        if c44 <= 0.0 or c33 <= 0.0:
            return False
        if c11 <= abs(c12):
            return False
        return (c11 + c12) * c33 > 2.0 * (c13 ** 2)

    def _sample_ti_parameters(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        n = int(self.config["N"])
        max_tries = int(self.config.get("max_tries_per_sample", 2000))

        c11_range = self.config["c11_over_c44"]
        c12_range = self.config["c12_over_c44"]
        c13_range = self.config["c13_over_c44"]
        c33_range = self.config["c33_over_c44"]
        rho_range = self.config.get("rho_over_rho0", [1.0, 1.0])
        delta_min = float(self.config["delta_min"])
        delta_max = float(self.config["delta_max"])

        c11n = np.empty(n, dtype=float)
        c12n = np.empty(n, dtype=float)
        c13n = np.empty(n, dtype=float)
        c33n = np.empty(n, dtype=float)
        c44n = np.ones(n, dtype=float)
        rho_n = np.empty(n, dtype=float)
        delta = rng.uniform(delta_min, delta_max, size=n)

        for i in range(n):
            valid = False
            for _ in range(max_tries):
                c11 = rng.uniform(float(c11_range[0]), float(c11_range[1]))
                c12 = rng.uniform(float(c12_range[0]), float(c12_range[1]))
                c13 = rng.uniform(float(c13_range[0]), float(c13_range[1]))
                c33 = rng.uniform(float(c33_range[0]), float(c33_range[1]))
                if self._is_stable_ti(c11, c12, c13, c33, 1.0):
                    c11n[i] = c11
                    c12n[i] = c12
                    c13n[i] = c13
                    c33n[i] = c33
                    valid = True
                    break
            if not valid:
                raise RuntimeError(
                    "Unable to sample stable transversely isotropic constants. "
                    "Broaden ranges or increase max_tries_per_sample."
                )
            rho_n[i] = rng.uniform(float(rho_range[0]), float(rho_range[1]))

        # Keep nondimensional frequency delta comparable across varying density.
        omega_n = delta / np.sqrt(np.maximum(rho_n, 1e-12))

        return {
            "c11_over_c44": c11n,
            "c12_over_c44": c12n,
            "c13_over_c44": c13n,
            "c33_over_c44": c33n,
            "c44_over_c44": c44n,
            "rho_over_rho0": rho_n,
            "delta": delta,
            "omega_for_solver": omega_n,
            "sampling_mode": np.array(["transversely_isotropic"] * n, dtype=object),
        }

    def _get_input_functions(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        mode = str(self.config.get("sampling_mode", "transversely_isotropic"))
        if mode == "isotropic_from_E_nu":
            return self._sample_isotropic_parameters(rng=rng)
        return self._sample_ti_parameters(rng=rng)

    def _get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        n_r = int(self.config["N_r"])
        n_z = int(self.config["N_z"])
        r_source = float(self.config["r_source"])

        r_min = float(self.config["r_min"]) + (r_source * 1e-2)
        r_max = float(self.config["r_max"])
        z_min = float(self.config["z_min"])
        z_max = float(self.config["z_max"])

        r_field = np.linspace(r_min, r_max, n_r) / r_source
        z_field = np.linspace(z_min, z_max, n_z) / r_source
        return r_field, z_field

    def _influencefunc(
        self,
        params: dict[str, np.ndarray],
        r_field: np.ndarray,
        z_field: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        n_samples = int(self.config["N"])
        n_r = int(self.config["N_r"])
        n_z = int(self.config["N_z"])
        damp = float(self.config["damp"])
        component = int(self.config["component"])
        loadtype = int(self.config["loadtype"])
        bvptype = int(self.config["bvptype"])
        z_source = float(self.config["z_source"]) / float(self.config["r_source"])
        r_source = float(self.config["r_source"]) / float(self.config["r_source"])
        l_source = float(self.config["l_source"])

        wd = np.zeros((n_samples, n_r, n_z), dtype=np.complex128)

        start = time.perf_counter_ns()
        for i in tqdm(range(n_samples), desc="Computing Green functions", colour="green"):
            c11 = float(params["c11_over_c44"][i])
            c12 = float(params["c12_over_c44"][i])
            c13 = float(params["c13_over_c44"][i])
            c33 = float(params["c33_over_c44"][i])
            c44 = float(params["c44_over_c44"][i])
            rho = float(params["rho_over_rho0"][i])
            omega = float(params["omega_for_solver"][i])

            for j in range(n_r):
                rj = float(r_field[j])
                for k in range(n_z):
                    wd[i, j, k] = influence(
                        c11_val=c11,
                        c12_val=c12,
                        c13_val=c13,
                        c33_val=c33,
                        c44_val=c44,
                        dens_val=rho,
                        damp_val=damp,
                        r_campo_val=rj,
                        z_campo_val=float(z_field[k]),
                        z_fonte_val=z_source,
                        r_fonte_val=r_source,
                        l_fonte_val=l_source,
                        freq_val=omega,
                        bvptype_val=bvptype,
                        loadtype_val=loadtype,
                        component_val=component,
                    )

        duration_s = (time.perf_counter_ns() - start) / 1e9
        return wd, duration_s

    @staticmethod
    def _stats(arr: np.ndarray) -> dict[str, Any]:
        return {
            "shape": [int(i) for i in arr.shape],
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    def generate(self):
        # Fail fast before the integration loop if the native Rajapakse solver
        # cannot be loaded on this machine.
        _ensure_library_available()
        seed = int(self.config.get("seed", 42))
        rng = np.random.default_rng(seed)

        params = self._get_input_functions(rng=rng)
        r_field, z_field = self._get_coordinates()
        displacements, duration_s = self._influencefunc(
            params=params,
            r_field=r_field,
            z_field=z_field,
        )

        logger.info(
            "Runtime for integration: %.3f s | shapes: xb=%s, g_u=%s, r=%s, z=%s",
            duration_s,
            tuple(params["delta"].shape),
            tuple(displacements.shape),
            tuple(r_field.shape),
            tuple(z_field.shape),
        )

        metadata = {
            "runtime_s": float(duration_s),
            "runtime_ms": float(duration_s * 1e3),
            "integration_time_per_sample_s": float(duration_s / max(int(self.config["N"]), 1)),
            "seed": seed,
            "sampling_mode": str(params["sampling_mode"][0]),
            "formulation": {
                "notes": [
                    "c_ij are normalized by c44 (c44/c44 = 1).",
                    "Frequency variable is handled through nondimensional delta.",
                    "Solver receives omega_for_solver and rho_over_rho0 such that omega_for_solver*sqrt(rho_over_rho0)=delta.",
                ],
                "component": int(self.config["component"]),
                "loadtype": int(self.config["loadtype"]),
                "bvptype": int(self.config["bvptype"]),
            },
            "parameters": {
                key: self._stats(np.asarray(val, dtype=float))
                for key, val in params.items()
                if key not in {"sampling_mode"}
            },
            "coordinate_statistics": {
                "r": self._stats(r_field),
                "z": self._stats(z_field),
            },
            "displacement_statistics": {
                "g_u": {
                    "shape": [int(i) for i in displacements.shape],
                    "real_min": float(np.min(displacements.real)),
                    "real_max": float(np.max(displacements.real)),
                    "real_mean": float(np.mean(displacements.real)),
                    "real_std": float(np.std(displacements.real)),
                    "imag_min": float(np.min(displacements.imag)),
                    "imag_max": float(np.max(displacements.imag)),
                    "imag_mean": float(np.mean(displacements.imag)),
                    "imag_std": float(np.std(displacements.imag)),
                }
            },
        }

        path = Path(self.config["data_filename"])
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays_to_save: dict[str, np.ndarray] = {
            "r": r_field,
            "z": z_field,
            "g_u": displacements,
        }
        for key, val in params.items():
            arrays_to_save[key] = val

        np.savez(path, **arrays_to_save)

        metadata_path = path.with_suffix(".yaml")
        with open(metadata_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(metadata, file, sort_keys=False, allow_unicode=True)

        logger.info("Saved data at %s", path)
        logger.info("Saved metadata at %s", metadata_path)
