from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.problems.base_generator import BaseProblemGenerator

logger = logging.getLogger(__name__)


class MultilayerHorizontalRockingGenerator(BaseProblemGenerator):
    """Generate multilayer horizontal rocking samples by calling the legacy Fortran solver."""

    CHANNEL_FILENAMES = {
        "urfx": ("SAIDA_URFx_W.out", "SAIDA_URFx_.out"),
        "uzfx": ("SAIDA_UZFx_W.out", "SAIDA_UZFx_.out"),
        "urmy": ("SAIDA_URMy_W.out", "SAIDA_URMy_.out"),
        "uzmy": ("SAIDA_UZMy_W.out", "SAIDA_UZMy_.out"),
        "uzz": ("SAIDA_UZZ_W.out", "SAIDA_UZZ_.out"),
    }
    PAPER_BASELINE_MATERIALS = {
        # Table 2 from Labaki et al. (2014), using normalized c'ij with c44=1.
        "m1": {"c11": 3.0000, "c12": 1.0000, "c13": 1.0000, "c33": 3.0000, "c44": 1.0000, "eta": 0.01, "rho": 1.0},
        "m2": {"c11": 2.8284, "c12": 0.8284, "c13": 0.8284, "c33": 4.2426, "c44": 1.0000, "eta": 0.01, "rho": 1.0},
        "m3": {"c11": 2.7749, "c12": 0.7749, "c13": 0.7749, "c33": 5.5497, "c44": 1.0000, "eta": 0.01, "rho": 1.0},
    }
    PAPER_BASELINE_CASES = {
        # Table 1: two finite layers + half-space.
        "A": [("m1", 0.5), ("m1", 0.5), ("m1", 0.0)],
        "B": [("m3", 0.5), ("m2", 0.5), ("m1", 0.0)],
        "C": [("m3", 0.3), ("m2", 0.7), ("m1", 0.0)],
    }

    def __init__(self, config: str | dict[str, Any]):
        super().__init__(config=config)
        if isinstance(config, (str, Path)):
            self.config_path = Path(config)
            self.config = self.load_config()
        else:
            self.config_path = None
            self.config = config

    def load_config(self) -> dict[str, Any]:
        if self.config_path is None:
            return self.config
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _get_solver_path(self) -> Path:
        solver_path = self.config.get("executable_path")
        if solver_path is None:
            solver_path = Path(__file__).parent / "libs" / "multilayer.exe"
        path = Path(solver_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Solver executable not found: {path}")
        return path

    def _get_range(self, key: str) -> tuple[float, float]:
        sampling = self.config["sampling"]
        values = sampling[key]
        if len(values) != 2:
            raise ValueError(f"Expected two values in sampling range for '{key}', got: {values}")
        return float(values[0]), float(values[1])

    @staticmethod
    def _is_stable_transverse_isotropic(c11: float, c12: float, c13: float, c33: float, c44: float) -> bool:
        if c11 <= abs(c12) or c33 <= 0.0 or c44 <= 0.0:
            return False
        # Basic positive-definiteness condition for TI elastic constants.
        return (c11 + c12) * c33 > 2.0 * (c13**2)

    def _sample_properties(self, rng: np.random.Generator) -> np.ndarray:
        n_layers = int(self.config["N"])
        sampling = self.config["sampling"]
        max_tries = int(sampling.get("max_tries_per_layer", 1000))

        c11_min, c11_max = self._get_range("c11")
        c12_min, c12_max = self._get_range("c12")
        c13_min, c13_max = self._get_range("c13")
        c33_min, c33_max = self._get_range("c33")
        c44_min, c44_max = self._get_range("c44")
        eta_min, eta_max = self._get_range("eta")
        rho_min, rho_max = self._get_range("rho")
        h_min, h_max = self._get_range("h")

        props = np.zeros((n_layers + 1, 8), dtype=float)
        for layer in range(n_layers + 1):
            for _ in range(max_tries):
                c11 = rng.uniform(c11_min, c11_max)
                c33 = rng.uniform(c33_min, c33_max)
                c44 = rng.uniform(c44_min, c44_max)

                c12_upper = min(c12_max, 0.95 * c11)
                c12_lower = min(c12_min, c12_upper)
                c12 = rng.uniform(c12_lower, c12_upper)

                # Keep c13 inside stability envelope.
                c13_limit = np.sqrt(max(((c11 + c12) * c33) / 2.0, 1e-12)) * 0.95
                c13_upper = min(c13_max, c13_limit)
                c13_lower = min(c13_min, c13_upper)
                c13 = rng.uniform(c13_lower, c13_upper)

                if self._is_stable_transverse_isotropic(c11, c12, c13, c33, c44):
                    break
            else:
                raise RuntimeError(
                    "Unable to sample stable material constants; broaden ranges or increase max_tries_per_layer."
                )

            eta = rng.uniform(eta_min, eta_max)
            rho = rng.uniform(rho_min, rho_max)
            h = sampling.get("semi_space_h", 0.0) if layer == n_layers else rng.uniform(h_min, h_max)
            props[layer] = np.array([c11, c12, c13, c33, c44, eta, rho, h], dtype=float)

        return props

    def _sample_properties_paper_case(self, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        n_layers = int(self.config["N"])
        n_rows = n_layers + 1
        sampling = self.config["sampling"]

        requested_cases = sampling.get("paper_cases", ["A", "B", "C"])
        case_labels = [str(c).upper() for c in requested_cases]
        for case in case_labels:
            if case not in self.PAPER_BASELINE_CASES:
                raise ValueError(
                    f"Unknown paper case '{case}'. Valid values: {sorted(self.PAPER_BASELINE_CASES.keys())}"
                )

        probabilities = np.asarray(sampling.get("paper_case_probabilities", None), dtype=float)
        if probabilities.size == 0:
            probabilities = np.ones(len(case_labels), dtype=float) / len(case_labels)
        if probabilities.size != len(case_labels):
            raise ValueError(
                "paper_case_probabilities must have same length as paper_cases. "
                f"Got {probabilities.size} and {len(case_labels)}."
            )
        probabilities = probabilities / np.sum(probabilities)

        case_label = str(rng.choice(case_labels, p=probabilities))
        case_rows = self.PAPER_BASELINE_CASES[case_label]
        if n_rows != len(case_rows):
            raise ValueError(
                "paper_case sampling currently expects N=2 (2 layers + half-space). "
                f"Got N={n_layers}, which implies {n_rows} rows, but case '{case_label}' has {len(case_rows)} rows."
            )

        c_jitter_rel = float(sampling.get("paper_c_jitter_rel", 0.0))
        eta_jitter_abs = float(sampling.get("paper_eta_jitter_abs", 0.0))
        rho_jitter_abs = float(sampling.get("paper_rho_jitter_abs", 0.0))
        h_jitter_abs = float(sampling.get("paper_h_jitter_abs", 0.0))
        max_tries = int(sampling.get("max_tries_per_layer", 1000))

        for _ in range(max_tries):
            props = np.zeros((n_rows, 8), dtype=float)
            stable = True
            for row_idx, (mat_name, base_h) in enumerate(case_rows):
                mat = self.PAPER_BASELINE_MATERIALS[mat_name]

                c11 = float(mat["c11"]) * (1.0 + rng.uniform(-c_jitter_rel, c_jitter_rel))
                c12 = float(mat["c12"]) * (1.0 + rng.uniform(-c_jitter_rel, c_jitter_rel))
                c13 = float(mat["c13"]) * (1.0 + rng.uniform(-c_jitter_rel, c_jitter_rel))
                c33 = float(mat["c33"]) * (1.0 + rng.uniform(-c_jitter_rel, c_jitter_rel))
                c44 = float(mat["c44"]) * (1.0 + rng.uniform(-c_jitter_rel, c_jitter_rel))
                eta = float(mat["eta"]) + rng.uniform(-eta_jitter_abs, eta_jitter_abs)
                rho = float(mat["rho"]) + rng.uniform(-rho_jitter_abs, rho_jitter_abs)

                if row_idx == n_rows - 1:
                    h = 0.0
                else:
                    h = float(base_h) + rng.uniform(-h_jitter_abs, h_jitter_abs)

                if not self._is_stable_transverse_isotropic(c11, c12, c13, c33, c44):
                    stable = False
                    break
                if eta <= 0.0 or rho <= 0.0 or h < 0.0:
                    stable = False
                    break

                props[row_idx] = np.array([c11, c12, c13, c33, c44, eta, rho, h], dtype=float)

            if stable:
                return props, case_label

        raise RuntimeError(
            "Unable to sample stable paper-case properties; decrease paper jitters "
            "or increase max_tries_per_layer."
        )

    @staticmethod
    def _write_input_file(
        path: Path,
        omega: float,
        n_layers: int,
        nload: int,
        b_value: float,
        m_value: int,
        outputfilename: str,
        codes: list[int],
        properties: np.ndarray,
    ) -> None:
        with open(path, "w", encoding="utf-8") as f:
            # Single-frequency solve per sample.
            f.write(f"{omega:.16E} {omega:.16E} 1.0\n")
            f.write(f"{n_layers:d} {nload:d} {b_value:.16E} {m_value:d} {outputfilename}\n")
            f.write(" ".join(str(int(c)) for c in codes) + "\n")
            for row in properties:
                f.write(" ".join(f"{value:.16E}" for value in row) + "\n")

    @staticmethod
    def _parse_output_file(path: Path, m_value: int) -> dict[float, dict[int, np.ndarray]]:
        if not path.exists():
            raise FileNotFoundError(f"Solver output file not found: {path}")

        data: dict[float, dict[int, list[complex]]] = {}
        current_omega: float | None = None
        current_nrec: int | None = None
        saw_nrec_blocks = False

        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                upper = line.upper()
                if upper.startswith("OMEGA="):
                    current_omega = float(line.split("=", maxsplit=1)[1].strip())
                    data.setdefault(current_omega, {})
                    current_nrec = None
                    continue

                if upper.startswith("NREC="):
                    saw_nrec_blocks = True
                    if current_omega is None:
                        raise ValueError(f"Found Nrec block before OMEGA in file: {path}")
                    current_nrec = int(float(line.split("=", maxsplit=1)[1].strip()))
                    data[current_omega].setdefault(current_nrec, [])
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue
                value = complex(float(parts[0]), float(parts[1]))

                if current_omega is None:
                    raise ValueError(f"Found value row before OMEGA in file: {path}")
                if not saw_nrec_blocks:
                    current_nrec = 1
                    data[current_omega].setdefault(current_nrec, [])
                if current_nrec is None:
                    raise ValueError(f"Found value row before Nrec block in file: {path}")
                data[current_omega][current_nrec].append(value)

        parsed: dict[float, dict[int, np.ndarray]] = {}
        expected_values = m_value * m_value
        for omega, by_nrec in data.items():
            parsed[omega] = {}
            for nrec, values in by_nrec.items():
                if len(values) != expected_values:
                    raise ValueError(
                        f"Expected {expected_values} values for omega={omega}, nrec={nrec}, got {len(values)} "
                        f"in file {path}."
                    )
                parsed[omega][nrec] = np.asarray(values, dtype=np.complex128).reshape(m_value, m_value)
        return parsed

    @staticmethod
    def _first_existing(candidates: tuple[str, ...], workdir: Path) -> Path:
        for filename in candidates:
            file_path = workdir / filename
            if file_path.exists():
                return file_path
        raise FileNotFoundError(f"None of the expected output files were found in {workdir}: {candidates}")

    def _run_solver(self, omega: float, properties: np.ndarray) -> dict[str, np.ndarray]:
        solver_path = self._get_solver_path()
        n_layers = int(self.config["N"])
        nload = int(self.config.get("Nload", 1))
        b_value = float(self.config.get("B", 0.0))
        m_value = int(self.config["M"])
        timeout = int(self.config.get("solver_timeout_seconds", 300))
        outputfilename = str(self.config.get("outputfilename", "dontmatter"))
        codes = list(self.config.get("codes", [1, 1, 1, 1]))
        if any(int(c) != 1 for c in codes):
            raise ValueError(
                "This generator currently requires all CODE flags to be 1 because "
                "the Fortran driver writes all output channel files (including UZZ) "
                "using the legacy four CODE flags."
            )
        receiver_interface = int(self.config.get("receiver_interface", 1))

        with tempfile.TemporaryDirectory(prefix="multilayer_sample_") as temp_dir:
            workdir = Path(temp_dir)
            local_solver = workdir / solver_path.name
            shutil.copy2(solver_path, local_solver)
            local_solver.chmod(local_solver.stat().st_mode | 0o111)

            self._write_input_file(
                path=workdir / "INPUT.TXT",
                omega=omega,
                n_layers=n_layers,
                nload=nload,
                b_value=b_value,
                m_value=m_value,
                outputfilename=outputfilename,
                codes=codes,
                properties=properties,
            )

            result = subprocess.run(
                [f"./{local_solver.name}"],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Solver failed with return code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                )

            channel_data: dict[str, np.ndarray] = {}
            for channel, filenames in self.CHANNEL_FILENAMES.items():
                parsed = self._parse_output_file(self._first_existing(filenames, workdir), m_value=m_value)
                omega_key = min(parsed.keys(), key=lambda v: abs(v - omega))
                if receiver_interface not in parsed[omega_key]:
                    available = sorted(parsed[omega_key].keys())
                    raise ValueError(
                        f"Receiver interface {receiver_interface} not found for channel '{channel}'. "
                        f"Available interfaces: {available}"
                    )
                channel_data[channel] = parsed[omega_key][receiver_interface]

        return channel_data

    @staticmethod
    def _all_finite_complex(arr: np.ndarray) -> bool:
        return np.all(np.isfinite(arr.real)) and np.all(np.isfinite(arr.imag))

    def generate(self) -> None:
        seed = int(self.config.get("seed", 42))
        num_samples = int(self.config["num_samples"])
        max_attempts_per_sample = int(self.config.get("max_attempts_per_sample", 20))
        n_layers = int(self.config["N"])
        m_value = int(self.config["M"])
        b_value = float(self.config.get("B", 0.0))
        omega_min = float(self.config["omega"]["min"])
        omega_max = float(self.config["omega"]["max"])

        if not (0.0 <= b_value < 1.0):
            raise ValueError(f"B must satisfy 0 <= B < 1. Got: {b_value}")
        if not (1 <= self.config.get("receiver_interface", 1) <= n_layers + 1):
            raise ValueError(
                f"receiver_interface must be in [1, N+1]. Got {self.config.get('receiver_interface')} with N={n_layers}."
            )

        rng = np.random.default_rng(seed)
        omegas = rng.uniform(omega_min, omega_max, size=num_samples)

        xb_rows: list[np.ndarray] = []
        properties_samples: list[np.ndarray] = []
        g_u_samples: list[np.ndarray] = []
        sampled_case_labels: list[str] = []
        rejected_total = 0
        retries_per_sample: list[int] = []

        l_value = (1.0 - b_value) / m_value
        r_centers = b_value + l_value * np.arange(m_value) + l_value / 2.0
        s_centers = r_centers.copy()
        rr, ss = np.meshgrid(r_centers, s_centers, indexing="ij")
        xt = np.column_stack([rr.ravel(), ss.ravel()])

        sampling_mode = str(self.config.get("sampling", {}).get("mode", "random")).lower()
        if sampling_mode not in {"random", "paper_case"}:
            raise ValueError(f"Unknown sampling.mode '{sampling_mode}'. Valid values: 'random', 'paper_case'.")

        logger.info("Generating multilayer dataset with %d samples (mode=%s)...", num_samples, sampling_mode)

        for idx, omega in enumerate(omegas, start=1):
            sample_retries = 0
            while True:
                if sampling_mode == "paper_case":
                    props, case_label = self._sample_properties_paper_case(rng)
                else:
                    props = self._sample_properties(rng)
                    case_label = "random"
                channels = self._run_solver(float(omega), props)

                g_u = np.stack(
                    [
                        channels["urfx"].ravel(),
                        channels["uzfx"].ravel(),
                        channels["urmy"].ravel(),
                        channels["uzmy"].ravel(),
                        channels["uzz"].ravel(),
                    ],
                    axis=-1,
                )

                if self._all_finite_complex(g_u):
                    break

                sample_retries += 1
                rejected_total += 1
                if sample_retries >= max_attempts_per_sample:
                    raise RuntimeError(
                        f"Unable to generate finite solver outputs for sample {idx} "
                        f"(omega={omega:.6g}) after {max_attempts_per_sample} attempts."
                    )
                logger.warning(
                    "Rejected non-finite output for sample %d (omega=%.6g). Retrying (%d/%d).",
                    idx,
                    omega,
                    sample_retries,
                    max_attempts_per_sample,
                )

            xb = np.concatenate([props.ravel(), np.array([omega], dtype=float)])

            xb_rows.append(xb)
            properties_samples.append(props)
            g_u_samples.append(g_u)
            retries_per_sample.append(sample_retries)
            sampled_case_labels.append(case_label)

            if idx % max(1, num_samples // 10) == 0 or idx == num_samples:
                logger.info("Generated %d/%d samples.", idx, num_samples)

        xb_arr = np.asarray(xb_rows, dtype=float)
        props_arr = np.asarray(properties_samples, dtype=float)
        g_u_arr = np.asarray(g_u_samples, dtype=np.complex128)

        path = Path(self.config["data_filename"])
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            xb=xb_arr,
            xt=xt,
            g_u=g_u_arr,
            omega=omegas,
            properties=props_arr,
            r=r_centers,
            s=s_centers,
            paper_case_label=np.asarray(sampled_case_labels, dtype="U16"),
        )

        metadata = {
            "generator": "MultilayerHorizontalRockingGenerator",
            "num_samples": num_samples,
            "seed": seed,
            "solver_executable": str(self._get_solver_path()),
            "N_layers": n_layers,
            "M": m_value,
            "B": b_value,
            "receiver_interface": int(self.config.get("receiver_interface", 1)),
            "omega_range": [omega_min, omega_max],
            "xb_shape": list(xb_arr.shape),
            "xt_shape": list(xt.shape),
            "g_u_shape": list(g_u_arr.shape),
            "g_u_channels": ["urfx", "uzfx", "urmy", "uzmy", "uzz"],
            "properties_layout": [
                "c11",
                "c12",
                "c13",
                "c33",
                "c44",
                "eta",
                "rho",
                "h",
            ],
            "config_snapshot": self.config,
            "max_attempts_per_sample": max_attempts_per_sample,
            "rejected_non_finite_samples_total": rejected_total,
            "retries_per_sample": retries_per_sample,
            "sampling_mode": sampling_mode,
            "paper_case_counts": {
                str(label): int(np.sum(np.asarray(sampled_case_labels) == label))
                for label in sorted(set(sampled_case_labels))
            },
        }

        metadata_path = path.with_suffix(".yaml")
        with open(metadata_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(metadata, f, sort_keys=False, allow_unicode=True)

        logger.info("Saved data at %s", path)
        logger.info("Saved metadata at %s", metadata_path)
