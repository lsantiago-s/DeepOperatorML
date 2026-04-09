from __future__ import annotations

import errno
import logging
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.problems.base_generator import BaseProblemGenerator

logger = logging.getLogger(__name__)


class MultilayerHorizontalRockingGenerator(BaseProblemGenerator):
    """Generate bi-material full influence matrices by calling the legacy Fortran solver."""

    CHANNEL_FILENAMES = {
        "urfx": ("SAIDA_URFx_W.out", "SAIDA_URFx_.out"),
        "uzfx": ("SAIDA_UZFx_W.out", "SAIDA_UZFx_.out"),
        "urmy": ("SAIDA_URMy_W.out", "SAIDA_URMy_.out"),
        "uzmy": ("SAIDA_UZMy_W.out", "SAIDA_UZMy_.out"),
        "uzz": ("SAIDA_UZZ_W.out", "SAIDA_UZZ_.out"),
    }

    # Labaki et al. (2013), Table 2, normalized with c44 = 1 and rho = 1.
    PAPER_BASELINE_MATERIALS = {
        "A": {"c11": 3.0000, "c12": 1.0000, "c13": 1.0000, "c33": 3.0000, "c44": 1.0000, "eta": 0.01, "rho": 1.0},
        "B": {"c11": 2.7749, "c12": 0.7749, "c13": 0.7749, "c33": 5.5497, "c44": 1.0000, "eta": 0.01, "rho": 1.0},
        "C": {"c11": 2.7321, "c12": 0.7321, "c13": 0.7321, "c33": 8.1962, "c44": 1.0000, "eta": 0.01, "rho": 1.0},
        "D": {"c11": 2.7136, "c12": 0.7136, "c13": 0.7136, "c33": 10.8543, "c44": 1.0000, "eta": 0.01, "rho": 1.0},
    }

    # Paper-aligned bi-material cases:
    # - 4.3 anisotropy sweep: medium 1 fixed as A, medium 2 in {A,B,C}
    # - 4.4 damping study: medium 1 = D, medium 2 = B
    PAPER_BIMATERIAL_CASES = {
        "A": ("A", "A"),
        "B": ("A", "B"),
        "C": ("A", "C"),
        "D": ("A", "D"),
        "DB_DAMPING": ("D", "B"),
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

    @staticmethod
    def _default_solver_candidates() -> tuple[Path, ...]:
        libs_dir = Path(__file__).parent / "libs"
        system = platform.system().lower()

        if system == "linux":
            names = ("multilayer_linux.exe", "multilayer.exe", "HORROCK_190615.exe")
        elif system == "darwin":
            names = ("multilayer.exe", "multilayer_linux.exe", "HORROCK_190615.exe")
        elif system == "windows":
            names = ("HORROCK_190615.exe", "multilayer.exe", "multilayer_linux.exe")
        else:
            names = ("multilayer_linux.exe", "multilayer.exe", "HORROCK_190615.exe")

        return tuple(libs_dir / name for name in names)

    def _get_solver_path(self) -> Path:
        solver_path = self.config.get("executable_path")
        if solver_path not in (None, "", "auto"):
            path = Path(solver_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Solver executable not found: {path}")
            return path

        candidates = self._default_solver_candidates()
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        tried = ", ".join(str(path.resolve()) for path in candidates)
        raise FileNotFoundError(f"Solver executable not found. Tried: {tried}")

    @staticmethod
    def _is_stable_transverse_isotropic(c11: float, c12: float, c13: float, c33: float, c44: float) -> bool:
        if c11 <= abs(c12) or c33 <= 0.0 or c44 <= 0.0:
            return False
        return (c11 + c12) * c33 > 2.0 * (c13**2)

    def _get_medium_param_range(self, medium_cfg: dict[str, Any], fallback_cfg: dict[str, Any], key: str) -> tuple[float, float]:
        values = medium_cfg.get(key, fallback_cfg.get(key, None))
        if values is None:
            raise KeyError(f"Missing sampling range for '{key}'.")
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"Expected two values for sampling '{key}', got {values}")
        return float(values[0]), float(values[1])

    def _sample_medium(self, rng: np.random.Generator, medium_cfg: dict[str, Any], fallback_cfg: dict[str, Any], max_tries: int) -> np.ndarray:
        c11_min, c11_max = self._get_medium_param_range(medium_cfg, fallback_cfg, "c11")
        c12_min, c12_max = self._get_medium_param_range(medium_cfg, fallback_cfg, "c12")
        c13_min, c13_max = self._get_medium_param_range(medium_cfg, fallback_cfg, "c13")
        c33_min, c33_max = self._get_medium_param_range(medium_cfg, fallback_cfg, "c33")
        c44_min, c44_max = self._get_medium_param_range(medium_cfg, fallback_cfg, "c44")
        eta_min, eta_max = self._get_medium_param_range(medium_cfg, fallback_cfg, "eta")
        rho_min, rho_max = self._get_medium_param_range(medium_cfg, fallback_cfg, "rho")

        for _ in range(max_tries):
            c11 = rng.uniform(c11_min, c11_max)
            c33 = rng.uniform(c33_min, c33_max)
            c44 = rng.uniform(c44_min, c44_max)

            c12_upper = min(c12_max, 0.95 * c11)
            c12_lower = min(c12_min, c12_upper)
            c12 = rng.uniform(c12_lower, c12_upper)

            c13_limit = np.sqrt(max(((c11 + c12) * c33) / 2.0, 1e-12)) * 0.95
            c13_upper = min(c13_max, c13_limit)
            c13_lower = min(c13_min, c13_upper)
            c13 = rng.uniform(c13_lower, c13_upper)

            if not self._is_stable_transverse_isotropic(c11, c12, c13, c33, c44):
                continue

            eta = rng.uniform(eta_min, eta_max)
            rho = rng.uniform(rho_min, rho_max)
            if eta <= 0.0 or rho <= 0.0:
                continue
            return np.asarray([c11, c12, c13, c33, c44, eta, rho], dtype=float)

        raise RuntimeError("Unable to sample stable TI medium; adjust sampling ranges or max_tries_per_medium.")

    def _sample_bimaterial_random(self, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        sampling = self.config["sampling"]
        max_tries = int(sampling.get("max_tries_per_medium", 1000))
        medium1_cfg = dict(sampling.get("medium1", {}))
        medium2_cfg = dict(sampling.get("medium2", {}))

        p1 = self._sample_medium(rng=rng, medium_cfg=medium1_cfg, fallback_cfg=sampling, max_tries=max_tries)
        p2 = self._sample_medium(rng=rng, medium_cfg=medium2_cfg, fallback_cfg=sampling, max_tries=max_tries)

        h_top = float(sampling.get("medium1_h", 1.0))
        props = np.zeros((2, 8), dtype=float)
        props[0, :7] = p1
        props[0, 7] = h_top
        props[1, :7] = p2
        props[1, 7] = 0.0
        return props, "random"

    def _sample_bimaterial_paper_case(self, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        sampling = self.config["sampling"]
        requested_cases = [str(c).upper() for c in sampling.get("paper_cases", ["A", "B", "C"])]
        for case in requested_cases:
            if case not in self.PAPER_BIMATERIAL_CASES:
                raise ValueError(f"Unknown paper case '{case}'. Valid: {sorted(self.PAPER_BIMATERIAL_CASES)}")

        probabilities = np.asarray(sampling.get("paper_case_probabilities", None), dtype=float)
        if probabilities.size == 0:
            probabilities = np.ones(len(requested_cases), dtype=float) / len(requested_cases)
        if probabilities.size != len(requested_cases):
            raise ValueError("paper_case_probabilities must match paper_cases length.")
        probabilities = probabilities / np.sum(probabilities)

        case_label = str(rng.choice(requested_cases, p=probabilities))
        m1_name, m2_name = self.PAPER_BIMATERIAL_CASES[case_label]
        m1 = self.PAPER_BASELINE_MATERIALS[m1_name]
        m2 = self.PAPER_BASELINE_MATERIALS[m2_name]

        c_jitter_rel = float(sampling.get("paper_c_jitter_rel", 0.0))
        eta_jitter_abs = float(sampling.get("paper_eta_jitter_abs", 0.0))
        rho_jitter_abs = float(sampling.get("paper_rho_jitter_abs", 0.0))
        shared_eta_jitter = bool(sampling.get("paper_shared_eta_jitter", False))
        h_top = float(sampling.get("medium1_h", 1.0))

        eta_shift = rng.uniform(-eta_jitter_abs, eta_jitter_abs) if shared_eta_jitter else None

        def _jitter_material(mat: dict[str, float], eta_shift_val: float | None = None) -> np.ndarray:
            c11 = float(mat["c11"]) * (1.0 + rng.uniform(-c_jitter_rel, c_jitter_rel))
            c12 = float(mat["c12"]) * (1.0 + rng.uniform(-c_jitter_rel, c_jitter_rel))
            c13 = float(mat["c13"]) * (1.0 + rng.uniform(-c_jitter_rel, c_jitter_rel))
            c33 = float(mat["c33"]) * (1.0 + rng.uniform(-c_jitter_rel, c_jitter_rel))
            c44 = float(mat["c44"]) * (1.0 + rng.uniform(-c_jitter_rel, c_jitter_rel))
            eta = float(mat["eta"]) + (
                float(eta_shift_val) if eta_shift_val is not None else rng.uniform(-eta_jitter_abs, eta_jitter_abs)
            )
            rho = float(mat["rho"]) + rng.uniform(-rho_jitter_abs, rho_jitter_abs)
            if not self._is_stable_transverse_isotropic(c11, c12, c13, c33, c44):
                raise RuntimeError("Unstable paper-case material after jitter.")
            if eta <= 0.0 or rho <= 0.0:
                raise RuntimeError("Invalid eta/rho after jitter.")
            return np.asarray([c11, c12, c13, c33, c44, eta, rho], dtype=float)

        p1 = _jitter_material(m1, eta_shift_val=eta_shift)
        p2 = _jitter_material(m2, eta_shift_val=eta_shift)

        props = np.zeros((2, 8), dtype=float)
        props[0, :7] = p1
        props[0, 7] = h_top
        props[1, :7] = p2
        props[1, 7] = 0.0
        return props, case_label

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
                        f"Expected {expected_values} values for omega={omega}, nrec={nrec}, got {len(values)} in {path}."
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
            raise ValueError("This generator requires all CODE flags = 1 for full channel output parsing.")
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

            try:
                result = subprocess.run(
                    [f"./{local_solver.name}"],
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except OSError as exc:
                if exc.errno == errno.ENOEXEC:
                    raise RuntimeError(
                        "Solver executable format is incompatible with this platform. "
                        f"Selected solver: {solver_path} | platform={platform.system()}. "
                        "Use executable_path: auto with a native build in libs/, "
                        "or point executable_path to a platform-compatible solver binary."
                    ) from exc
                raise
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

    def _compute_omega_from_a0(self, a0: float, medium1: np.ndarray) -> float:
        # medium1 = [c11, c12, c13, c33, c44, eta, rho]
        c44 = float(medium1[4])
        rho = float(medium1[6])
        a_ref = float(self.config.get("a_ref", 1.0))
        c_s1 = np.sqrt(c44 / max(rho, 1e-14))
        return float(a0 * c_s1 / max(a_ref, 1e-14))

    def _assemble_full_matrix(self, channels: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        mapping = self.config.get(
            "block_channel_map",
            {"uxx": "urfx", "uzx": "uzfx", "uxz": "urmy", "uzz": "uzmy"},
        )
        required = ["uxx", "uzx", "uxz", "uzz"]
        for key in required:
            if key not in mapping:
                raise KeyError(f"block_channel_map missing key '{key}'.")
            if mapping[key] not in channels:
                raise KeyError(f"Channel '{mapping[key]}' required for block '{key}' not available.")

        uxx = channels[mapping["uxx"]]
        uzx = channels[mapping["uzx"]]
        uxz = channels[mapping["uxz"]]
        uzz = channels[mapping["uzz"]]

        full = np.block([[uxx, uxz], [uzx, uzz]])
        blocks = {"Uxx": uxx, "Uxz": uxz, "Uzx": uzx, "Uzz": uzz}
        return full, blocks

    def _sample_a0(self, rng: np.random.Generator) -> float:
        a0_cfg = self.config.get("a0", None)
        if a0_cfg is None:
            omega_cfg = self.config.get("omega", None)
            if omega_cfg is None:
                raise KeyError("Config must provide either 'a0' range or legacy 'omega' range.")
            return float(rng.uniform(float(omega_cfg["min"]), float(omega_cfg["max"])))
        return float(rng.uniform(float(a0_cfg["min"]), float(a0_cfg["max"])))

    def generate(self) -> None:
        start = time.perf_counter()
        seed = int(self.config.get("seed", 42))
        num_samples = int(self.config["num_samples"])
        max_attempts_per_sample = int(self.config.get("max_attempts_per_sample", 20))
        n_layers = int(self.config["N"])
        m_value = int(self.config["M"])
        b_value = float(self.config.get("B", 0.0))

        if n_layers != 1:
            raise ValueError(
                "This formulation requires N=1 (bi-material: medium 1 + medium 2 half-space)."
            )
        if not (0.0 <= b_value < 1.0):
            raise ValueError(f"B must satisfy 0 <= B < 1. Got: {b_value}")

        rng = np.random.default_rng(seed)

        xb_rows: list[np.ndarray] = []
        properties_samples: list[np.ndarray] = []
        g_u_samples: list[np.ndarray] = []
        block_samples: list[np.ndarray] = []
        a0_samples: list[float] = []
        omega_samples: list[float] = []
        sampled_case_labels: list[str] = []
        rejected_total = 0
        retries_per_sample: list[int] = []

        # Full influence matrix coordinates over (receiver dof, source dof) in normalized index space.
        n_dof = 2 * m_value
        dof_axis = np.arange(n_dof, dtype=float)
        if n_dof > 1:
            dof_axis = dof_axis / (n_dof - 1)
        ii, jj = np.meshgrid(dof_axis, dof_axis, indexing="ij")
        xt = np.column_stack([ii.ravel(), jj.ravel()])

        sampling_mode = str(self.config.get("sampling", {}).get("mode", "random")).lower()
        if sampling_mode not in {"random", "paper_case"}:
            raise ValueError(f"Unknown sampling.mode '{sampling_mode}'. Valid: random, paper_case")

        logger.info(
            "Generating bi-material full-matrix dataset with %d samples (mode=%s, input_dim=15)...",
            num_samples,
            sampling_mode,
        )

        for idx in range(1, num_samples + 1):
            sample_retries = 0
            while True:
                if sampling_mode == "paper_case":
                    props, case_label = self._sample_bimaterial_paper_case(rng)
                else:
                    props, case_label = self._sample_bimaterial_random(rng)

                medium1 = props[0, :7]
                medium2 = props[1, :7]
                a0_val = self._sample_a0(rng)
                omega_val = self._compute_omega_from_a0(a0=a0_val, medium1=medium1)
                attempt_idx = sample_retries + 1

                logger.info(
                    "Sample %d/%d: running solver (attempt %d, case=%s, a0=%.6g, omega=%.6g).",
                    idx,
                    num_samples,
                    attempt_idx,
                    case_label,
                    a0_val,
                    omega_val,
                )
                sample_start = time.perf_counter()

                channels = self._run_solver(float(omega_val), props)
                u_full, blocks = self._assemble_full_matrix(channels=channels)
                sample_duration = time.perf_counter() - sample_start

                if self._all_finite_complex(u_full):
                    logger.info(
                        "Sample %d/%d: solver finished in %.2f s.",
                        idx,
                        num_samples,
                        sample_duration,
                    )
                    break

                sample_retries += 1
                rejected_total += 1
                logger.warning(
                    "Sample %d/%d: solver returned non-finite output after %.2f s.",
                    idx,
                    num_samples,
                    sample_duration,
                )
                if sample_retries >= max_attempts_per_sample:
                    raise RuntimeError(
                        f"Unable to generate finite output for sample {idx} after {max_attempts_per_sample} attempts."
                    )
                logger.warning(
                    "Rejected non-finite output for sample %d (a0=%.6g, omega=%.6g). Retrying (%d/%d).",
                    idx,
                    a0_val,
                    omega_val,
                    sample_retries,
                    max_attempts_per_sample,
                )

            xb = np.concatenate([medium1, medium2, np.array([a0_val], dtype=float)])

            xb_rows.append(xb)
            properties_samples.append(props)
            g_u_samples.append(u_full.ravel())
            block_samples.append(np.stack([blocks["Uxx"], blocks["Uxz"], blocks["Uzx"], blocks["Uzz"]], axis=-1))
            a0_samples.append(a0_val)
            omega_samples.append(omega_val)
            retries_per_sample.append(sample_retries)
            sampled_case_labels.append(case_label)

            if idx % max(1, num_samples // 10) == 0 or idx == num_samples:
                logger.info("Generated %d/%d samples.", idx, num_samples)

        xb_arr = np.asarray(xb_rows, dtype=float)
        props_arr = np.asarray(properties_samples, dtype=float)
        g_u_arr = np.asarray(g_u_samples, dtype=np.complex128)
        g_blocks_arr = np.asarray(block_samples, dtype=np.complex128)
        a0_arr = np.asarray(a0_samples, dtype=float)
        omega_arr = np.asarray(omega_samples, dtype=float)

        path = Path(self.config["data_filename"])
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            xb=xb_arr,
            xt=xt,
            g_u=g_u_arr,
            g_u_blocks=g_blocks_arr,
            a0=a0_arr,
            omega=omega_arr,
            properties=props_arr,
            paper_case_label=np.asarray(sampled_case_labels, dtype="U16"),
            block_names=np.asarray(["Uxx", "Uxz", "Uzx", "Uzz"], dtype="U8"),
        )

        duration = time.perf_counter() - start
        metadata = {
            "generator": "MultilayerHorizontalRockingGenerator",
            "formulation": "bimaterial_full_matrix",
            "runtime_s": float(duration),
            "runtime_ms": float(duration * 1e3),
            "timing_breakdown": {
                "direct_solver_total_s": float(duration),
                "direct_solver_per_sample_s": float(duration / max(num_samples, 1)),
                "solver_kind": "legacy_fortran_multilayer_bimaterial",
            },
            "num_samples": num_samples,
            "seed": seed,
            "solver_executable": str(self._get_solver_path()),
            "N_layers": n_layers,
            "M": m_value,
            "B": b_value,
            "input_dim": 15,
            "input_layout": [
                "c11_1",
                "c12_1",
                "c13_1",
                "c33_1",
                "c44_1",
                "eta_1",
                "rho_1",
                "c11_2",
                "c12_2",
                "c13_2",
                "c33_2",
                "c44_2",
                "eta_2",
                "rho_2",
                "a0",
            ],
            "output_layout": {
                "matrix_shape": [2 * m_value, 2 * m_value],
                "blocks": ["Uxx", "Uxz", "Uzx", "Uzz"],
                "block_channel_map": self.config.get(
                    "block_channel_map",
                    {"uxx": "urfx", "uzx": "uzfx", "uxz": "urmy", "uzz": "uzmy"},
                ),
            },
            "a0_range": [float(np.min(a0_arr)), float(np.max(a0_arr))],
            "omega_range": [float(np.min(omega_arr)), float(np.max(omega_arr))],
            "xb_shape": list(xb_arr.shape),
            "xt_shape": list(xt.shape),
            "g_u_shape": list(g_u_arr.shape),
            "g_u_blocks_shape": list(g_blocks_arr.shape),
            "properties_layout": ["c11", "c12", "c13", "c33", "c44", "eta", "rho", "h"],
            "config_snapshot": self.config,
            "max_attempts_per_sample": max_attempts_per_sample,
            "rejected_non_finite_samples_total": rejected_total,
            "retries_per_sample": retries_per_sample,
            "sampling_mode": sampling_mode,
            "paper_case_counts": {
                str(label): int(np.sum(np.asarray(sampled_case_labels) == label))
                for label in sorted(set(sampled_case_labels))
            },
            "paper_case_definitions": self.PAPER_BIMATERIAL_CASES,
            "paper_material_table": self.PAPER_BASELINE_MATERIALS,
            "paper_reference": {
                "labaki_2013_reported_time_s_for_Uzz_M20": 310.0,
                "note": "Reported reference is for filling Uzz only; this dataset learns full coupled U matrix.",
            },
        }

        metadata_path = path.with_suffix(".yaml")
        with open(metadata_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(metadata, f, sort_keys=False, allow_unicode=True)

        logger.info("Saved data at %s", path)
        logger.info("Saved metadata at %s", metadata_path)
