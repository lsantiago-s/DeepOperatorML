#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec


def lhs_sample(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros((n, d), dtype=float)
    for i in range(d):
        intervals = np.arange(n, dtype=float) / n
        jitter = rng.random(n) / n
        col = intervals + jitter
        x[:, i] = col[rng.permutation(n)]
    return x


def sample_parameters(
    n_samples: int,
    rng_seed: int,
    c44_range: tuple[float, float],
    c11_ratio_range: tuple[float, float],
    c33_ratio_range: tuple[float, float],
    c13_ratio_range: tuple[float, float],
    rho_range: tuple[float, float],
    eta_range: tuple[float, float],
    a0_range: tuple[float, float],
    source_half_width: float,
    freq_range_hz: tuple[float, float] | None,
) -> np.ndarray:
    """Sample homogeneous TI parameters and convert to (a0, omega).

    Output layout:
    [c11, c13, c33, c44, rho, eta, a0, omega]
    """
    rng = np.random.default_rng(rng_seed)
    lhs = lhs_sample(n_samples, 7, rng)
    params = np.zeros((n_samples, 8), dtype=float)

    for i in range(n_samples):
        c44 = c44_range[0] + lhs[i, 0] * (c44_range[1] - c44_range[0])
        c11 = c44 * (c11_ratio_range[0] + lhs[i, 1] * (c11_ratio_range[1] - c11_ratio_range[0]))
        c33 = c44 * (c33_ratio_range[0] + lhs[i, 2] * (c33_ratio_range[1] - c33_ratio_range[0]))

        max_c13_ratio = (math.sqrt(max(c11 * c33, 1e-14)) - c44) / max(c44, 1e-14)
        sampled_c13_ratio = c13_ratio_range[0] + lhs[i, 3] * (c13_ratio_range[1] - c13_ratio_range[0])
        c13_ratio = min(sampled_c13_ratio, max_c13_ratio * 0.95)
        c13_ratio = max(c13_ratio, 0.0)
        c13 = c44 * c13_ratio

        rho = rho_range[0] + lhs[i, 4] * (rho_range[1] - rho_range[0])
        eta = eta_range[0] + lhs[i, 5] * (eta_range[1] - eta_range[0])
        c_s = math.sqrt(max(c44 / max(rho, 1e-14), 1e-14))

        if freq_range_hz is None:
            a0 = a0_range[0] + lhs[i, 6] * (a0_range[1] - a0_range[0])
            omega = a0 * c_s / max(source_half_width, 1e-14)
        else:
            freq_hz = freq_range_hz[0] + lhs[i, 6] * (freq_range_hz[1] - freq_range_hz[0])
            omega = 2.0 * math.pi * freq_hz
            a0 = omega * source_half_width / max(c_s, 1e-14)

        params[i, :] = [c11, c13, c33, c44, rho, eta, a0, omega]

    return params


def _safe_zeta(zeta: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(zeta, dtype=np.complex128)
    return np.where(np.abs(arr) < 1e-12, 1e-12 + 0j, arr)


def influencefunctions(
    lty: int,
    rho: float,
    omega: float,
    eta: float,
    c11: float,
    c13: float,
    c33: float,
    c44: float,
    halflength: float,
    x_field: float,
    x_source: float,
    z_field: float,
    z_source: float,
    abs_tol: float,
) -> np.ndarray:
    """Rajapakse-style strip-load influence kernels integrated over zeta."""
    if c11 == c33:
        c11 = c11 + 1e-5

    z = abs(z_source - z_field)
    alpha0 = c33 / c44
    beta0 = c11 / c44
    kappa0 = (c13 + c44) / c44
    g = 1.0 + (alpha0 * beta0) - (kappa0**2)
    delta = np.sqrt((rho * (omega**2)) / c44)
    k = abs(x_source - x_field)
    j = 1j

    def phi(zeta: np.ndarray) -> np.ndarray:
        return (((g * (zeta**2)) - 1.0 - alpha0) ** 2) - (
            4.0 * alpha0 * ((beta0 * zeta**4) - ((1.0 + beta0) * zeta**2) + 1.0)
        )

    def xi1(zeta: np.ndarray) -> np.ndarray:
        return np.sqrt((g * zeta**2) - 1.0 - alpha0 + np.sqrt(phi(zeta))) / np.sqrt(2.0 * alpha0)

    def xi2(zeta: np.ndarray) -> np.ndarray:
        return np.sqrt((g * zeta**2) - 1.0 - alpha0 - np.sqrt(phi(zeta))) / np.sqrt(2.0 * alpha0)

    # Viscoelastic correspondence: c_ij* = c_ij (1 + i eta)
    alpha = alpha0 * (1.0 + (eta * j))
    beta = beta0 * (1.0 + (eta * j))
    kappa = kappa0 * (1.0 + (eta * j))

    def kernel_bundle(zeta_raw: np.ndarray | float) -> np.ndarray:
        zeta = _safe_zeta(zeta_raw)
        xi1v = xi1(zeta)
        xi2v = xi2(zeta)
        xi1_safe = np.where(np.abs(xi1v) < 1e-12, 1e-12 + 0j, xi1v)
        xi2_safe = np.where(np.abs(xi2v) < 1e-12, 1e-12 + 0j, xi2v)
        omega1v = ((alpha * xi1v**2) - (zeta**2) + 1.0) / (j * kappa * zeta * xi1_safe)
        omega2v = ((alpha * xi2v**2) - (zeta**2) + 1.0) / (j * kappa * zeta * xi2_safe)
        eta3 = -(xi1v * omega1v) + (j * zeta)
        eta4 = -(xi2v * omega2v) + (j * zeta)
        eta5 = ((kappa - 1.0) * j * zeta * omega1v) - (alpha * xi1v)
        eta6 = ((kappa - 1.0) * j * zeta * omega2v) - (alpha * xi2v)

        denom = np.sin(delta * zeta * halflength)
        denom = np.where(np.abs(denom) < 1e-12, 1e-12 + 0j, denom)
        rc = zeta * ((eta3 * eta6) - (eta4 * eta5)) / denom
        rc = np.where(np.abs(rc) < 1e-12, 1e-12 + 0j, rc)

        exp1 = np.exp(-delta * xi1v * z)
        exp2 = np.exp(-delta * xi2v * z)
        cos_term = np.cos(delta * zeta * k)
        sin_term = np.sin(delta * zeta * k)

        fun1 = ((eta6 * omega1v * exp1) - (eta5 * omega2v * exp2)) * cos_term / rc
        fun3 = ((eta6 * exp1) - (eta5 * exp2)) * sin_term / rc
        fun4 = ((eta4 * exp1) - (eta3 * exp2)) * cos_term / rc
        fun1 = np.where(np.isfinite(fun1), fun1, 0.0 + 0.0j)
        fun3 = np.where(np.isfinite(fun3), fun3, 0.0 + 0.0j)
        fun4 = np.where(np.isfinite(fun4), fun4, 0.0 + 0.0j)

        if lty == 0:
            return np.vstack(
                [
                    np.real(fun1),
                    np.imag(fun1),
                    np.real(fun3),
                    np.imag(fun3),
                    np.real(fun4),
                    np.imag(fun4),
                ]
            )
        raise NotImplementedError(f"Only lty=0 is implemented, got lty={lty}.")

    integrals, _ = quad_vec(kernel_bundle, 0.0, np.inf, epsabs=abs_tol)
    integrals = np.asarray(integrals).reshape(-1)
    fun1_int = integrals[0] + 1j * integrals[1]
    fun3_int = integrals[2] + 1j * integrals[3]
    fun4_int = integrals[4] + 1j * integrals[5]

    uxx = ((-2.0) / (math.pi * c44 * delta)) * fun1_int
    uzx = ((-2.0j) / (math.pi * c44 * delta)) * fun3_int
    uxz = -uzx
    uzz = (2.0 / (math.pi * c44 * delta)) * fun4_int

    ui = np.array([[uxx, uxz], [uzx, uzz]], dtype=np.complex128)
    t = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return t.T @ ui @ t


def compute_sample(
    sample_idx: int,
    params_row: np.ndarray,
    x_positions: np.ndarray,
    source_half_width: float,
    lty: int,
    z_field: float,
    z_source: float,
    abs_tol: float,
) -> tuple[int, np.ndarray, float]:
    start = time.perf_counter()
    n_points = x_positions.shape[0]
    g_full = np.zeros((2 * n_points, 2 * n_points), dtype=np.complex128)

    c11, c13, c33, c44, rho, eta, _, omega = [float(v) for v in params_row]

    for j in range(n_points):
        x_source = float(x_positions[j])
        for i in range(j + 1):
            x_field = float(x_positions[i])
            uc = influencefunctions(
                lty=lty,
                rho=rho,
                omega=omega,
                eta=eta,
                c11=c11,
                c13=c13,
                c33=c33,
                c44=c44,
                halflength=source_half_width,
                x_field=x_field,
                x_source=x_source,
                z_field=z_field,
                z_source=z_source,
                abs_tol=abs_tol,
            )
            row_idx = slice(2 * i, 2 * i + 2)
            col_idx = slice(2 * j, 2 * j + 2)
            g_full[row_idx, col_idx] = uc
            if i != j:
                g_full[col_idx, row_idx] = uc.T

    elapsed = time.perf_counter() - start
    return sample_idx, g_full, elapsed


def compute_sample_star(args: tuple) -> tuple[int, np.ndarray, float]:
    return compute_sample(*args)


def write_dataset_bundle(
    out_dir: Path,
    g_samples: np.ndarray,
    params_array: np.ndarray,
    metadata: dict,
    sample_times: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    flat_real = np.stack([g.real.reshape(-1, order="F") for g in g_samples], axis=0)
    flat_imag = np.stack([g.imag.reshape(-1, order="F") for g in g_samples], axis=0)

    np.savetxt(out_dir / "G_samples_real_flat.csv", flat_real, delimiter=",")
    np.savetxt(out_dir / "G_samples_imag_flat.csv", flat_imag, delimiter=",")
    np.savetxt(out_dir / "params_array.csv", params_array, delimiter=",")
    np.savetxt(out_dir / "sample_times.csv", sample_times.reshape(-1, 1), delimiter=",")

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _sample_file(path: Path, sample_idx: int) -> Path:
    return path / f"sample_{sample_idx:05d}.npz"


def _signature(args: argparse.Namespace) -> dict[str, object]:
    return {
        "n_samples": int(args.n_samples),
        "n_points": int(args.n_points),
        "half_span": float(args.half_span),
        "lty": int(args.lty),
        "z_source": float(args.z_source),
        "z_field": float(args.z_field),
        "seed": int(args.seed),
        "abs_tol": float(args.abs_tol),
        "c44_min": float(args.c44_min),
        "c44_max": float(args.c44_max),
        "c11_ratio_min": float(args.c11_ratio_min),
        "c11_ratio_max": float(args.c11_ratio_max),
        "c33_ratio_min": float(args.c33_ratio_min),
        "c33_ratio_max": float(args.c33_ratio_max),
        "c13_ratio_min": float(args.c13_ratio_min),
        "c13_ratio_max": float(args.c13_ratio_max),
        "rho_min": float(args.rho_min),
        "rho_max": float(args.rho_max),
        "eta_min": float(args.eta_min),
        "eta_max": float(args.eta_max),
        "a0_min": float(args.a0_min),
        "a0_max": float(args.a0_max),
        "freq_min": None if args.freq_min is None else float(args.freq_min),
        "freq_max": None if args.freq_max is None else float(args.freq_max),
    }


def _signature_hash(signature: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _load_completed_samples(
    partials_dir: Path,
    g_samples: np.ndarray,
    sample_times: np.ndarray,
) -> np.ndarray:
    completed = np.zeros((g_samples.shape[0],), dtype=bool)
    for sample_path in sorted(partials_dir.glob("sample_*.npz")):
        with np.load(sample_path) as data:
            idx = int(data["sample_idx"])
            g_samples[idx] = data["g_full"]
            sample_times[idx] = float(data["elapsed"])
            completed[idx] = True
    return completed


def _write_sample_checkpoint(partials_dir: Path, sample_idx: int, g_full: np.ndarray, elapsed: float) -> None:
    sample_path = _sample_file(partials_dir, sample_idx)
    tmp_path = partials_dir / f".sample_{sample_idx:05d}.tmp.npz"
    np.savez(tmp_path, sample_idx=sample_idx, elapsed=elapsed, g_full=g_full)
    os.replace(tmp_path, sample_path)


def _print_progress(
    *,
    completed_count: int,
    total_samples: int,
    started_at: float,
    sample_times: np.ndarray,
    resumed_count: int,
    max_workers: int,
) -> None:
    elapsed_wall = time.perf_counter() - started_at
    done_times = sample_times[:completed_count]
    mean_sample = float(np.mean(done_times)) if completed_count else 0.0
    remaining = max(total_samples - completed_count, 0)
    parallel_efficiency = max(max_workers, 1)
    eta_seconds = (remaining * mean_sample) / parallel_efficiency if completed_count else float("nan")
    eta_msg = f"{eta_seconds / 3600:.2f} h" if np.isfinite(eta_seconds) else "unknown"
    print(
        (
            f"[progress] {completed_count}/{total_samples} samples "
            f"({completed_count / max(total_samples, 1):.1%}) | "
            f"wall={elapsed_wall / 3600:.2f} h | "
            f"mean_sample={mean_sample:.1f} s | "
            f"eta~{eta_msg} | resumed={resumed_count} | workers={max_workers}"
        ),
        flush=True,
    )


def _write_progress_file(
    *,
    progress_path: Path,
    completed_count: int,
    total_samples: int,
    started_at: float,
    sample_times: np.ndarray,
    resumed_count: int,
    max_workers: int,
    signature_hash: str,
    status: str,
) -> None:
    elapsed_wall = time.perf_counter() - started_at
    done_times = sample_times[:completed_count]
    mean_sample = float(np.mean(done_times)) if completed_count else None
    remaining = max(total_samples - completed_count, 0)
    eta_seconds = None
    if completed_count and mean_sample is not None:
        eta_seconds = float((remaining * mean_sample) / max(max_workers, 1))
    _write_json_atomic(
        progress_path,
        {
            "status": status,
            "signature_hash": signature_hash,
            "completed_samples": completed_count,
            "total_samples": total_samples,
            "fraction_complete": completed_count / max(total_samples, 1),
            "elapsed_wall_s": float(elapsed_wall),
            "mean_sample_s": mean_sample,
            "eta_s": eta_seconds,
            "resumed_samples": resumed_count,
            "max_workers": int(max_workers),
            "updated_at_epoch_s": time.time(),
        },
    )


def _final_bundle_exists(out_dir: Path) -> bool:
    required = (
        "G_samples_real_flat.csv",
        "G_samples_imag_flat.csv",
        "params_array.csv",
        "sample_times.csv",
        "metadata.json",
    )
    return all((out_dir / name).exists() for name in required)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ground-vibration external CSV bundle.")
    parser.add_argument(
        "--out-dir",
        default="./data/raw/ground_vibration/influence_dataset_N100_samples100_csv",
        help="Directory where G_samples_*.csv, params_array.csv, and metadata.json will be written.",
    )
    parser.add_argument("--n-samples", type=int, default=100, help="Number of parameter samples.")
    parser.add_argument("--n-points", type=int, default=100, help="Number of boundary points along x.")
    parser.add_argument("--half-span", type=float, default=2.0, help="Half-length L of the surface interval [-L, L].")
    parser.add_argument("--lty", type=int, default=0, help="Load type selector from the original MATLAB script.")
    parser.add_argument("--z-source", type=float, default=0.0)
    parser.add_argument("--z-field", type=float, default=0.0)
    parser.add_argument(
        "--damping",
        type=float,
        default=None,
        help="Deprecated alias for fixed eta. If set, eta_min=eta_max=damping.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--abs-tol", type=float, default=5e-5, help="Absolute tolerance for quadrature.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Parallel workers over samples. Use 0 to auto-detect from SLURM_CPUS_PER_TASK/os.cpu_count().",
    )
    parser.add_argument("--c44-min", type=float, default=10e9)
    parser.add_argument("--c44-max", type=float, default=30e9)
    parser.add_argument("--c11-ratio-min", type=float, default=1.5)
    parser.add_argument("--c11-ratio-max", type=float, default=4.0)
    parser.add_argument("--c33-ratio-min", type=float, default=1.2)
    parser.add_argument("--c33-ratio-max", type=float, default=3.5)
    parser.add_argument("--c13-ratio-min", type=float, default=0.5)
    parser.add_argument("--c13-ratio-max", type=float, default=1.8)
    parser.add_argument("--rho-min", type=float, default=2000.0)
    parser.add_argument("--rho-max", type=float, default=3000.0)
    parser.add_argument("--eta-min", type=float, default=0.005)
    parser.add_argument("--eta-max", type=float, default=0.05)
    parser.add_argument("--a0-min", type=float, default=0.05)
    parser.add_argument("--a0-max", type=float, default=4.0)
    parser.add_argument(
        "--freq-min",
        type=float,
        default=None,
        help="Optional frequency lower bound in Hz. If set with --freq-max, sampling is done in frequency space.",
    )
    parser.add_argument(
        "--freq-max",
        type=float,
        default=None,
        help="Optional frequency upper bound in Hz. If set with --freq-min, sampling is done in frequency space.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Print and checkpoint progress every N completed samples.",
    )
    args = parser.parse_args()

    if (args.freq_min is None) != (args.freq_max is None):
        raise ValueError("Provide both --freq-min and --freq-max, or neither.")
    if args.a0_min <= 0.0 or args.a0_max <= 0.0:
        raise ValueError("a0 bounds must be positive.")
    if args.eta_min < 0.0 or args.eta_max < 0.0:
        raise ValueError("eta bounds must be non-negative.")

    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    partials_dir = out_dir / "partials"
    progress_path = out_dir / "progress.json"
    signature_path = out_dir / "checkpoint_signature.json"

    if _final_bundle_exists(out_dir):
        print(f"Final dataset bundle already exists at {out_dir}; nothing to do.", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    partials_dir.mkdir(parents=True, exist_ok=True)

    if args.max_workers <= 0:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus is not None:
            max_workers = max(int(slurm_cpus), 1)
        else:
            max_workers = max(os.cpu_count() or 1, 1)
    else:
        max_workers = args.max_workers

    signature = _signature(args)
    signature_hash = _signature_hash(signature)
    if signature_path.exists():
        with open(signature_path, "r", encoding="utf-8") as f:
            existing_signature = json.load(f)
        if existing_signature.get("signature_hash") != signature_hash:
            raise RuntimeError(
                f"Checkpoint directory {out_dir} contains partial samples from a different configuration. "
                "Use a different --out-dir or clear the partial checkpoints first."
            )
    else:
        _write_json_atomic(signature_path, {"signature_hash": signature_hash, "signature": signature})

    x_positions = np.linspace(-args.half_span, args.half_span, args.n_points, dtype=float)
    if args.n_points > 1:
        dx = float(x_positions[1] - x_positions[0])
    else:
        dx = 2.0 * args.half_span
    source_half_width = dx / 2.0

    source_s1 = x_positions - source_half_width
    source_s2 = x_positions + source_half_width

    eta_min = float(args.eta_min)
    eta_max = float(args.eta_max)
    if args.damping is not None:
        eta_min = float(args.damping)
        eta_max = float(args.damping)

    freq_range_hz = None
    if args.freq_min is not None and args.freq_max is not None:
        freq_range_hz = (float(args.freq_min), float(args.freq_max))

    params_array = sample_parameters(
        n_samples=args.n_samples,
        rng_seed=args.seed,
        c44_range=(args.c44_min, args.c44_max),
        c11_ratio_range=(args.c11_ratio_min, args.c11_ratio_max),
        c33_ratio_range=(args.c33_ratio_min, args.c33_ratio_max),
        c13_ratio_range=(args.c13_ratio_min, args.c13_ratio_max),
        rho_range=(args.rho_min, args.rho_max),
        eta_range=(eta_min, eta_max),
        a0_range=(args.a0_min, args.a0_max),
        source_half_width=source_half_width,
        freq_range_hz=freq_range_hz,
    )

    metadata = {
        "N": int(args.n_points),
        "L": float(args.half_span),
        "x_positions": x_positions.tolist(),
        "source_element_s1": source_s1.tolist(),
        "source_element_s2": source_s2.tolist(),
        "source_element_midpoints": x_positions.tolist(),
        "dx": dx,
        "source_half_width": source_half_width,
        "strip_half_width": source_half_width,
        "operator_query_layout": ["x_field", "s1_source", "s2_source"],
        "params_array_layout": ["c11", "c13", "c33", "c44", "rho", "eta", "a0", "omega"],
        "param_names": ["c11", "c13", "c33", "c44", "rho", "eta", "a0", "omega"],
        "param_ranges": {
            "c44": [args.c44_min, args.c44_max],
            "c11_ratio": [args.c11_ratio_min, args.c11_ratio_max],
            "c33_ratio": [args.c33_ratio_min, args.c33_ratio_max],
            "c13_ratio": [args.c13_ratio_min, args.c13_ratio_max],
            "rho": [args.rho_min, args.rho_max],
            "eta": [eta_min, eta_max],
            "a0": [args.a0_min, args.a0_max],
            "frequency_hz": None if freq_range_hz is None else [freq_range_hz[0], freq_range_hz[1]],
        },
        "lty": int(args.lty),
        "z_source": float(args.z_source),
        "z_field": float(args.z_field),
        "a0_definition": "a0 = omega * b / cS, cS = sqrt(c44/rho), b = source_half_width",
        "formulation": "homogeneous_transversely_isotropic_halfspace_full_influence_matrix",
        "generator": "scripts/ground_vibration/generate_external_dataset.py",
    }

    tasks = [
        (
            sample_idx,
            params_array[sample_idx],
            x_positions,
            source_half_width,
            args.lty,
            args.z_field,
            args.z_source,
            args.abs_tol,
        )
        for sample_idx in range(args.n_samples)
    ]

    start = time.perf_counter()
    g_samples = np.zeros((args.n_samples, 2 * args.n_points, 2 * args.n_points), dtype=np.complex128)
    sample_times = np.zeros((args.n_samples,), dtype=float)
    completed_mask = _load_completed_samples(partials_dir, g_samples, sample_times)
    resumed_count = int(np.count_nonzero(completed_mask))
    completed_count = resumed_count

    if resumed_count:
        print(
            f"Resuming dataset generation from {out_dir}: found {resumed_count}/{args.n_samples} completed samples.",
            flush=True,
        )
    else:
        print(
            f"Starting dataset generation in {out_dir} with {args.n_samples} samples, "
            f"{args.n_points} points, and {max_workers} worker(s).",
            flush=True,
        )

    _write_progress_file(
        progress_path=progress_path,
        completed_count=completed_count,
        total_samples=args.n_samples,
        started_at=start,
        sample_times=sample_times[completed_mask],
        resumed_count=resumed_count,
        max_workers=max_workers,
        signature_hash=signature_hash,
        status="running",
    )

    pending_tasks = [task for task in tasks if not completed_mask[task[0]]]

    if max_workers <= 1:
        for task in pending_tasks:
            idx, g_full, elapsed = compute_sample(*task)
            g_samples[idx] = g_full
            sample_times[idx] = elapsed
            completed_mask[idx] = True
            _write_sample_checkpoint(partials_dir, idx, g_full, elapsed)
            completed_count += 1
            if (
                completed_count == args.n_samples
                or completed_count == resumed_count + 1
                or completed_count % max(args.progress_every, 1) == 0
            ):
                done_times = sample_times[completed_mask]
                _print_progress(
                    completed_count=completed_count,
                    total_samples=args.n_samples,
                    started_at=start,
                    sample_times=done_times,
                    resumed_count=resumed_count,
                    max_workers=max_workers,
                )
                _write_progress_file(
                    progress_path=progress_path,
                    completed_count=completed_count,
                    total_samples=args.n_samples,
                    started_at=start,
                    sample_times=done_times,
                    resumed_count=resumed_count,
                    max_workers=max_workers,
                    signature_hash=signature_hash,
                    status="running",
                )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(compute_sample_star, task): task[0] for task in pending_tasks}
            for future in as_completed(futures):
                idx, g_full, elapsed = future.result()
                g_samples[idx] = g_full
                sample_times[idx] = elapsed
                completed_mask[idx] = True
                _write_sample_checkpoint(partials_dir, idx, g_full, elapsed)
                completed_count += 1
                if (
                    completed_count == args.n_samples
                    or completed_count == resumed_count + 1
                    or completed_count % max(args.progress_every, 1) == 0
                ):
                    done_times = sample_times[completed_mask]
                    _print_progress(
                        completed_count=completed_count,
                        total_samples=args.n_samples,
                        started_at=start,
                        sample_times=done_times,
                        resumed_count=resumed_count,
                        max_workers=max_workers,
                    )
                    _write_progress_file(
                        progress_path=progress_path,
                        completed_count=completed_count,
                        total_samples=args.n_samples,
                        started_at=start,
                        sample_times=done_times,
                        resumed_count=resumed_count,
                        max_workers=max_workers,
                        signature_hash=signature_hash,
                        status="running",
                    )

    total_time = time.perf_counter() - start
    metadata["total_time_s"] = float(total_time)
    metadata["max_workers"] = int(max_workers)
    metadata["sample_times_mean_s"] = float(np.mean(sample_times))
    metadata["sample_times_std_s"] = float(np.std(sample_times))

    write_dataset_bundle(
        out_dir=out_dir,
        g_samples=g_samples,
        params_array=params_array,
        metadata=metadata,
        sample_times=sample_times,
    )

    _write_progress_file(
        progress_path=progress_path,
        completed_count=args.n_samples,
        total_samples=args.n_samples,
        started_at=start,
        sample_times=sample_times,
        resumed_count=resumed_count,
        max_workers=max_workers,
        signature_hash=signature_hash,
        status="completed",
    )

    for sample_path in partials_dir.glob("sample_*.npz"):
        sample_path.unlink()
    try:
        partials_dir.rmdir()
    except OSError:
        pass

    print(f"Wrote dataset bundle to {out_dir}", flush=True)
    print(f"Total runtime: {total_time:.2f} s | mean per sample: {np.mean(sample_times):.2f} s", flush=True)


if __name__ == "__main__":
    main()
