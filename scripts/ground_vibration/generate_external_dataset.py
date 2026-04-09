#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
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
    freq_range: tuple[float, float],
) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    lhs = lhs_sample(n_samples, 6, rng)
    params = np.zeros((n_samples, 6), dtype=float)

    for i in range(n_samples):
        c44 = c44_range[0] + lhs[i, 0] * (c44_range[1] - c44_range[0])
        c11 = c44 * (c11_ratio_range[0] + lhs[i, 1] * (c11_ratio_range[1] - c11_ratio_range[0]))
        c33 = c44 * (c33_ratio_range[0] + lhs[i, 2] * (c33_ratio_range[1] - c33_ratio_range[0]))

        max_c13_ratio = (math.sqrt(max(c11 * c33, 1e-14)) - c44) / c44
        sampled_c13_ratio = c13_ratio_range[0] + lhs[i, 3] * (c13_ratio_range[1] - c13_ratio_range[0])
        c13_ratio = min(sampled_c13_ratio, max_c13_ratio * 0.95)
        c13 = c44 * c13_ratio

        rho = rho_range[0] + lhs[i, 4] * (rho_range[1] - rho_range[0])
        freq = freq_range[0] + lhs[i, 5] * (freq_range[1] - freq_range[0])
        omega = 2.0 * math.pi * freq
        params[i, :] = [c11, c13, c33, c44, rho, omega]

    return params


def _safe_zeta(zeta: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(zeta, dtype=np.complex128)
    return np.where(np.abs(arr) < 1e-12, 1e-12 + 0j, arr)


def influencefunctions(
    lty: int,
    rho: float,
    omega: float,
    damping: float,
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

    alpha = alpha0 * (1.0 + (damping * j))
    beta = beta0 * (1.0 + (damping * j))
    kappa = kappa0 * (1.0 + (damping * j))

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
            return np.vstack([
                np.real(fun1),
                np.imag(fun1),
                np.real(fun3),
                np.imag(fun3),
                np.real(fun4),
                np.imag(fun4),
            ])
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
    damping: float,
    z_field: float,
    z_source: float,
    abs_tol: float,
) -> tuple[int, np.ndarray, float]:
    start = time.perf_counter()
    n_points = x_positions.shape[0]
    g_full = np.zeros((2 * n_points, 2 * n_points), dtype=np.complex128)

    c11, c13, c33, c44, rho, omega = [float(v) for v in params_row]

    for j in range(n_points):
        x_source = float(x_positions[j])
        for i in range(j + 1):
            x_field = float(x_positions[i])
            uc = influencefunctions(
                lty=lty,
                rho=rho,
                omega=omega,
                damping=damping,
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
    parser.add_argument("--damping", type=float, default=0.01)
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
    parser.add_argument("--freq-min", type=float, default=10.0)
    parser.add_argument("--freq-max", type=float, default=200.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    if args.max_workers <= 0:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus is not None:
            max_workers = max(int(slurm_cpus), 1)
        else:
            max_workers = max(os.cpu_count() or 1, 1)
    else:
        max_workers = args.max_workers

    params_array = sample_parameters(
        n_samples=args.n_samples,
        rng_seed=args.seed,
        c44_range=(args.c44_min, args.c44_max),
        c11_ratio_range=(args.c11_ratio_min, args.c11_ratio_max),
        c33_ratio_range=(args.c33_ratio_min, args.c33_ratio_max),
        c13_ratio_range=(args.c13_ratio_min, args.c13_ratio_max),
        rho_range=(args.rho_min, args.rho_max),
        freq_range=(args.freq_min, args.freq_max),
    )

    x_positions = np.linspace(-args.half_span, args.half_span, args.n_points, dtype=float)
    if args.n_points > 1:
        dx = float(x_positions[1] - x_positions[0])
    else:
        dx = 2.0 * args.half_span
    source_half_width = dx / 2.0

    metadata = {
        "N": int(args.n_points),
        "L": float(args.half_span),
        "x_positions": x_positions.tolist(),
        "dx": dx,
        "source_half_width": source_half_width,
        "strip_half_width": source_half_width,
        "param_names": ["c11", "c13", "c33", "c44", "rho", "omega"],
        "param_ranges": {
            "c44": [args.c44_min, args.c44_max],
            "c11_ratio": [args.c11_ratio_min, args.c11_ratio_max],
            "c33_ratio": [args.c33_ratio_min, args.c33_ratio_max],
            "c13_ratio": [args.c13_ratio_min, args.c13_ratio_max],
            "rho": [args.rho_min, args.rho_max],
            "freq": [args.freq_min, args.freq_max],
        },
        "lty": int(args.lty),
        "z_source": float(args.z_source),
        "z_field": float(args.z_field),
        "damping": float(args.damping),
        "generator": "scripts/ground_vibration/generate_external_dataset.py",
    }

    tasks = [
        (
            sample_idx,
            params_array[sample_idx],
            x_positions,
            source_half_width,
            args.lty,
            args.damping,
            args.z_field,
            args.z_source,
            args.abs_tol,
        )
        for sample_idx in range(args.n_samples)
    ]

    start = time.perf_counter()
    g_samples = np.zeros((args.n_samples, 2 * args.n_points, 2 * args.n_points), dtype=np.complex128)
    sample_times = np.zeros((args.n_samples,), dtype=float)

    if max_workers <= 1:
        for task in tasks:
            idx, g_full, elapsed = compute_sample(*task)
            g_samples[idx] = g_full
            sample_times[idx] = elapsed
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for idx, g_full, elapsed in executor.map(compute_sample_star, tasks):
                g_samples[idx] = g_full
                sample_times[idx] = elapsed

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

    print(f"Wrote dataset bundle to {out_dir}")
    print(f"Total runtime: {total_time:.2f} s | mean per sample: {np.mean(sample_times):.2f} s")


if __name__ == "__main__":
    main()
