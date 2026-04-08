from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from src.modules.models.config import DataConfig, TestConfig
from src.problems.rajapakse_fixed_material.plot_axis import plot_axis
from src.problems.rajapakse_fixed_material.plot_basis import plot_basis
from src.problems.rajapakse_fixed_material.plot_coeffs import (
    plot_coefficients,
    plot_coefficients_mean,
)
from src.problems.rajapakse_fixed_material.plot_field import plot_2D_field

logger = logging.getLogger(__file__)


def _slugify(label: str) -> str:
    keep = []
    for ch in label:
        if ch.isalnum() or ch in {"_", "-"}:
            keep.append(ch)
    return "".join(keep) or "param"


def plot_planes_helper(
    data: dict[str, dict[str, Any]],
    data_cfg: DataConfig,
    metadata: dict[str, Any],
    plot_path: Path,
):
    sample_map = metadata["reference_samples"]
    reference_parameter = str(metadata["reference_parameter"])
    percentiles = metadata["percentiles"]

    for count, idx in tqdm(enumerate(sample_map["indices"]), colour="green"):
        param_val = float(sample_map["values"][count])
        row = data["xb_test"][idx]
        fig_plane = plot_2D_field(
            coords=data["coordinates"],
            truth_field=data["ground_truths"][idx],
            pred_field=data["predictions"][idx],
            input_function_labels=data_cfg.input_functions,
            input_function_value=[float(v) for v in row],
            target_labels=data_cfg.targets_labels,
        )
        val_str = f"{param_val:.2E}"
        file_name = f"{percentiles[count]:.0f}_th_percentile_{_slugify(reference_parameter)}={val_str}.png"
        fig_plane.savefig(plot_path / file_name)
        plt.close(fig_plane)


def plot_axis_helper(data: dict[str, Any], data_cfg: DataConfig, metadata: dict[str, Any], plot_path: Path):
    sample_map = metadata["reference_samples"]
    reference_parameter = str(metadata["reference_parameter"])

    for count, idx in tqdm(enumerate(sample_map["indices"]), colour="green"):
        param_val = float(sample_map["values"][count])
        fig_axis = plot_axis(
            coords=data["coordinates"],
            truth_field=data["ground_truths"][idx],
            pred_field=data["predictions"][idx],
            param_map={reference_parameter: param_val},  # type: ignore[arg-type]
            target_labels=data_cfg.targets_labels,
        )
        val_str = f"{param_val:.2E}"
        file_name = f"{_slugify(reference_parameter)}={val_str}.png"
        fig_axis.savefig(plot_path / file_name)
        plt.close(fig_axis)


def plot_basis_helper(data: dict[str, Any], data_cfg: DataConfig, plot_path: Path):
    if data["bias"].ndim > 1:
        fig_bias = plot_basis(
            coords=data["coordinates"],
            basis=data["bias"],
            index=0,
            target_labels=data_cfg.targets_labels,
        )
        fig_bias.savefig(plot_path / "bias.png")
        plt.close(fig_bias)

    for i in tqdm(range(1, len(data["basis"]) + 1), colour="blue"):
        fig_basis = plot_basis(
            coords=data["coordinates"],
            basis=data["basis"][i - 1],
            index=i,
            target_labels=data_cfg.targets_labels,
        )
        fig_basis.savefig(plot_path / f"vector_{i}.png")
        plt.close(fig_basis)


def plot_coefficients_helper(data: dict[str, Any], data_cfg: DataConfig, metadata: dict[str, Any], plot_path: Path):
    sample_map = metadata["reference_samples"]
    reference_parameter = str(metadata["reference_parameter"])

    for count, idx in tqdm(enumerate(sample_map["indices"]), colour="blue"):
        param_val = float(sample_map["values"][count])
        fig_coeffs = plot_coefficients(
            branch_output_sample=data["coefficients"][idx],
            basis=data["basis"],
            input_function_map={reference_parameter: param_val},
            target_labels=data_cfg.targets_labels,
        )
        val_str = f"{param_val:.2E}"
        fig_coeffs.savefig(plot_path / f"coeffs_{_slugify(reference_parameter)}={val_str}.png")
        plt.close(fig_coeffs)


def plot_coefficients_mean_helper(data: dict[str, Any], data_cfg: DataConfig, test_cfg: TestConfig, plot_path: Path):
    if test_cfg.config is None:
        raise AttributeError("Missing test plotting config")

    fig_coeffs_mean = plot_coefficients_mean(
        vectors=data["basis"],
        coefficients=data["coefficients"],
        num_vectors_to_highlight=test_cfg.config["vectors_to_highlight"],
        target_labels=data_cfg.targets_labels,
    )
    fig_coeffs_mean.savefig(plot_path / "coeffs_mean.png")
    plt.close(fig_coeffs_mean)


def plot_error_vs_parameter(
    parameter: np.ndarray,
    errors: dict[str, np.ndarray],
    parameter_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for name, values in errors.items():
        ax.scatter(parameter, values, s=12, alpha=0.65, label=name)
    ax.set_xlabel(parameter_label)
    ax.set_ylabel("relative L2 error")
    ax.set_title("Error versus reference parameter")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_timing_bars(labels: list[str], times_s: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bars = ax.bar(labels, times_s, color=["#444", "#1f77b4", "#2ca02c"][: len(labels)])
    ax.set_ylabel("seconds")
    ax.set_title("Integration and inference timing")
    ax.grid(axis="y", alpha=0.2)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3e}", ha="center", va="bottom", fontsize=8)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_parameter_coverage(
    xb: np.ndarray,
    labels: list[str],
    out_path: Path,
) -> None:
    n_params = xb.shape[1]
    fig, axes = plt.subplots(n_params, 1, figsize=(7.5, 2.2 * n_params), constrained_layout=True)
    if n_params == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.hist(xb[:, i], bins=40, color="#4e79a7", alpha=0.85)
        ax.set_ylabel("count")
        ax.set_xlabel(labels[i])
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Input-parameter coverage (paper-aligned nondimensional variables)")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
