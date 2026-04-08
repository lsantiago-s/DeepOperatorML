import logging
from typing import Any
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
from src.modules.models.config import DataConfig, TestConfig
from src.problems.ground_vibration.plot_influence_matrix import plot_influence_matrix
from src.problems.ground_vibration.plot_axis import plot_axis
from src.problems.ground_vibration.plot_basis import plot_basis
from src.problems.ground_vibration.plot_coeffs import plot_coefficients, plot_coefficients_mean

logger = logging.getLogger(__file__)

def _slugify_label(label: str) -> str:
    keep = []
    for ch in label:
        if ch.isalnum():
            keep.append(ch)
        elif ch in {"_", "-"}:
            keep.append(ch)
    return "".join(keep) or "param"


def plot_influence_matrix_helper(data: dict[str, dict[str, Any]], data_cfg: DataConfig, metadata: dict[str, Any], plot_path: Path):
    sample_map = metadata["reference_samples"]
    ref_parameter = str(metadata["reference_parameter"])
    percentiles = metadata['percentiles']

    for count, idx in tqdm(enumerate(sample_map['indices']), colour='green'):
        param_val = float(sample_map['values'][count])
        param_row = data["xb_test"][idx]
        param_map = {
            str(key): float(value)
            for key, value in zip(data_cfg.input_functions, param_row)
        }

        fig_inf_matrix = plot_influence_matrix(
            U_true=data['ground_truths'][idx],
            U_pred=data['predictions'][idx],
            param_map=param_map,
            target_labels=["u_xx", "u_xz", "u_zx", "u_zz"],
        )

        val_str = f"{param_val:.2E}"
        ref_slug = _slugify_label(ref_parameter)
        file_name = f"{percentiles[count]:.0f}_th_percentile_{ref_slug}={val_str}.png"
        fig_inf_matrix_path = plot_path / file_name
        fig_inf_matrix.savefig(fig_inf_matrix_path)
        plt.close()
        

def plot_nodal_displacements_helper(data: dict[str, Any], data_cfg: DataConfig, metadata: dict[str, Any], plot_path: Path):
    sample_map = metadata["reference_samples"]
    ref_parameter = str(metadata["reference_parameter"])

    for count, idx in tqdm(enumerate(sample_map['indices']), colour='green'):
        param_val = float(sample_map['values'][count])
        fig_axis = plot_axis(
            coords=data['coordinates'],
            truth_field=data['ground_truths'][idx],
            pred_field=data['predictions'][idx],
            param_map={ref_parameter: param_val},  # type: ignore[arg-type]
            target_labels=data_cfg.targets_labels
        )
        val_str = f"{param_val:.2f}"
        file_name = _slugify_label(ref_parameter) + f"={val_str}.png"
        fig_axis_path = plot_path / file_name
        fig_axis.savefig(fig_axis_path)
        plt.close()


def plot_basis_helper(data: dict[str, Any], data_cfg: DataConfig, plot_path: Path):
    if data['bias'].ndim > 1:
        fig_bias = plot_basis(
            coords=data['coordinates'],
            basis=data['bias'],
            index=0,
            target_labels=data_cfg.targets_labels
        )
        fig_basis_path = plot_path / f"bias.png"
        fig_bias.savefig(fig_basis_path)
        plt.close()

    for i in tqdm(range(1, len(data['basis']) + 1), colour='blue'):
        fig_basis = plot_basis(
            basis=data['U_basis'][i - 1],
            index=i,
            target_labels=data_cfg.targets_labels
        )
        fig_basis_path = plot_path / f"vector_{i}.png"
        fig_basis.savefig(fig_basis_path)
        plt.close()


def plot_coefficients_helper(data: dict[str, Any], data_cfg: DataConfig, metadata: dict[str, Any], plot_path: Path):
    sample_map = metadata["reference_samples"]
    ref_parameter = str(metadata["reference_parameter"])

    for count, idx in tqdm(enumerate(sample_map['indices']), colour='blue'):
        param_val = float(sample_map['values'][count])
        fig_coeffs = plot_coefficients(
            branch_output_sample=data['coefficients'][idx],
            basis=data['basis'],
            input_function_map={
                ref_parameter: param_val},
            target_labels=data_cfg.targets_labels
        )
        val_str = f"{param_val:.2E}"
        file_name = f"coeffs_{_slugify_label(ref_parameter)}={val_str}.png"
        fig_coeffs_path = plot_path / file_name
        fig_coeffs.savefig(fig_coeffs_path)
        plt.close()


def plot_coefficients_mean_helper(data: dict[str, Any], data_cfg: DataConfig, test_cfg: TestConfig, plot_path: Path):
    if test_cfg.config is None:
        raise AttributeError(f"Missing attribute 'config'")
    fig_coeffs_mean = plot_coefficients_mean(
        vectors=data['basis'],
        coefficients=data['coefficients'],
        num_vectors_to_highlight=test_cfg.config['vectors_to_highlight'],
        target_labels=data_cfg.targets_labels
    )
    fig_coeffs_mean_path = plot_path / \
        f"coeffs_mean.png"
    fig_coeffs_mean.savefig(fig_coeffs_mean_path)
    plt.close()
