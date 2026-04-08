from __future__ import annotations
import yaml
import logging
import numpy as np
from typing import Any
from pathlib import Path
import matplotlib.pyplot as plt
from src.problems.ground_vibration import postprocessing as ppr
from src.problems.ground_vibration import plot_helper as helper
from src.modules.models.config import DataConfig, TestConfig

logger = logging.getLogger(__file__)

def _normalize_label(label: str) -> str:
    return "".join(ch for ch in str(label).lower() if ch.isalnum())

def _resolve_reference_parameter(data_cfg: DataConfig, test_cfg: TestConfig) -> str:
    if test_cfg.config is None:
        raise KeyError("Plot config not found")

    requested = test_cfg.config.get("plot_reference_parameter", None)
    if requested is not None:
        req_norm = _normalize_label(str(requested))
        for param in data_cfg.input_functions:
            p_norm = _normalize_label(str(param))
            if p_norm == req_norm or req_norm in p_norm:
                return str(param)

    for param in data_cfg.input_functions:
        if "omega" in _normalize_label(param):
            return str(param)
    return str(data_cfg.input_functions[0])

def get_plotted_samples_indices(data_cfg: DataConfig, test_cfg: TestConfig) -> tuple[dict[str, dict[str, list[np.intp | float]]], np.ndarray]:
    if test_cfg.config is None:
        raise KeyError("Plot config not found")
    percentile_sample_by_parameter = {}
    chosen_percentiles = min(test_cfg.config["percentiles"], len(
        data_cfg.split_indices[data_cfg.features[0] + '_test']))
    percentiles = np.linspace(
        0, 100, num=chosen_percentiles)
    for pos, parameter in enumerate(data_cfg.input_functions):
        indices = []
        targets = []
        for perc in percentiles:
            target = np.percentile(
                data_cfg.data[data_cfg.features[0]][data_cfg.split_indices[data_cfg.features[0] + '_test']][:, pos], perc)
            idx = np.argmin(
                np.abs(data_cfg.data[data_cfg.features[0]][data_cfg.split_indices[data_cfg.features[0] + '_test']][:, pos] - target))
            indices.append(idx)
            targets.append(target)
        percentile_sample_by_parameter[parameter] = {
            'indices': indices,
            'values': targets
        }
    return percentile_sample_by_parameter, percentiles

def get_plot_metatada(
    percentiles_sample_map: dict[str, dict[str, list[Any]]],
    percentiles: np.ndarray,
    reference_parameter: str,
):
    metadata = {}
    parameters_info = percentiles_sample_map.copy()
    for param in parameters_info:
        parameters_info[param]['values'] = [
            float(f"{v:.3f}") for v in parameters_info[param]['values']]
        parameters_info[param]['indices'] = [
            int(i) for i in parameters_info[param]['indices']]
    metadata = {
        'percentiles': [round(perc) for perc in percentiles],
        'reference_parameter': reference_parameter,
        'reference_samples': parameters_info[reference_parameter],
        'all_parameter_samples': parameters_info,
    }
    return metadata

def save_plot_metadata(metadata: dict[str, Any], save_path: str):
    with open(f'{save_path}/plot_metadata.yaml', mode='w') as file:
        yaml.safe_dump(metadata, file, allow_unicode=True)

def _build_surface_traction_vectors(x: np.ndarray) -> dict[str, np.ndarray]:
    n_nodes = len(x)
    q_uniform = np.zeros((2 * n_nodes,), dtype=np.complex128)
    q_uniform[1::2] = 1.0

    domain = float(np.max(x) - np.min(x))
    k = 2.0 * np.pi / max(domain, 1e-12)
    q_harmonic = np.zeros((2 * n_nodes,), dtype=np.complex128)
    q_harmonic[1::2] = np.exp(-1j * k * x)

    mask = (x >= (-0.25 * domain)) & (x <= (0.25 * domain))
    patch = mask.astype(float)
    patch = patch / max(np.sum(patch), 1.0)
    q_patch = np.zeros((2 * n_nodes,), dtype=np.complex128)
    q_patch[1::2] = patch

    return {
        "uniform_vertical": q_uniform,
        "harmonic_vertical": q_harmonic,
        "patch_vertical": q_patch,
    }

def _relative_norm_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-14))

def _build_operator_report(
    ground_truths: np.ndarray,
    predictions: np.ndarray,
    U_true: np.ndarray,
    U_pred: np.ndarray,
    x: np.ndarray,
) -> dict[str, Any]:
    # Channel-wise operator error over test split.
    channel_names = ["u_xx", "u_xz", "u_zx", "u_zz"]
    channel_errors = {}
    for ch, name in enumerate(channel_names):
        channel_errors[name] = _relative_norm_error(ground_truths[..., ch], predictions[..., ch])

    sample_rel_errors = np.asarray([_relative_norm_error(U_true[i], U_pred[i]) for i in range(U_true.shape[0])], dtype=float)
    reciprocity_true = np.asarray([_relative_norm_error(U_true[i], U_true[i].T) for i in range(U_true.shape[0])], dtype=float)
    reciprocity_pred = np.asarray([_relative_norm_error(U_pred[i], U_pred[i].T) for i in range(U_pred.shape[0])], dtype=float)

    traction_vectors = _build_surface_traction_vectors(x=x)
    response_metrics: dict[str, Any] = {}
    for load_name, q in traction_vectors.items():
        rel_total = []
        rel_vertical = []
        for i in range(U_true.shape[0]):
            w_true = U_true[i] @ q
            w_pred = U_pred[i] @ q
            rel_total.append(_relative_norm_error(w_true, w_pred))
            rel_vertical.append(_relative_norm_error(w_true[1::2], w_pred[1::2]))
        rel_total = np.asarray(rel_total, dtype=float)
        rel_vertical = np.asarray(rel_vertical, dtype=float)
        response_metrics[load_name] = {
            "mean_relative_error_full_dof": float(np.mean(rel_total)),
            "p90_relative_error_full_dof": float(np.percentile(rel_total, 90)),
            "mean_relative_error_vertical_dof": float(np.mean(rel_vertical)),
            "p90_relative_error_vertical_dof": float(np.percentile(rel_vertical, 90)),
        }

    report = {
        "scope": "soil influence operator surrogate (not coupled wall-soil IBEM-FEM)",
        "test_samples": int(U_true.shape[0]),
        "operator_relative_error": {
            "mean": float(np.mean(sample_rel_errors)),
            "median": float(np.median(sample_rel_errors)),
            "p90": float(np.percentile(sample_rel_errors, 90)),
            "max": float(np.max(sample_rel_errors)),
        },
        "channel_relative_error": channel_errors,
        "reciprocity_relative_error": {
            "true_mean": float(np.mean(reciprocity_true)),
            "pred_mean": float(np.mean(reciprocity_pred)),
            "true_p90": float(np.percentile(reciprocity_true, 90)),
            "pred_p90": float(np.percentile(reciprocity_pred, 90)),
        },
        "response_under_surface_tractions": response_metrics,
    }
    return report

def _plot_operator_responses(
    U_true: np.ndarray,
    U_pred: np.ndarray,
    x: np.ndarray,
    out_dir: Path,
    num_samples: int = 3,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    traction_vectors = _build_surface_traction_vectors(x=x)

    total_samples = min(int(num_samples), U_true.shape[0])
    for sample_idx in range(total_samples):
        for load_name, q in traction_vectors.items():
            w_true = U_true[sample_idx] @ q
            w_pred = U_pred[sample_idx] @ q
            wz_true = w_true[1::2]
            wz_pred = w_pred[1::2]

            fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
            axes[0].plot(x, wz_true.real, "-k", lw=1.5, label="true")
            axes[0].plot(x, wz_pred.real, "--r", lw=1.2, label="pred")
            axes[0].set_title("Re(w_z)")
            axes[0].set_xlabel("x")
            axes[0].legend()

            axes[1].plot(x, wz_true.imag, "-k", lw=1.5, label="true")
            axes[1].plot(x, wz_pred.imag, "--r", lw=1.2, label="pred")
            axes[1].set_title("Im(w_z)")
            axes[1].set_xlabel("x")
            axes[1].legend()

            axes[2].plot(x, np.abs(wz_true), "-k", lw=1.5, label="true")
            axes[2].plot(x, np.abs(wz_pred), "--r", lw=1.2, label="pred")
            axes[2].set_title("|w_z|")
            axes[2].set_xlabel("x")
            axes[2].legend()

            fig.suptitle(f"Sample {sample_idx} - {load_name}")
            fig.savefig(out_dir / f"sample_{sample_idx:03d}_{load_name}.png", dpi=180)
            plt.close(fig)

def run_problem_specific_plotting(data: dict[str, Any], data_cfg: DataConfig, test_cfg: TestConfig):
    if test_cfg.problem is None:
        raise AttributeError(f"'Problem' attribute is missing.")
    plots_path = test_cfg.output_path / test_cfg.problem / \
        test_cfg.experiment_version / 'plots'
    plots_path.mkdir(parents=True, exist_ok=True)
    percentile_sample_by_parameter, percentiles = get_plotted_samples_indices(
        data_cfg=data_cfg, test_cfg=test_cfg)
    reference_parameter = _resolve_reference_parameter(data_cfg=data_cfg, test_cfg=test_cfg)
    metadata = get_plot_metatada(
        percentiles_sample_map=percentile_sample_by_parameter,
        percentiles=percentiles,
        reference_parameter=reference_parameter,
    )

    save_plot_metadata(metadata, str(plots_path))

    plane_plots_path = plots_path / 'plane_plots'
    axis_plots_path = plots_path / 'axis_plots'
    basis_plots_path = plots_path / 'basis_plots'
    coefficients_plots_path = plots_path / 'coefficients_plots'

    for path in [plane_plots_path, axis_plots_path, basis_plots_path, coefficients_plots_path]:
        path.mkdir(exist_ok=True)

    if test_cfg.config is None:
        raise AttributeError("Plotting config not found.")

    if test_cfg.config['plot_plane']:
        helper.plot_influence_matrix_helper(
            data=data, 
            data_cfg=data_cfg, 
            metadata=metadata, 
            plot_path=plane_plots_path
        )

    # if test_cfg.config['plot_axis']:
    #     helper.plot_axis_helper(
    #         data=data,
    #         data_cfg=data_cfg,
    #         metadata=metadata,
    #         plot_path=axis_plots_path
    #     )

    if test_cfg.config['plot_basis']:
        helper.plot_basis_helper(
            data=data,
            data_cfg=data_cfg,
            plot_path=basis_plots_path
        )

    if test_cfg.config['plot_coefficients']:
        helper.plot_coefficients_helper(
            data=data,
            data_cfg=data_cfg,
            metadata=metadata,
            plot_path=coefficients_plots_path
        )

    if test_cfg.config['plot_coefficients_mean']:
        helper.plot_coefficients_mean_helper(
            data=data,
            data_cfg=data_cfg,
            test_cfg=test_cfg,
            plot_path=coefficients_plots_path
        )

    operator_metrics_path = plots_path / "operator_metrics"
    operator_metrics_path.mkdir(exist_ok=True)
    report = _build_operator_report(
        ground_truths=data["ground_truths"],
        predictions=data["predictions"],
        U_true=data["U_true"],
        U_pred=data["U_pred"],
        x=data["x_vector"],
    )
    with open(operator_metrics_path / "operator_report.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, sort_keys=False, allow_unicode=True)

    do_response_plots = bool(test_cfg.config.get("plot_operator_responses", True))
    if do_response_plots:
        _plot_operator_responses(
            U_true=data["U_true"],
            U_pred=data["U_pred"],
            x=data["x_vector"],
            out_dir=operator_metrics_path / "response_profiles",
            num_samples=int(test_cfg.config.get("operator_response_samples", 3)),
        )


def plot_metrics(test_cfg: TestConfig, data_cfg: DataConfig):
    input_functions = ppr.get_input_functions(data_cfg)
    coordinates = ppr.get_coordinates(data_cfg)
    output_data = ppr.get_output_data(test_cfg)
    ground_truths = ppr.format_target(output_data[data_cfg.targets[0]], data_cfg)
    predictions = ppr.format_target(output_data['predictions'], data_cfg)
    coefficients = ppr.reshape_coefficients(output_data['branch_output'], data_cfg, test_cfg)
    basis = ppr.reshape_basis(output_data['trunk_output'], data_cfg, test_cfg)
    bias = ppr.format_bias(output_data['bias'], data_cfg, test_cfg)

    U_basis = ppr.get_U(basis)
    U_true = ppr.get_U(ground_truths)
    U_pred = ppr.get_U(predictions)
    U_matrix_path = Path(data_cfg.raw_outputs_path) / data_cfg.problem / test_cfg.experiment_version / 'aux'
    ppr.save_U_matrix(
        influence_matrix_true=ground_truths,
        influence_matrix_pred=predictions,
        save_path=U_matrix_path
    )

    xt_test = output_data['xt']
    x_vector = np.unique(np.asarray(xt_test)[:, 0])
    xb_test = data_cfg.data[data_cfg.features[0]][data_cfg.split_indices['xb_test']]

    data = {
        'input_functions': input_functions,
        'coordinates': coordinates,
        'output_data': output_data,
        'ground_truths': ground_truths,
        'predictions': predictions,
        'coefficients': coefficients,
        'basis': basis,
        'bias': bias,
        'U_basis': U_basis,
        'U_true': U_true,
        'U_pred': U_pred,
        'x_vector': x_vector,
        'xb_test': xb_test,
    }

    run_problem_specific_plotting(
        data=data, data_cfg=data_cfg, test_cfg=test_cfg)
