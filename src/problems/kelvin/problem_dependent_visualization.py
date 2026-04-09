from __future__ import annotations
import csv
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any
from src.problems.kelvin import postprocessing as ppr
from src.problems.kelvin import plot_helper as helper
from src.modules.models.config import DataConfig, TestConfig

logger = logging.getLogger(__file__)


def _relative_error_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
    num = np.linalg.norm(y_true_flat - y_pred_flat, axis=1)
    den = np.linalg.norm(y_true_flat, axis=1) + 1e-14
    return num / den


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _per_channel_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    channel_labels: list[str],
) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    for ch, label in enumerate(channel_labels):
        err = _relative_error_per_sample(y_true[:, ch, ...], y_pred[:, ch, ...])
        summaries[str(label)] = _summary(err)
    return summaries


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _get_solver_timing(data_cfg: DataConfig) -> dict[str, float | None]:
    metadata_path = Path(data_cfg.raw_metadata_path)
    if not metadata_path.exists():
        return {"solver_total_s": None, "solver_per_sample_s": None}

    with open(metadata_path, "r", encoding="utf-8") as f:
        raw_meta = yaml.safe_load(f) or {}

    timing_breakdown = raw_meta.get("timing_breakdown", {})
    total = _safe_float(timing_breakdown.get("direct_solver_total_s"))
    per_sample = _safe_float(timing_breakdown.get("direct_solver_per_sample_s"))

    if total is None:
        total = _safe_float(raw_meta.get("runtime_s"))
    if total is None:
        runtime_ms = _safe_float(raw_meta.get("runtime_ms"))
        if runtime_ms is not None:
            if ("runtime_s" in raw_meta) or ("timing_breakdown" in raw_meta):
                total = runtime_ms / 1e3
            else:
                total = runtime_ms / 1e3

    if per_sample is None and total is not None:
        n_total = int(data_cfg.data[data_cfg.features[0]].shape[0])
        per_sample = total / max(n_total, 1)

    return {
        "solver_total_s": total,
        "solver_per_sample_s": per_sample,
    }


def _get_inference_timing(test_cfg: TestConfig) -> dict[str, float | None]:
    if test_cfg.problem is None:
        return {"inference_total_s": None, "inference_per_sample_s": None}

    time_path = (
        test_cfg.output_path
        / test_cfg.problem
        / test_cfg.experiment_version
        / "metrics"
        / "test_time.yaml"
    )
    if not time_path.exists():
        return {"inference_total_s": None, "inference_per_sample_s": None}

    with open(time_path, "r", encoding="utf-8") as f:
        test_time = yaml.safe_load(f) or {}

    return {
        "inference_total_s": _safe_float(test_time.get("inference_time")),
        "inference_per_sample_s": None,
    }


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _save_metrics_table_csv(path: Path, report: dict[str, Any], channel_labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = report["methods"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "method",
            "overall_mean",
            "overall_median",
            "overall_p90",
            "overall_max",
        ] + [f"{label}_mean" for label in channel_labels]
        writer.writerow(header)

        for method_name, metrics in methods.items():
            row = [
                method_name,
                metrics["overall"]["mean"],
                metrics["overall"]["median"],
                metrics["overall"]["p90"],
                metrics["overall"]["max"],
            ]
            for label in channel_labels:
                row.append(metrics["by_channel"][label]["mean"])
            writer.writerow(row)


def _plot_method_error_distributions(
    method_errors: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5), constrained_layout=True)
    labels = list(method_errors.keys())
    vals = [method_errors[name] for name in labels]
    ax.boxplot(vals, labels=labels, showfliers=False)
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Error distribution on test split")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_error_vs_inputs(
    xb_test: np.ndarray,
    method_errors: dict[str, np.ndarray],
    input_labels: list[str],
    output_path: Path,
) -> None:
    num_dims = int(min(2, xb_test.shape[1]))
    if num_dims <= 0:
        return

    fig, axes = plt.subplots(1, num_dims, figsize=(6 * num_dims, 4.5), constrained_layout=True)
    if num_dims == 1:
        axes = np.array([axes])

    for ax_i, ax in enumerate(axes):
        x = xb_test[:, ax_i]
        order = np.argsort(x)
        for method_name, errs in method_errors.items():
            ax.plot(x[order], errs[order], lw=1.2, label=method_name)
        label = input_labels[ax_i] if ax_i < len(input_labels) else f"branch_{ax_i}"
        ax.set_xlabel(str(label))
        ax.set_ylabel("Relative L2 error")
        ax.grid(alpha=0.25)
    axes[0].set_title("Error versus branch parameter 1")
    if num_dims > 1:
        axes[1].set_title("Error versus branch parameter 2")
        axes[1].legend()
    else:
        axes[0].legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_timing_comparison(timing_report: dict[str, Any], output_path: Path) -> None:
    solver = timing_report["reference_solver"].get("per_sample_s")
    infer = timing_report["inference"].get("per_sample_s")
    if solver is None or infer is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    labels = ["Direct Solver", "ML Inference"]
    values = [solver, infer]
    bars = ax.bar(labels, values, color=["#1f77b4", "#d62728"])
    ax.set_yscale("log")
    ax.set_ylabel("Seconds per sample (log scale)")
    ax.set_title("Per-sample runtime comparison")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _compute_baselines(data_cfg: DataConfig) -> dict[str, np.ndarray]:
    branch_key = data_cfg.features[0]
    target_key = data_cfg.targets[0]

    xb_train = data_cfg.data[branch_key][data_cfg.split_indices[f"{branch_key}_train"]]
    xb_test = data_cfg.data[branch_key][data_cfg.split_indices[f"{branch_key}_test"]]
    y_train = data_cfg.data[target_key][data_cfg.split_indices[f"{branch_key}_train"]]

    mean_pred = np.repeat(np.mean(y_train, axis=0, keepdims=True), xb_test.shape[0], axis=0)

    dists = np.linalg.norm(xb_test[:, None, :] - xb_train[None, :, :], axis=2)
    nn_indices = np.argmin(dists, axis=1)
    nn_pred = y_train[nn_indices]

    return {
        "xb_test": xb_test,
        "mean_pred": mean_pred,
        "nn_pred": nn_pred,
    }


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


def get_plot_metatada(percentiles_sample_map: dict[str, dict[str, list[Any]]], percentiles: np.ndarray):
    metadata = {}
    parameters_info = percentiles_sample_map.copy()
    for param in parameters_info:
        parameters_info[param]['values'] = [
            float(f"{v:.3f}") for v in parameters_info[param]['values']]
        parameters_info[param]['indices'] = [
            int(i) for i in parameters_info[param]['indices']]
    metadata = {
        'percentiles': [round(perc) for perc in percentiles],
        **parameters_info
    }
    return metadata


def save_plot_metadata(metadata: dict[str, Any], save_path: str):
    with open(f'{save_path}/plot_metadata.yaml', mode='w') as file:
        yaml.safe_dump(metadata, file, allow_unicode=True)


def _save_performance_and_timing_reports(
    data: dict[str, Any],
    data_cfg: DataConfig,
    test_cfg: TestConfig,
    plots_path: Path,
) -> None:
    metrics_dir = test_cfg.output_path / str(test_cfg.problem) / str(test_cfg.experiment_version) / "metrics"
    performance_plots_dir = plots_path / "performance_tracking"
    performance_plots_dir.mkdir(parents=True, exist_ok=True)

    y_true = data["ground_truths"]
    methods_pred = {
        "ml_model": data["predictions"],
        "mean_baseline": data["mean_baseline"],
        "nearest_neighbor_baseline": data["nearest_neighbor_baseline"],
    }
    method_errors = {
        name: _relative_error_per_sample(y_true, pred)
        for name, pred in methods_pred.items()
    }

    channel_labels = [str(label) for label in data_cfg.targets_labels]
    report = {
        "scope": "Kelvin operator surrogate (closed-form reference solver)",
        "test_samples": int(y_true.shape[0]),
        "methods": {
            name: {
                "overall": _summary(method_errors[name]),
                "by_channel": _per_channel_summary(y_true, pred, channel_labels=channel_labels),
            }
            for name, pred in methods_pred.items()
        },
    }

    model_mean = report["methods"]["ml_model"]["overall"]["mean"]
    mean_baseline_mean = report["methods"]["mean_baseline"]["overall"]["mean"]
    nn_baseline_mean = report["methods"]["nearest_neighbor_baseline"]["overall"]["mean"]
    report["improvement_factors"] = {
        "vs_mean_baseline": float(mean_baseline_mean / max(model_mean, 1e-14)),
        "vs_nearest_neighbor_baseline": float(nn_baseline_mean / max(model_mean, 1e-14)),
    }

    _save_yaml(metrics_dir / "baseline_performance_report.yaml", report)
    _save_metrics_table_csv(
        path=metrics_dir / "baseline_performance_table.csv",
        report=report,
        channel_labels=channel_labels,
    )

    _plot_method_error_distributions(
        method_errors=method_errors,
        output_path=performance_plots_dir / "error_distribution_by_method.png",
    )
    _plot_error_vs_inputs(
        xb_test=np.asarray(data["xb_test"]),
        method_errors=method_errors,
        input_labels=[str(lbl) for lbl in data_cfg.input_functions[:2]],
        output_path=performance_plots_dir / "error_vs_branch_parameters.png",
    )

    solver_timing = _get_solver_timing(data_cfg=data_cfg)
    inference_timing = _get_inference_timing(test_cfg=test_cfg)
    inference_total = inference_timing["inference_total_s"]
    inference_per_sample = None
    if inference_total is not None:
        inference_per_sample = inference_total / max(y_true.shape[0], 1)

    timing_report = {
        "reference_solver": {
            "total_s": solver_timing["solver_total_s"],
            "per_sample_s": solver_timing["solver_per_sample_s"],
            "solver_kind": "kelvin_closed_form",
        },
        "inference": {
            "total_s": inference_total,
            "per_sample_s": inference_per_sample,
            "test_samples": int(y_true.shape[0]),
        },
        "speedups": {},
    }

    solver_ps = solver_timing["solver_per_sample_s"]
    infer_ps = inference_per_sample
    if solver_ps is not None and infer_ps is not None:
        timing_report["speedups"]["per_sample_solver_over_inference"] = float(solver_ps / max(infer_ps, 1e-14))

    solver_total = solver_timing["solver_total_s"]
    if solver_total is not None and infer_ps is not None:
        n_total = int(data_cfg.data[data_cfg.features[0]].shape[0])
        infer_total_est = float(infer_ps * n_total)
        timing_report["inference"]["estimated_total_s_for_full_dataset"] = infer_total_est
        timing_report["speedups"]["estimated_full_dataset_solver_over_inference"] = float(
            solver_total / max(infer_total_est, 1e-14)
        )

    _save_yaml(metrics_dir / "timing_comparison_report.yaml", timing_report)
    _plot_timing_comparison(
        timing_report=timing_report,
        output_path=performance_plots_dir / "timing_comparison.png",
    )


def run_problem_specific_plotting(data: dict[str, Any], data_cfg: DataConfig, test_cfg: TestConfig):
    if test_cfg.problem is None:
        raise AttributeError(f"'Problem' attribute is missing.")
    plots_path = test_cfg.output_path / test_cfg.problem / \
        test_cfg.experiment_version / 'plots'
    plots_path.mkdir(parents=True, exist_ok=True)

    percentile_sample_by_parameter, percentiles = get_plotted_samples_indices(
        data_cfg=data_cfg, test_cfg=test_cfg)
    metadata = get_plot_metatada(percentile_sample_by_parameter, percentiles)

    save_plot_metadata(metadata, str(plots_path))

    plane_plots_path = plots_path / 'plane_plots'
    basis_plots_path = plots_path / 'basis_plots'
    coefficients_plots_path = plots_path / 'coefficients_plots'

    for path in [coefficients_plots_path, plane_plots_path, basis_plots_path]:
        path.mkdir(exist_ok=True)

    if test_cfg.config is None:
        raise AttributeError("Plotting config not found.")

    if test_cfg.config['plot_plane']:
        helper.plot_planes_helper(
            data=data,
            data_cfg=data_cfg,
            metadata=metadata,
            plot_path=plane_plots_path,
        )

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

    _save_performance_and_timing_reports(
        data=data,
        data_cfg=data_cfg,
        test_cfg=test_cfg,
        plots_path=plots_path,
    )


def plot_metrics(test_cfg: TestConfig, data_cfg: DataConfig):
    input_functions = ppr.get_input_functions(data_cfg)
    coordinates = ppr.get_coordinates(data_cfg)
    output_data = ppr.get_output_data(test_cfg)
    ground_truths = ppr.format_target(output_data[data_cfg.targets[0]], data_cfg)
    predictions = ppr.format_target(output_data['predictions'], data_cfg)
    basis = ppr.reshape_basis(output_data['trunk_output'], data_cfg, test_cfg)
    coefficients = ppr.reshape_coefficients(
        output_data['branch_output'], data_cfg, test_cfg)
    bias = ppr.format_bias(output_data['bias'], data_cfg, test_cfg)

    baselines = _compute_baselines(data_cfg=data_cfg)
    mean_baseline = ppr.format_target(baselines["mean_pred"], data_cfg)
    nearest_neighbor_baseline = ppr.format_target(baselines["nn_pred"], data_cfg)

    data = {
        'input_functions': input_functions,
        'coordinates': coordinates,
        'output_data': output_data,
        'ground_truths': ground_truths,
        'predictions': predictions,
        'coefficients': coefficients,
        'basis': basis,
        'bias': bias,
        'xb_test': baselines["xb_test"],
        'mean_baseline': mean_baseline,
        'nearest_neighbor_baseline': nearest_neighbor_baseline,
    }

    run_problem_specific_plotting(
        data=data, data_cfg=data_cfg, test_cfg=test_cfg)
