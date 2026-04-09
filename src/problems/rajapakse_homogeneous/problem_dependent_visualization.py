from __future__ import annotations
import csv
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Any

from src.modules.models.config import DataConfig, TestConfig
from src.problems.rajapakse_homogeneous import postprocessing as ppr
from src.problems.rajapakse_homogeneous import plot_helper as helper

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
        if "delta" in _normalize_label(param):
            return str(param)

    return str(data_cfg.input_functions[0])


def get_plotted_samples_indices(
    data_cfg: DataConfig,
    test_cfg: TestConfig,
) -> tuple[dict[str, dict[str, list[np.intp | float]]], np.ndarray]:
    if test_cfg.config is None:
        raise KeyError("Plot config not found")

    percentile_sample_by_parameter = {}
    chosen_percentiles = min(
        test_cfg.config["percentiles"],
        len(data_cfg.split_indices[data_cfg.features[0] + "_test"]),
    )
    percentiles = np.linspace(0, 100, num=chosen_percentiles)

    xb_test = data_cfg.data[data_cfg.features[0]][data_cfg.split_indices[data_cfg.features[0] + "_test"]]

    for pos, parameter in enumerate(data_cfg.input_functions):
        indices = []
        targets = []
        for perc in percentiles:
            target = np.percentile(xb_test[:, pos], perc)
            idx = np.argmin(np.abs(xb_test[:, pos] - target))
            indices.append(idx)
            targets.append(target)
        percentile_sample_by_parameter[parameter] = {
            "indices": indices,
            "values": targets,
        }

    return percentile_sample_by_parameter, percentiles


def get_plot_metadata(
    percentiles_sample_map: dict[str, dict[str, list[Any]]],
    percentiles: np.ndarray,
    reference_parameter: str,
) -> dict[str, Any]:
    parameters_info = percentiles_sample_map.copy()
    for param in parameters_info:
        parameters_info[param]["values"] = [float(f"{v:.4e}") for v in parameters_info[param]["values"]]
        parameters_info[param]["indices"] = [int(i) for i in parameters_info[param]["indices"]]
    return {
        "percentiles": [round(float(perc), 3) for perc in percentiles],
        "reference_parameter": reference_parameter,
        "reference_samples": parameters_info[reference_parameter],
        "all_parameter_samples": parameters_info,
    }


def save_plot_metadata(metadata: dict[str, Any], save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "plot_metadata.yaml", mode="w", encoding="utf-8") as file:
        yaml.safe_dump(metadata, file, allow_unicode=True, sort_keys=False)


def _relative_l2_by_sample(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_flat = y_true.reshape(y_true.shape[0], -1)
    pred_flat = y_pred.reshape(y_pred.shape[0], -1)
    num = np.linalg.norm(pred_flat - true_flat, axis=1)
    den = np.linalg.norm(true_flat, axis=1) + 1e-14
    return num / den


def _nearest_neighbor_baseline(
    xb_test: np.ndarray,
    xb_train: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    x_mean = xb_train.mean(axis=0, keepdims=True)
    x_std = xb_train.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-12, 1.0, x_std)

    train_n = (xb_train - x_mean) / x_std
    test_n = (xb_test - x_mean) / x_std

    d2 = np.sum((test_n[:, None, :] - train_n[None, :, :]) ** 2, axis=2)
    nn_idx = np.argmin(d2, axis=1)
    return y_train[nn_idx]


def _summarize_errors(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "max": float(np.max(errors)),
    }


def _write_metrics_table(metrics: dict[str, dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    table_rows = []
    for model_name, summary in metrics.items():
        table_rows.append(
            {
                "model": model_name,
                "mean_rel_l2": summary["mean"],
                "median_rel_l2": summary["median"],
                "p90_rel_l2": summary["p90"],
                "max_rel_l2": summary["max"],
            }
        )

    csv_path = output_dir / "metrics_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)


def _read_raw_generation_timing(data_cfg: DataConfig) -> dict[str, float | int | None]:
    with open(data_cfg.raw_metadata_path, "r", encoding="utf-8") as file:
        raw_metadata = yaml.safe_load(file)

    runtime_s = raw_metadata.get("runtime_s", None)
    if runtime_s is None and "runtime_ms" in raw_metadata:
        runtime_s = float(raw_metadata["runtime_ms"]) / 1e3
    elif runtime_s is not None:
        runtime_s = float(runtime_s)

    integration_per_sample = raw_metadata.get("integration_time_per_sample_s", None)
    if integration_per_sample is not None:
        integration_per_sample = float(integration_per_sample)

    n_samples = None
    try:
        n_samples = int(raw_metadata["displacement_statistics"][data_cfg.targets[0]]["shape"][0])
    except Exception:
        pass

    if integration_per_sample is None and runtime_s is not None and n_samples is not None and n_samples > 0:
        integration_per_sample = runtime_s / n_samples

    return {
        "integration_total_s": runtime_s,
        "integration_per_sample_s": integration_per_sample,
        "n_generated_samples": n_samples,
    }


def _read_inference_timing(test_cfg: TestConfig) -> float | None:
    if test_cfg.problem is None:
        return None
    timing_path = (
        test_cfg.output_path
        / test_cfg.problem
        / test_cfg.experiment_version
        / "metrics"
        / "test_time.yaml"
    )
    if not timing_path.exists():
        return None
    with open(timing_path, "r", encoding="utf-8") as file:
        timing = yaml.safe_load(file)
    value = timing.get("inference_time", None)
    return None if value is None else float(value)


def run_problem_specific_plotting(data: dict[str, Any], data_cfg: DataConfig, test_cfg: TestConfig):
    if test_cfg.problem is None:
        raise AttributeError("'Problem' attribute is missing.")
    if test_cfg.config is None:
        raise AttributeError("Plotting config not found.")

    plots_path = test_cfg.output_path / test_cfg.problem / test_cfg.experiment_version / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    percentile_sample_by_parameter, percentiles = get_plotted_samples_indices(data_cfg=data_cfg, test_cfg=test_cfg)
    reference_parameter = _resolve_reference_parameter(data_cfg=data_cfg, test_cfg=test_cfg)
    metadata = get_plot_metadata(
        percentiles_sample_map=percentile_sample_by_parameter,
        percentiles=percentiles,
        reference_parameter=reference_parameter,
    )
    save_plot_metadata(metadata=metadata, save_path=plots_path)

    plane_plots_path = plots_path / "plane_plots"
    axis_plots_path = plots_path / "axis_plots"
    basis_plots_path = plots_path / "basis_plots"
    coefficients_plots_path = plots_path / "coefficients_plots"
    reports_path = plots_path / "reports"

    for path in [plane_plots_path, axis_plots_path, basis_plots_path, coefficients_plots_path, reports_path]:
        path.mkdir(exist_ok=True)

    if test_cfg.config.get("plot_plane", True):
        helper.plot_planes_helper(data=data, data_cfg=data_cfg, metadata=metadata, plot_path=plane_plots_path)

    if test_cfg.config.get("plot_axis", True):
        helper.plot_axis_helper(data=data, data_cfg=data_cfg, metadata=metadata, plot_path=axis_plots_path)

    if test_cfg.config.get("plot_basis", True):
        helper.plot_basis_helper(data=data, data_cfg=data_cfg, plot_path=basis_plots_path)

    if test_cfg.config.get("plot_coefficients", True):
        helper.plot_coefficients_helper(data=data, data_cfg=data_cfg, metadata=metadata, plot_path=coefficients_plots_path)

    if test_cfg.config.get("plot_coefficients_mean", True):
        helper.plot_coefficients_mean_helper(data=data, data_cfg=data_cfg, test_cfg=test_cfg, plot_path=coefficients_plots_path)

    helper.plot_error_vs_parameter(
        parameter=data["reference_parameter_values"],
        errors={
            "DeepONet": data["errors_model"],
            "Nearest-neighbor baseline": data["errors_nn"],
            "Mean baseline": data["errors_mean"],
        },
        parameter_label=str(reference_parameter),
        out_path=reports_path / "error_vs_reference_parameter.png",
    )

    helper.plot_parameter_coverage(
        xb=data["xb_all"],
        labels=data_cfg.input_functions,
        out_path=reports_path / "input_parameter_coverage.png",
    )

    timing_labels = []
    timing_values = []
    if data["timing"]["integration_total_s"] is not None:
        timing_labels.append("integration total")
        timing_values.append(float(data["timing"]["integration_total_s"]))
    if data["timing"]["inference_total_s"] is not None:
        timing_labels.append("inference total (test)")
        timing_values.append(float(data["timing"]["inference_total_s"]))
    if data["timing"]["inference_equiv_dataset_s"] is not None:
        timing_labels.append("inference (dataset-equivalent)")
        timing_values.append(float(data["timing"]["inference_equiv_dataset_s"]))
    if timing_labels:
        helper.plot_timing_bars(
            labels=timing_labels,
            times_s=timing_values,
            out_path=reports_path / "timing_comparison.png",
        )

    with open(reports_path / "performance_report.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(data["report"], file, sort_keys=False, allow_unicode=True)

    _write_metrics_table(metrics=data["report"]["performance_summary"], output_dir=reports_path)


def plot_metrics(test_cfg: TestConfig, data_cfg: DataConfig):
    input_functions = ppr.get_input_functions(data_cfg)
    coordinates = ppr.get_coordinates(data_cfg)
    output_data = ppr.get_output_data(test_cfg)

    ground_truths = ppr.format_target(output_data[data_cfg.targets[0]], data_cfg)
    predictions = ppr.format_target(output_data["predictions"], data_cfg)
    coefficients = ppr.reshape_coefficients(output_data["branch_output"], data_cfg, test_cfg)
    basis = ppr.reshape_basis(output_data["trunk_output"], data_cfg, test_cfg)
    bias = ppr.format_bias(output_data["bias"], data_cfg, test_cfg)

    y_test_flat = ppr.get_flat_test_target(output_data=output_data, data_cfg=data_cfg)
    y_pred_flat = output_data["predictions"]

    xb_test = data_cfg.data[data_cfg.features[0]][data_cfg.split_indices["xb_test"]]
    xb_train = data_cfg.data[data_cfg.features[0]][data_cfg.split_indices["xb_train"]]
    y_train_flat = data_cfg.data[data_cfg.targets[0]][data_cfg.split_indices["xb_train"]]

    y_mean = np.repeat(y_train_flat.mean(axis=0, keepdims=True), repeats=len(xb_test), axis=0)
    y_nn = _nearest_neighbor_baseline(xb_test=xb_test, xb_train=xb_train, y_train=y_train_flat)

    errors_model = _relative_l2_by_sample(y_true=y_test_flat, y_pred=y_pred_flat)
    errors_mean = _relative_l2_by_sample(y_true=y_test_flat, y_pred=y_mean)
    errors_nn = _relative_l2_by_sample(y_true=y_test_flat, y_pred=y_nn)

    performance_summary = {
        "DeepONet": _summarize_errors(errors_model),
        "NearestNeighbor": _summarize_errors(errors_nn),
        "MeanField": _summarize_errors(errors_mean),
    }

    reference_parameter = _resolve_reference_parameter(data_cfg=data_cfg, test_cfg=test_cfg)
    ref_index = data_cfg.input_functions.index(reference_parameter)

    timing_raw = _read_raw_generation_timing(data_cfg=data_cfg)
    inference_total = _read_inference_timing(test_cfg=test_cfg)
    n_test = int(len(xb_test))
    n_generated = timing_raw["n_generated_samples"]

    inference_per_sample = None
    if inference_total is not None and n_test > 0:
        inference_per_sample = inference_total / n_test

    inference_equiv_dataset = None
    if inference_per_sample is not None and n_generated is not None:
        inference_equiv_dataset = inference_per_sample * n_generated

    speedup_per_sample = None
    if timing_raw["integration_per_sample_s"] is not None and inference_per_sample is not None:
        speedup_per_sample = float(timing_raw["integration_per_sample_s"]) / inference_per_sample

    speedup_dataset_equiv = None
    if timing_raw["integration_total_s"] is not None and inference_equiv_dataset is not None and inference_equiv_dataset > 0:
        speedup_dataset_equiv = float(timing_raw["integration_total_s"]) / inference_equiv_dataset

    report = {
        "objective": "Surrogate for homogeneous half-space Green-function evaluation (influence-function operator)",
        "formulation_alignment": {
            "reference_parameter": reference_parameter,
            "input_functions": data_cfg.input_functions,
            "coordinates": data_cfg.coordinates,
            "targets": data_cfg.targets_labels,
        },
        "performance_summary": performance_summary,
        "improvement_over_baselines": {
            "vs_nearest_neighbor_mean_error_ratio": float(performance_summary["NearestNeighbor"]["mean"] / max(performance_summary["DeepONet"]["mean"], 1e-14)),
            "vs_mean_field_mean_error_ratio": float(performance_summary["MeanField"]["mean"] / max(performance_summary["DeepONet"]["mean"], 1e-14)),
        },
        "timing": {
            "integration_total_s": timing_raw["integration_total_s"],
            "integration_per_sample_s": timing_raw["integration_per_sample_s"],
            "inference_total_s_test_split": inference_total,
            "inference_per_sample_s": inference_per_sample,
            "inference_equivalent_dataset_s": inference_equiv_dataset,
            "speedup_per_sample_integration_over_inference": speedup_per_sample,
            "speedup_dataset_total_integration_over_inference": speedup_dataset_equiv,
            "n_generated_samples": n_generated,
            "n_test_samples": n_test,
        },
    }

    data = {
        "input_functions": input_functions,
        "coordinates": coordinates,
        "output_data": output_data,
        "ground_truths": ground_truths,
        "predictions": predictions,
        "coefficients": coefficients,
        "basis": basis,
        "bias": bias,
        "xb_test": xb_test,
        "xb_all": data_cfg.data[data_cfg.features[0]],
        "errors_model": errors_model,
        "errors_nn": errors_nn,
        "errors_mean": errors_mean,
        "reference_parameter_values": xb_test[:, ref_index],
        "report": report,
        "timing": {
            "integration_total_s": timing_raw["integration_total_s"],
            "inference_total_s": inference_total,
            "inference_equiv_dataset_s": inference_equiv_dataset,
        },
    }

    run_problem_specific_plotting(data=data, data_cfg=data_cfg, test_cfg=test_cfg)
