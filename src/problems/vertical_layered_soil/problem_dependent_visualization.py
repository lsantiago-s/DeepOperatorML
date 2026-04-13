from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.modules.models.config import DataConfig, TestConfig
from src.problems.vertical_layered_soil import plot_helper
from src.problems.vertical_layered_soil import postprocessing as ppr

logger = logging.getLogger(__name__)


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


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


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


def _plot_method_error_distributions(method_errors: dict[str, np.ndarray], output_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5), constrained_layout=True)
    labels = list(method_errors.keys())
    vals = [method_errors[name] for name in labels]
    ax.boxplot(vals, labels=labels, showfliers=False)
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Error distribution on test split")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_error_vs_frequency(
    axis_test: np.ndarray,
    axis_label: str,
    method_errors: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    order = np.argsort(axis_test)
    x_sorted = axis_test[order]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5), constrained_layout=True)
    for method_name, errs in method_errors.items():
        ax.plot(x_sorted, errs[order], lw=1.2, label=method_name)
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Relative L2 error")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title(f"Error vs {axis_label}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_timing_comparison(timing_report: dict[str, Any], output_path: Path) -> None:
    solver = timing_report["reference_solver"].get("per_sample_s")
    infer = timing_report["inference"].get("per_sample_s")
    if solver is None or infer is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    labels = ["Reference Solver", "ML Inference"]
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


def _get_solver_timing(data_cfg: DataConfig) -> dict[str, float | None]:
    meta_path = Path(data_cfg.raw_metadata_path)
    if not meta_path.exists():
        return {"solver_total_s": None, "solver_per_sample_s": None}

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    timing_breakdown = meta.get("timing_breakdown", {})
    total = _safe_float(timing_breakdown.get("direct_solver_total_s"))
    per_sample = _safe_float(timing_breakdown.get("direct_solver_per_sample_s"))
    if total is None:
        total = _safe_float(meta.get("runtime_s"))
    if total is None:
        runtime_ms = _safe_float(meta.get("runtime_ms"))
        if runtime_ms is not None:
            if ("runtime_s" in meta) or ("timing_breakdown" in meta):
                total = runtime_ms / 1e3
            else:
                total = runtime_ms
    if per_sample is None and total is not None:
        n_total = int(data_cfg.data[data_cfg.features[0]].shape[0])
        per_sample = total / max(n_total, 1)
    return {"solver_total_s": total, "solver_per_sample_s": per_sample}


def _get_inference_timing(test_cfg: TestConfig) -> dict[str, float | None]:
    if test_cfg.problem is None:
        return {"inference_total_s": None}
    time_path = (
        test_cfg.output_path
        / test_cfg.problem
        / test_cfg.experiment_version
        / "metrics"
        / "test_time.yaml"
    )
    if not time_path.exists():
        return {"inference_total_s": None}
    with open(time_path, "r", encoding="utf-8") as f:
        test_time = yaml.safe_load(f) or {}
    return {"inference_total_s": _safe_float(test_time.get("inference_time"))}


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
    return {"xb_test": xb_test, "mean_pred": mean_pred, "nn_pred": nn_pred}


def _save_common_contract_reports(
    test_cfg: TestConfig,
    data_cfg: DataConfig,
    plots_path: Path,
    truth_test_channels: np.ndarray,
    pred_test_channels: np.ndarray,
    truth_test_full: np.ndarray,
    pred_test_full: np.ndarray,
    axis_test: np.ndarray,
    axis_label: str,
    mean_test_channels: np.ndarray,
    nn_test_channels: np.ndarray,
    channel_labels: list[str],
) -> None:
    metrics_dir = test_cfg.output_path / str(test_cfg.problem) / str(test_cfg.experiment_version) / "metrics"
    performance_plots_dir = plots_path / "performance_tracking"
    performance_plots_dir.mkdir(parents=True, exist_ok=True)

    method_pred = {
        "ml_model": pred_test_channels,
        "mean_baseline": mean_test_channels,
        "nearest_neighbor_baseline": nn_test_channels,
    }
    method_errors = {name: _relative_error_per_sample(truth_test_channels, pred) for name, pred in method_pred.items()}

    report = {
        "scope": "vertical layered full influence matrix surrogate",
        "test_samples": int(truth_test_channels.shape[0]),
        "frequency_axis_label": axis_label,
        "methods": {},
    }
    for name, pred in method_pred.items():
        by_channel = {}
        for ch, label in enumerate(channel_labels):
            by_channel[label] = _summary(
                _relative_error_per_sample(truth_test_channels[..., ch], pred[..., ch])
            )
        report["methods"][name] = {
            "overall": _summary(method_errors[name]),
            "by_channel": by_channel,
        }

    model_mean = report["methods"]["ml_model"]["overall"]["mean"]
    mean_baseline_mean = report["methods"]["mean_baseline"]["overall"]["mean"]
    nn_baseline_mean = report["methods"]["nearest_neighbor_baseline"]["overall"]["mean"]
    report["improvement_factors"] = {
        "vs_mean_baseline": float(mean_baseline_mean / max(model_mean, 1e-14)),
        "vs_nearest_neighbor_baseline": float(nn_baseline_mean / max(model_mean, 1e-14)),
    }

    eps_f = _relative_error_per_sample(truth_test_full, pred_test_full)
    report["matrix_level_error_frobenius"] = _summary(eps_f)

    _save_yaml(metrics_dir / "baseline_performance_report.yaml", report)
    _save_metrics_table_csv(metrics_dir / "baseline_performance_table.csv", report, channel_labels)

    _plot_method_error_distributions(
        method_errors=method_errors,
        output_path=performance_plots_dir / "error_distribution_by_method.png",
    )
    _plot_error_vs_frequency(
        axis_test=axis_test,
        axis_label=axis_label,
        method_errors=method_errors,
        output_path=performance_plots_dir / "error_vs_frequency.png",
    )

    solver_timing = _get_solver_timing(data_cfg=data_cfg)
    inference_timing = _get_inference_timing(test_cfg=test_cfg)
    inference_total = inference_timing["inference_total_s"]
    inference_per_sample = None
    if inference_total is not None:
        inference_per_sample = inference_total / max(truth_test_channels.shape[0], 1)

    timing_report = {
        "reference_solver": {
            "total_s": solver_timing["solver_total_s"],
            "per_sample_s": solver_timing["solver_per_sample_s"],
            "solver_kind": "legacy_fortran_vertical_layered_soil",
        },
        "inference": {
            "total_s": inference_total,
            "per_sample_s": inference_per_sample,
            "test_samples": int(truth_test_channels.shape[0]),
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


def plot_metrics(test_cfg: TestConfig, data_cfg: DataConfig) -> None:
    if test_cfg.problem is None:
        raise ValueError("TestConfig.problem must be set before plotting.")
    if test_cfg.config is None:
        raise ValueError("Missing test config dictionary in TestConfig.config.")

    raw_data = ppr.load_raw_data(data_cfg)
    if "properties" not in raw_data:
        raise KeyError("Raw dataset must contain 'properties'.")
    if "a0" not in raw_data and "omega" not in raw_data:
        raise KeyError("Raw dataset must contain either 'a0' or 'omega'.")

    output_data = ppr.load_output_data(test_cfg)
    truth_test, pred_test, test_indices = ppr.get_truth_pred_complex(output_data=output_data, data_cfg=data_cfg)
    truth_test_channels = ppr.reshape_channels(truth_test)
    pred_test_channels = ppr.reshape_channels(pred_test)
    truth_test_full = ppr.channels_to_full_matrix(truth_test_channels)
    pred_test_full = ppr.channels_to_full_matrix(pred_test_channels)

    baselines = _compute_baselines(data_cfg=data_cfg)
    mean_test = ppr.to_complex_channels(baselines["mean_pred"])
    nn_test = ppr.to_complex_channels(baselines["nn_pred"])
    mean_test_channels = ppr.reshape_channels(mean_test)
    nn_test_channels = ppr.reshape_channels(nn_test)

    plots_path = (
        Path(test_cfg.output_path)
        / test_cfg.problem
        / str(test_cfg.experiment_version)
        / "plots"
    )
    plots_path.mkdir(parents=True, exist_ok=True)

    b_value = 0.0
    axis_label = "a0"
    channel_names: list[str] = ["Uxx", "Uxz", "Uzx", "Uzz"]
    try:
        with open(data_cfg.raw_metadata_path, "r", encoding="utf-8") as f:
            raw_meta = yaml.safe_load(f)
        b_value = float(raw_meta.get("B", 0.0))
        axis_label = str(raw_meta.get("frequency_axis_label", axis_label))
        if isinstance(raw_meta.get("g_u_channels"), list) and len(raw_meta["g_u_channels"]) > 0:
            channel_names = [str(c) for c in raw_meta["g_u_channels"]]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse B from raw metadata (%s). Using B=0.0", exc)

    axis_all = np.asarray(raw_data["a0"] if "a0" in raw_data else raw_data["omega"], dtype=float)

    logger.info("Generating vertical layered paper-style plots at %s", plots_path)
    plot_helper.run_all_multilayer_plots(
        plots_path=plots_path,
        properties_all=np.asarray(raw_data["properties"], dtype=float),
        profiles_all=(
            np.asarray(raw_data["profiles"], dtype=float)
            if "profiles" in raw_data
            else None
        ),
        z_grid=(
            np.asarray(raw_data["z"], dtype=float)
            if "z" in raw_data
            else None
        ),
        omega_all=axis_all,
        raw_u_all=(
            ppr.channels_to_full_matrix(ppr.reshape_channels(np.asarray(raw_data["g_u"])))
            if "g_u" in raw_data
            else None
        ),
        case_labels_all=np.asarray(raw_data["paper_case_label"]) if "paper_case_label" in raw_data else None,
        truth_test_matrix=truth_test_full,
        pred_test_matrix=pred_test_full,
        test_indices=np.asarray(test_indices, dtype=int),
        case_labels_test=(
            np.asarray(raw_data["paper_case_label"])[np.asarray(test_indices, dtype=int)]
            if "paper_case_label" in raw_data
            else None
        ),
        b_value=b_value,
        config={**test_cfg.config, "frequency_axis_label": axis_label},
        channel_names=channel_names,
    )

    axis_test = axis_all[np.asarray(test_indices, dtype=int)]
    _save_common_contract_reports(
        test_cfg=test_cfg,
        data_cfg=data_cfg,
        plots_path=plots_path,
        truth_test_channels=truth_test_channels,
        pred_test_channels=pred_test_channels,
        truth_test_full=truth_test_full,
        pred_test_full=pred_test_full,
        axis_test=axis_test,
        axis_label=axis_label,
        mean_test_channels=mean_test_channels,
        nn_test_channels=nn_test_channels,
        channel_labels=channel_names,
    )
    logger.info("Finished vertical layered paper-style plotting.")
