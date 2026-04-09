#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'numpy'. Install project dependencies before running benchmark aggregation."
    ) from exc

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'PyYAML'. Install project dependencies before running benchmark aggregation."
    ) from exc


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _validate_shapes(output_data_path: Path, target_key: str = "g_u") -> list[str]:
    issues: list[str] = []
    if not output_data_path.exists():
        return [f"Missing output data file: {output_data_path}"]
    data = dict(np.load(output_data_path))
    if "predictions" not in data:
        return [f"Missing predictions in {output_data_path}"]
    if target_key not in data:
        return [f"Missing target key '{target_key}' in {output_data_path}"]

    pred = np.asarray(data["predictions"])
    truth = np.asarray(data[target_key])
    if pred.ndim != truth.ndim:
        issues.append(
            f"Prediction ndim ({pred.ndim}) differs from truth ndim ({truth.ndim}) in {output_data_path}"
        )
        return issues
    if pred.shape[1:] != truth.shape[1:]:
        issues.append(
            f"Prediction shape tail {pred.shape[1:]} differs from truth tail {truth.shape[1:]} in {output_data_path}"
        )
    if not (pred.shape[0] == truth.shape[0] or pred.shape[0] <= truth.shape[0]):
        issues.append(
            f"Prediction rows {pred.shape[0]} not compatible with truth rows {truth.shape[0]} in {output_data_path}"
        )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark reports from manifest.")
    parser.add_argument(
        "--manifest",
        default="./configs/benchmarks/paper_ready_benchmark.yaml",
        help="Path to benchmark manifest.",
    )
    parser.add_argument(
        "--track",
        choices=["don", "fno", "all"],
        default="don",
        help="Model track(s) to aggregate.",
    )
    parser.add_argument(
        "--output-dir",
        default="./output/benchmark_reports",
        help="Directory for aggregated benchmark outputs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if validation issues are found.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = (repo_root / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    manifest = _load_yaml(manifest_path)
    benchmark_name = str(manifest.get("name", manifest_path.stem))

    out_root = (repo_root / args.output_dir / benchmark_name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    required_contract = manifest.get("required_contract", {})
    required_metrics_files = _as_list(required_contract.get("metrics_files"))
    required_plot_globs = _as_list(required_contract.get("plot_globs"))
    defaults = manifest.get("defaults", {})
    repro_tol = float(defaults.get("reproducibility_tolerance", 1e-4))

    problems_cfg = manifest.get("problems", {})
    if not isinstance(problems_cfg, dict):
        raise ValueError("Manifest 'problems' must be a mapping.")

    selected_tracks = ["don", "fno"] if args.track == "all" else [args.track]

    error_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for problem, cfg in problems_cfg.items():
        if not bool(cfg.get("enabled", True)):
            continue

        exp_cfg_path = (repo_root / str(cfg["experiment_config"])).resolve()
        local_exp_cfg = _load_yaml(exp_cfg_path) if exp_cfg_path.exists() else {}
        local_dataset_version = str(local_exp_cfg.get("dataset_version", ""))
        canonical_dataset_version = str(cfg.get("canonical_dataset_version", ""))

        for track in selected_tracks:
            exp_versions = _as_list(cfg.get("canonical_experiments", {}).get(track))
            exp_versions = [str(v) for v in exp_versions if v not in (None, "", "null")]
            if not exp_versions:
                issues.append(
                    {
                        "problem": problem,
                        "track": track,
                        "severity": "warning",
                        "message": "No canonical experiment versions configured.",
                    }
                )
                continue

            means_for_repro: list[float] = []

            for exp_version in exp_versions:
                run_root = repo_root / "output" / problem / exp_version
                metrics_root = run_root / "metrics"
                plots_root = run_root / "plots"
                aux_root = run_root / "aux"

                missing_metrics = [name for name in required_metrics_files if not (metrics_root / name).exists()]
                if missing_metrics:
                    issues.append(
                        {
                            "problem": problem,
                            "track": track,
                            "experiment_version": exp_version,
                            "severity": "error",
                            "message": f"Missing required metrics files: {missing_metrics}",
                        }
                    )
                    continue

                missing_plots: list[str] = []
                for pattern in required_plot_globs:
                    if len(list(run_root.glob(pattern))) == 0:
                        missing_plots.append(pattern)
                if missing_plots:
                    issues.append(
                        {
                            "problem": problem,
                            "track": track,
                            "experiment_version": exp_version,
                            "severity": "error",
                            "message": f"Missing required plot patterns: {missing_plots}",
                        }
                    )

                baseline_report = _load_yaml(metrics_root / "baseline_performance_report.yaml")
                timing_report = _load_yaml(metrics_root / "timing_comparison_report.yaml")
                ml_overall = baseline_report.get("methods", {}).get("ml_model", {}).get("overall", {})
                ml_by_channel = baseline_report.get("methods", {}).get("ml_model", {}).get("by_channel", {})

                error_rows.append(
                    {
                        "problem": problem,
                        "track": track,
                        "experiment_version": exp_version,
                        "dataset_version_expected": canonical_dataset_version,
                        "dataset_version_config": local_dataset_version,
                        "mean": ml_overall.get("mean"),
                        "median": ml_overall.get("median"),
                        "p90": ml_overall.get("p90"),
                        "max": ml_overall.get("max"),
                        "channels": "; ".join(
                            f"{k}:{v.get('mean')}" for k, v in ml_by_channel.items()
                        ),
                    }
                )
                if ml_overall.get("mean") is not None:
                    means_for_repro.append(float(ml_overall["mean"]))

                speedups = timing_report.get("speedups", {})
                timing_rows.append(
                    {
                        "problem": problem,
                        "track": track,
                        "experiment_version": exp_version,
                        "reference_total_s": timing_report.get("reference_solver", {}).get("total_s")
                        or timing_report.get("integration", {}).get("total_s"),
                        "reference_per_sample_s": timing_report.get("reference_solver", {}).get("per_sample_s")
                        or timing_report.get("integration", {}).get("per_sample_s"),
                        "inference_total_s": timing_report.get("inference", {}).get("total_s"),
                        "inference_per_sample_s": timing_report.get("inference", {}).get("per_sample_s"),
                        "speedup_primary": speedups.get("per_sample_solver_over_inference")
                        or speedups.get("per_sample_integration_over_inference"),
                        "speedup_full_dataset": speedups.get("estimated_full_dataset_solver_over_inference")
                        or speedups.get("estimated_full_dataset_integration_over_inference"),
                    }
                )

                shape_issues = _validate_shapes(aux_root / "output_data.npz", target_key="g_u")
                for msg in shape_issues:
                    issues.append(
                        {
                            "problem": problem,
                            "track": track,
                            "experiment_version": exp_version,
                            "severity": "error",
                            "message": msg,
                        }
                    )

                exp_cfg_in_output_path = run_root / "experiment_config.yaml"
                if exp_cfg_in_output_path.exists():
                    exp_cfg_in_output = _load_yaml(exp_cfg_in_output_path)
                    run_dataset = str(exp_cfg_in_output.get("dataset_version", ""))
                    if canonical_dataset_version and run_dataset != canonical_dataset_version:
                        issues.append(
                            {
                                "problem": problem,
                                "track": track,
                                "experiment_version": exp_version,
                                "severity": "error",
                                "message": (
                                    "Dataset version mismatch between run and manifest "
                                    f"({run_dataset} != {canonical_dataset_version})"
                                ),
                            }
                        )
                    if local_dataset_version and run_dataset != local_dataset_version:
                        issues.append(
                            {
                                "problem": problem,
                                "track": track,
                                "experiment_version": exp_version,
                                "severity": "error",
                                "message": (
                                    "Dataset version mismatch between run and config_experiment "
                                    f"({run_dataset} != {local_dataset_version})"
                                ),
                            }
                        )

                for png in sorted(plots_root.rglob("*.png")):
                    rel = png.relative_to(repo_root)
                    figure_rows.append(
                        {
                            "problem": problem,
                            "track": track,
                            "experiment_version": exp_version,
                            "path": str(rel),
                            "title_hint": png.stem.replace("_", " "),
                        }
                    )

                bundle_root = out_root / "bundles" / problem / track / exp_version
                bundle_root.mkdir(parents=True, exist_ok=True)
                for metric_file in required_metrics_files:
                    _copy_if_exists(metrics_root / metric_file, bundle_root / "metrics" / metric_file)
                _copy_if_exists(metrics_root / "test_metrics.yaml", bundle_root / "metrics" / "test_metrics.yaml")
                _copy_if_exists(metrics_root / "test_time.yaml", bundle_root / "metrics" / "test_time.yaml")
                if (plots_root / "performance_tracking").exists():
                    shutil.copytree(
                        plots_root / "performance_tracking",
                        bundle_root / "plots" / "performance_tracking",
                        dirs_exist_ok=True,
                    )
                if (plots_root / "operator_metrics").exists():
                    shutil.copytree(
                        plots_root / "operator_metrics",
                        bundle_root / "plots" / "operator_metrics",
                        dirs_exist_ok=True,
                    )
                if exp_cfg_in_output_path.exists():
                    _copy_if_exists(exp_cfg_in_output_path, bundle_root / "configs" / "experiment_config.yaml")
                for cfg_name in ["datagen_config", "preprocessing_config", "experiment_config", "test_config"]:
                    cfg_path = cfg.get(cfg_name)
                    if cfg_path:
                        src = (repo_root / str(cfg_path)).resolve()
                        _copy_if_exists(src, bundle_root / "configs" / Path(cfg_path).name)

            if len(means_for_repro) >= 2:
                diff = float(np.max(np.abs(np.asarray(means_for_repro) - means_for_repro[0])))
                if diff > repro_tol:
                    issues.append(
                        {
                            "problem": problem,
                            "track": track,
                            "severity": "error",
                            "message": (
                                "Reproducibility gate failed: "
                                f"max mean error drift={diff:.3e} > tolerance={repro_tol:.3e}"
                            ),
                        }
                    )

    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            with open(path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(["empty"])
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    _write_csv(out_root / "global_error_table.csv", error_rows)
    _write_csv(out_root / "global_timing_table.csv", timing_rows)
    _write_csv(out_root / "figure_index.csv", figure_rows)

    validation_report = {
        "manifest": str(manifest_path),
        "benchmark_name": benchmark_name,
        "strict": bool(args.strict),
        "num_error_rows": len(error_rows),
        "num_timing_rows": len(timing_rows),
        "num_figures": len(figure_rows),
        "num_issues": len(issues),
        "issues": issues,
        "status": "failed" if any(i.get("severity") == "error" for i in issues) else "passed",
    }
    _dump_yaml(out_root / "validation_report.yaml", validation_report)

    print(f"Wrote benchmark aggregation artifacts to: {out_root}")
    if args.strict and validation_report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
