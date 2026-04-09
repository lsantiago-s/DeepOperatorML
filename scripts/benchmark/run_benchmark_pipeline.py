#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'PyYAML'. Install project dependencies before running benchmark scripts."
    ) from exc


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _latest_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    dirs = [p for p in path.iterdir() if p.is_dir()]
    if not dirs:
        raise RuntimeError(f"No directories found under: {path}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _run_cmd(cmd: list[str], dry_run: bool) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _is_fno_stub(repo_root: Path) -> bool:
    marker = repo_root / "src" / "modules" / "pipe" / "fno_train.py"
    if not marker.exists():
        return True
    content = marker.read_text(encoding="utf-8")
    return "NotImplementedError" in content


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark pipeline from manifest.")
    parser.add_argument(
        "--manifest",
        default="./configs/benchmarks/paper_ready_benchmark.yaml",
        help="Path to benchmark manifest YAML.",
    )
    parser.add_argument(
        "--stage",
        choices=["gen", "preprocess", "train", "test", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--problem",
        default=None,
        help="Optional single problem name to run. Defaults to all enabled problems in manifest.",
    )
    parser.add_argument(
        "--model-track",
        choices=["don", "fno"],
        default="don",
        help="Model track for train/test stages.",
    )
    parser.add_argument(
        "--freeze-dataset-version",
        action="store_true",
        help="After preprocessing, write latest processed dataset hash into config_experiment.yaml.",
    )
    parser.add_argument(
        "--freeze-experiment-version",
        action="store_true",
        help="After training, write latest output experiment folder into config_test.yaml.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = (repo_root / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    manifest = _load_yaml(manifest_path)
    problems_cfg = manifest.get("problems", {})
    train_configs = manifest.get("train_configs", {})
    if not isinstance(problems_cfg, dict):
        raise ValueError("Manifest 'problems' must be a mapping.")

    if args.model_track == "fno" and _is_fno_stub(repo_root):
        raise NotImplementedError(
            "FNO track requested, but src/modules/pipe/fno_train.py is scaffold-only."
        )

    selected_names = [args.problem] if args.problem else [
        name for name, cfg in problems_cfg.items() if bool(cfg.get("enabled", True))
    ]
    if not selected_names:
        raise RuntimeError("No enabled problems found in manifest.")

    do_gen = args.stage in {"gen", "all"}
    do_preprocess = args.stage in {"preprocess", "all"}
    do_train = args.stage in {"train", "all"}
    do_test = args.stage in {"test", "all"}

    train_config_rel = train_configs.get(args.model_track, "./configs/training/config_don_train.yaml")
    train_config_path = str((repo_root / str(train_config_rel)).resolve())

    for problem in selected_names:
        if problem not in problems_cfg:
            raise KeyError(f"Problem '{problem}' not found in manifest.")
        cfg = problems_cfg[problem]
        if not bool(cfg.get("enabled", True)):
            print(f"[SKIP] {problem} is disabled in manifest.")
            continue

        print(f"\n=== Problem: {problem} | Track: {args.model_track} ===")
        datagen_cfg = str((repo_root / str(cfg["datagen_config"])).resolve())
        preprocessing_cfg = str((repo_root / str(cfg["preprocessing_config"])).resolve())
        exp_cfg_path = (repo_root / str(cfg["experiment_config"])).resolve()
        test_cfg_path = (repo_root / str(cfg["test_config"])).resolve()
        exp_cfg_path_str = str(exp_cfg_path)
        test_cfg_path_str = str(test_cfg_path)

        if do_gen:
            _run_cmd(
                ["python3", "gen_data.py", "--problem", problem, "--config", datagen_cfg],
                dry_run=args.dry_run,
            )

        if do_preprocess:
            _run_cmd(
                [
                    "python3",
                    "preprocess_data.py",
                    "--problem",
                    problem,
                    "--config",
                    preprocessing_cfg,
                ],
                dry_run=args.dry_run,
            )
            if args.freeze_dataset_version:
                latest_dataset = _latest_dir(repo_root / "data" / "processed" / problem).name
                exp_cfg = _load_yaml(exp_cfg_path)
                exp_cfg["dataset_version"] = latest_dataset
                if args.model_track:
                    exp_cfg["model"] = args.model_track
                if not args.dry_run:
                    _dump_yaml(exp_cfg_path, exp_cfg)
                print(f"[freeze] {problem} dataset_version -> {latest_dataset}")

        if do_train:
            if not args.dry_run and exp_cfg_path.exists():
                exp_cfg = _load_yaml(exp_cfg_path)
                exp_cfg["model"] = args.model_track
                _dump_yaml(exp_cfg_path, exp_cfg)
            _run_cmd(
                [
                    "python3",
                    "main.py",
                    "--problem",
                    problem,
                    "--train-config-path",
                    train_config_path,
                    "--experiment-config-path",
                    exp_cfg_path_str,
                    "--test-config-path",
                    test_cfg_path_str,
                ],
                dry_run=args.dry_run,
            )
            if args.freeze_experiment_version:
                latest_exp = _latest_dir(repo_root / "output" / problem).name
                test_cfg = _load_yaml(test_cfg_path)
                test_cfg["experiment_version"] = latest_exp
                if not args.dry_run:
                    _dump_yaml(test_cfg_path, test_cfg)
                print(f"[freeze] {problem} experiment_version -> {latest_exp}")

        if do_test:
            _run_cmd(
                [
                    "python3",
                    "main.py",
                    "--problem",
                    problem,
                    "--test",
                    "--experiment-config-path",
                    exp_cfg_path_str,
                    "--test-config-path",
                    test_cfg_path_str,
                ],
                dry_run=args.dry_run,
            )

    print("\nBenchmark pipeline command sequence completed.")


if __name__ == "__main__":
    main()
