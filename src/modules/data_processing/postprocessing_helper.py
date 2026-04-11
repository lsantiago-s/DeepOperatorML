from __future__ import annotations
from pathlib import Path
import importlib.util
import sys
import types
from src.modules.models.config import DataConfig, TestConfig

def run_plotting(test_cfg: TestConfig, data_cfg: DataConfig) -> None:
    base_dir = Path(__file__).parent.parent.parent
    script_path = base_dir / 'problems' / \
        data_cfg.problem / 'problem_dependent_visualization.py'
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    if not script_path.exists():
        raise FileNotFoundError(f"Plotting script not found: {script_path}")

    # Avoid importing src/problems/__init__.py side effects when loading
    # problem-specific plotting scripts via file path.
    if "src.problems" not in sys.modules:
        problems_pkg = types.ModuleType("src.problems")
        problems_pkg.__path__ = [str(base_dir / "problems")]
        sys.modules["src.problems"] = problems_pkg

    module_name = f"problem_dependent_visualization"
    spec = importlib.util.spec_from_file_location(
        name=module_name, location=script_path)
    if spec is None:
        raise ModuleNotFoundError(f"Plotting module not found.")
    plot_module = importlib.util.module_from_spec(spec=spec)
    # maybe import sys and add plot_module to sys.modules dict here
    if spec.loader is None:
        raise AttributeError(f"{spec} has no 'loader' attribute.")
    spec.loader.exec_module(plot_module)
    plot_module.plot_metrics(test_cfg=test_cfg, data_cfg=data_cfg)
