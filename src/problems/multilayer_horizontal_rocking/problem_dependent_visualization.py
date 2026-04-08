from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.modules.models.config import DataConfig, TestConfig
from src.problems.multilayer_horizontal_rocking import plot_helper
from src.problems.multilayer_horizontal_rocking import postprocessing as ppr

logger = logging.getLogger(__name__)


def plot_metrics(test_cfg: TestConfig, data_cfg: DataConfig) -> None:
    if test_cfg.problem is None:
        raise ValueError("TestConfig.problem must be set before plotting.")

    if test_cfg.config is None:
        raise ValueError("Missing test config dictionary in TestConfig.config.")

    raw_data = ppr.load_raw_data(data_cfg)
    if "properties" not in raw_data or "omega" not in raw_data:
        raise KeyError("Raw dataset must contain 'properties' and 'omega' for paper comparison plots.")

    output_data = ppr.load_output_data(test_cfg)
    truth_test, pred_test, test_indices = ppr.get_truth_pred_complex(output_data=output_data, data_cfg=data_cfg)

    truth_test_matrix = ppr.reshape_influence(truth_test)
    pred_test_matrix = ppr.reshape_influence(pred_test)

    plots_path = (
        Path(test_cfg.output_path)
        / test_cfg.problem
        / str(test_cfg.experiment_version)
        / "plots"
    )
    plots_path.mkdir(parents=True, exist_ok=True)

    b_value = 0.0
    channel_names: list[str] | None = None
    try:
        with open(data_cfg.raw_metadata_path, "r", encoding="utf-8") as f:
            import yaml

            raw_meta = yaml.safe_load(f)
        b_value = float(raw_meta.get("B", 0.0))
        if isinstance(raw_meta.get("g_u_channels"), list):
            channel_names = [str(c) for c in raw_meta["g_u_channels"]]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse B from raw metadata (%s). Using B=0.0", exc)

    logger.info("Generating multilayer paper-style plots at %s", plots_path)

    plot_helper.run_all_multilayer_plots(
        plots_path=plots_path,
        properties_all=np.asarray(raw_data["properties"], dtype=float),
        omega_all=np.asarray(raw_data["omega"], dtype=float),
        truth_test_matrix=truth_test_matrix,
        pred_test_matrix=pred_test_matrix,
        test_indices=np.asarray(test_indices, dtype=int),
        b_value=b_value,
        config=test_cfg.config,
        channel_names=channel_names,
    )

    logger.info("Finished multilayer paper-style plotting.")
