from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.modules.models.config import DataConfig, TestConfig

logger = logging.getLogger(__name__)


def load_output_data(test_cfg: TestConfig) -> dict[str, np.ndarray]:
    if test_cfg.problem is None:
        raise ValueError("Problem name must be set in TestConfig.")

    base_dir = Path(__file__).parent.parent.parent.parent
    output_data_path = (
        base_dir
        / test_cfg.output_path
        / test_cfg.problem
        / test_cfg.experiment_version
        / "aux"
        / "output_data.npz"
    )
    if not output_data_path.exists():
        raise FileNotFoundError(f"Missing output data file: {output_data_path}")

    return {k: v for k, v in np.load(output_data_path).items()}


def load_raw_data(data_cfg: DataConfig) -> dict[str, np.ndarray]:
    raw = np.load(data_cfg.raw_data_path)
    return {k: raw[k] for k in raw.files}


def to_complex_channels(arr: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(arr):
        return np.asarray(arr, dtype=np.complex128)

    arr = np.asarray(arr)
    if arr.shape[-1] % 2 != 0:
        raise ValueError(
            f"Expected even number of channels for real/imag stacking, got {arr.shape[-1]}"
        )

    half = arr.shape[-1] // 2
    return arr[..., :half] + 1j * arr[..., half:]


def infer_grid_size(num_points: int) -> int:
    m = int(round(np.sqrt(num_points)))
    if m * m != num_points:
        raise ValueError(f"Expected square number of coordinates, got {num_points}")
    return m


def reshape_influence(influence: np.ndarray) -> np.ndarray:
    """Return influence as (samples, M, M, channels), complex."""
    influence_complex = to_complex_channels(influence)
    m = infer_grid_size(influence_complex.shape[1])
    return influence_complex.reshape(influence_complex.shape[0], m, m, influence_complex.shape[2])


def get_truth_pred_complex(
    output_data: dict[str, np.ndarray],
    data_cfg: DataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (truth_test, pred_test, test_indices) with complex channels."""
    target_key = data_cfg.targets[0]
    if target_key not in output_data:
        raise KeyError(f"Output data missing key '{target_key}'")
    if "predictions" not in output_data:
        raise KeyError("Output data missing key 'predictions'")

    test_indices_key = f"{data_cfg.features[0]}_test"
    if test_indices_key not in data_cfg.split_indices:
        raise KeyError(f"Split indices missing key '{test_indices_key}'")

    test_indices = np.asarray(data_cfg.split_indices[test_indices_key], dtype=int)

    truth_all = to_complex_channels(output_data[target_key])
    pred_test = to_complex_channels(output_data["predictions"])

    if truth_all.shape[0] == data_cfg.data[target_key].shape[0]:
        truth_test = truth_all[test_indices]
    elif truth_all.shape[0] == len(test_indices):
        truth_test = truth_all
    else:
        raise ValueError(
            "Unable to align truth data with test split: "
            f"truth rows={truth_all.shape[0]}, test rows={len(test_indices)}, "
            f"dataset rows={data_cfg.data[target_key].shape[0]}"
        )

    if pred_test.shape[0] != len(test_indices):
        min_rows = min(pred_test.shape[0], len(test_indices), truth_test.shape[0])
        logger.warning(
            "Prediction rows (%d) differ from test rows (%d). Truncating to %d rows.",
            pred_test.shape[0],
            len(test_indices),
            min_rows,
        )
        pred_test = pred_test[:min_rows]
        truth_test = truth_test[:min_rows]
        test_indices = test_indices[:min_rows]

    return truth_test, pred_test, test_indices
