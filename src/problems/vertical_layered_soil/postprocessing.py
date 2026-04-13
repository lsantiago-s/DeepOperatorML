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


def reshape_channels(influence: np.ndarray) -> np.ndarray:
    """Return influence channels as (samples, M, M, 4), complex."""
    influence_complex = to_complex_channels(influence)

    if influence_complex.ndim == 2:
        # Legacy layout: flattened full matrix scalar channel.
        n = infer_grid_size(influence_complex.shape[1])
        if n % 2 != 0:
            raise ValueError(f"Legacy full matrix size must be even. Got n={n}.")
        full = influence_complex.reshape(influence_complex.shape[0], n, n)
        m = n // 2
        return np.stack(
            [
                full[:, :m, :m],
                full[:, :m, m:],
                full[:, m:, :m],
                full[:, m:, m:],
            ],
            axis=-1,
        )

    if influence_complex.ndim != 3:
        raise ValueError(
            f"Expected influence with ndim=2 or 3 after channel conversion, got shape {influence_complex.shape}."
        )

    n_points = influence_complex.shape[1]
    n_channels = influence_complex.shape[2]

    if n_channels == 4:
        m = infer_grid_size(n_points)
        return influence_complex.reshape(influence_complex.shape[0], m, m, 4)

    if n_channels == 1:
        n = infer_grid_size(n_points)
        if n % 2 != 0:
            raise ValueError(f"Single-channel full matrix size must be even. Got n={n}.")
        full = influence_complex[..., 0].reshape(influence_complex.shape[0], n, n)
        m = n // 2
        return np.stack(
            [
                full[:, :m, :m],
                full[:, :m, m:],
                full[:, m:, :m],
                full[:, m:, m:],
            ],
            axis=-1,
        )

    raise ValueError(
        f"Unsupported number of complex channels ({n_channels}). Expected 1 or 4 for vertical_layered_soil."
    )


def channels_to_full_matrix(channels: np.ndarray) -> np.ndarray:
    """Convert (samples, M, M, 4) channels [Uxx,Uxz,Uzx,Uzz] to full U matrix (samples, 2M, 2M)."""
    arr = np.asarray(channels, dtype=np.complex128)
    if arr.ndim != 4 or arr.shape[-1] != 4:
        raise ValueError(f"Expected channels with shape (samples, M, M, 4), got {arr.shape}.")

    samples, m, _, _ = arr.shape
    full = np.zeros((samples, 2 * m, 2 * m), dtype=np.complex128)
    full[:, :m, :m] = arr[..., 0]
    full[:, :m, m:] = arr[..., 1]
    full[:, m:, :m] = arr[..., 2]
    full[:, m:, m:] = arr[..., 3]
    return full


def reshape_influence(influence: np.ndarray) -> np.ndarray:
    """Backward-compatible helper returning full matrices as (samples, 2M, 2M, 1), complex."""
    channels = reshape_channels(influence)
    full = channels_to_full_matrix(channels)
    return full[..., None]


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
