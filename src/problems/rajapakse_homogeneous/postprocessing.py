from __future__ import annotations
import yaml
import logging
import numpy as np
from pathlib import Path
from src.modules.models.config import DataConfig, TestConfig

logger = logging.getLogger(__file__)


def get_output_data(test_cfg: TestConfig) -> dict[str, np.ndarray]:
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
    return {key: value for key, value in np.load(output_data_path).items()}


def get_input_functions(data_cfg: DataConfig) -> dict[str, np.ndarray]:
    raw_data = np.load(data_cfg.raw_data_path, allow_pickle=True)
    input_functions = {}
    for key in [
        "c11_over_c44",
        "c12_over_c44",
        "c13_over_c44",
        "c33_over_c44",
        "rho_over_rho0",
        "delta",
    ]:
        if key in raw_data:
            input_functions[key] = raw_data[key]
    return input_functions


def get_coordinates(data_cfg: DataConfig) -> dict[str, np.ndarray]:
    raw_data = np.load(data_cfg.raw_data_path, allow_pickle=True)
    return {
        "r": raw_data["r"],
        "z": raw_data["z"],
    }


def _raw_grid_shape(data_cfg: DataConfig) -> tuple[int, int]:
    with open(data_cfg.raw_metadata_path, "r", encoding="utf-8") as file:
        raw_metadata = yaml.safe_load(file)
    shape = raw_metadata["displacement_statistics"][data_cfg.targets[0]]["shape"]
    return int(shape[1]), int(shape[2])


def _select_test_subset(displacements: np.ndarray, data_cfg: DataConfig) -> np.ndarray:
    n_test = len(data_cfg.split_indices["xb_test"])
    if len(displacements) > n_test:
        displacements = displacements[data_cfg.split_indices["xb_test"]]
    return displacements


def format_target(displacements: np.ndarray, data_cfg: DataConfig) -> np.ndarray:
    displacements = _select_test_subset(displacements=displacements, data_cfg=data_cfg)
    n_r, n_z = _raw_grid_shape(data_cfg)

    displacements = displacements.reshape(displacements.shape[0], n_r, n_z, -1)
    if displacements.shape[-1] != 2:
        raise ValueError(
            f"Expected 2 output channels (real/imag), got shape {displacements.shape}."
        )

    # channels-first for plotting helpers
    displacements = displacements.transpose(0, 3, 1, 2)

    # Rebuild full plane by mirroring across r=0.
    displacements_flipped = np.flip(displacements, axis=2)
    displacements_full = np.concatenate((displacements_flipped, displacements), axis=2)
    return displacements_full


def reshape_coefficients(branch_out: np.ndarray, data_cfg: DataConfig, test_cfg: TestConfig) -> np.ndarray:
    return branch_out.reshape(
        data_cfg.data[data_cfg.targets[0]][data_cfg.split_indices["xb_test"]].shape[0],
        -1,
        test_cfg.model.rescaling.embedding_dimension,  # type: ignore
    )


def reshape_basis(trunk_out: np.ndarray, data_cfg: DataConfig, test_cfg: TestConfig) -> np.ndarray:
    n_r, n_z = _raw_grid_shape(data_cfg)
    basis = trunk_out.T.reshape(
        test_cfg.model.rescaling.embedding_dimension,  # type: ignore
        -1,
        n_r,
        n_z,
    )
    basis_flipped = np.flip(basis, axis=2)
    return np.concatenate((basis_flipped, basis), axis=2)


def format_bias(bias: np.ndarray, data_cfg: DataConfig, test_cfg: TestConfig) -> np.ndarray:
    if test_cfg.model.strategy.name != "pod":  # type: ignore
        return bias

    n_r, n_z = _raw_grid_shape(data_cfg)
    bias = bias.T.reshape(-1, n_r, n_z)
    bias_flipped = np.flip(bias, axis=1)
    return np.concatenate((bias_flipped, bias), axis=1)


def get_flat_test_target(output_data: dict[str, np.ndarray], data_cfg: DataConfig) -> np.ndarray:
    target = output_data[data_cfg.targets[0]]
    return _select_test_subset(displacements=target, data_cfg=data_cfg)
