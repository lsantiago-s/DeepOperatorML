from __future__ import annotations
import yaml
import logging
import numpy as np
from pathlib import Path
from src.modules.models.config import DataConfig, TestConfig
logger = logging.getLogger(__file__)


def get_output_data(test_cfg: TestConfig) -> dict[str, np.ndarray]:
    if test_cfg.problem is None:
        raise ValueError(f"Problem name must be set in TestConfig.")
    base_dir = Path(__file__).parent.parent.parent.parent
    output_data_path = base_dir / test_cfg.output_path / test_cfg.problem / \
        test_cfg.experiment_version / 'aux' / 'output_data.npz'
    
    output_data = {i: j for i, j in np.load(output_data_path).items()}
    return output_data

def get_input_functions(data_cfg: DataConfig) -> dict[str, np.ndarray]:
    raw_data = np.load(data_cfg.raw_data_path)
    c11 = raw_data['c11']
    c13 = raw_data['c13']
    c33 = raw_data['c33']
    c44 = raw_data['c44']
    rho = raw_data['ρ']
    eta = raw_data['η'] if 'η' in raw_data else np.zeros_like(rho)
    a0 = raw_data['a0'] if 'a0' in raw_data else raw_data['ω']
    input_functions = {
        data_cfg.input_functions[0]: c11,
        data_cfg.input_functions[1]: c13,
        data_cfg.input_functions[2]: c33,
        data_cfg.input_functions[3]: c44,
        data_cfg.input_functions[4]: rho,
        data_cfg.input_functions[5]: eta,
        data_cfg.input_functions[6]: a0,
    }
    return input_functions


def get_coordinates(data_cfg: DataConfig) -> dict[str, np.ndarray]:
    raw_data = np.load(data_cfg.raw_data_path)
    x = raw_data['x']
    coordinates = {'x': x}
    return coordinates

def format_target(displacements: np.ndarray, data_cfg: DataConfig) -> np.ndarray:
    if len(displacements) > len(data_cfg.split_indices['xb_test']):
        test_indices = data_cfg.split_indices['xb_test']
        displacements = displacements[test_indices]
    import matplotlib.pyplot as plt
    displacements = displacements.reshape(
        displacements.shape[0],
        int(np.sqrt(displacements.shape[1])),
        int(np.sqrt(displacements.shape[1])),
        displacements.shape[2],
    )
    displacements_real = displacements[..., : displacements.shape[3] // 2]
    displacements_imag = displacements[..., displacements.shape[3] // 2 :]
    displacements = displacements_real + 1j * displacements_imag
    return displacements

def save_U_matrix(influence_matrix_true: np.ndarray, influence_matrix_pred: np.ndarray, save_path: Path):
    U_matrix_pred = get_U(influence_matrix_pred)
    U_matrix_true = get_U(influence_matrix_true)
    np.savez(save_path / 'U_matrices.npz', U_true=U_matrix_true, U_pred=U_matrix_pred)

def get_U(influence_matrix: np.ndarray) -> np.ndarray:
    samples, N_s, _, _ = influence_matrix.shape
    reshaped = influence_matrix.reshape(samples, N_s, N_s, 2, 2)
    transposed = reshaped.transpose(0, 1, 3, 2, 4)

    U = transposed.reshape(samples, N_s * 2, N_s * 2)
    return U

def reshape_coefficients(branch_out: np.ndarray, data_cfg: DataConfig, test_cfg: TestConfig) -> np.ndarray:
    return branch_out.reshape(
        data_cfg.data[data_cfg.targets[0]][data_cfg.split_indices['xb_test']].shape[0],
        -1,
        test_cfg.model.rescaling.embedding_dimension,  # type: ignore
    )

def reshape_basis(trunk_out: np.ndarray, data_cfg: DataConfig, test_cfg: TestConfig) -> np.ndarray:
    with open(data_cfg.raw_metadata_path, 'r') as file:
        raw_metadata = yaml.safe_load(file)
    basis = trunk_out.T.reshape(
        test_cfg.model.rescaling.embedding_dimension,  # type: ignore
        int(np.sqrt(raw_metadata["influence_matrix"][data_cfg.targets[0]]["shape"][1])),
        int(np.sqrt(raw_metadata["influence_matrix"][data_cfg.targets[0]]["shape"][1])),
        raw_metadata["influence_matrix"][data_cfg.targets[0]]["shape"][2] * 2,
    )

    basis_real = basis[..., : basis.shape[3] // 2]
    basis_imag = basis[..., basis.shape[3] // 2 :]
    basis = basis_real + 1j * basis_imag
    return basis

def format_bias(bias: np.ndarray, data_cfg: DataConfig, test_cfg: TestConfig) -> np.ndarray:
    if test_cfg.model.strategy.name != 'pod':  # type: ignore
        return bias
    else:
        with open(data_cfg.raw_metadata_path, 'r') as file:
            raw_metadata = yaml.safe_load(file)
        bias = bias.T.reshape(
            -1,
            raw_metadata["influence_matrix"][data_cfg.targets[0]]["shape"][2] * 2,
            int(np.sqrt(raw_metadata["influence_matrix"][data_cfg.targets[0]]["shape"][1])),
            int(np.sqrt(raw_metadata["influence_matrix"][data_cfg.targets[0]]["shape"][1])),
        )
        bias_flipped = np.flip(bias, axis=1)
        bias_full = np.concatenate((bias_flipped, bias), axis=1)
        return bias_full
