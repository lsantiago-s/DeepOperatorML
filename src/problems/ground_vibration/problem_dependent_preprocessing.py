from __future__ import annotations
import logging
import numpy as np
import numpy.typing as npt
from typing import Any, Iterable

logger = logging.getLogger(__name__)

def input_function_encoding(input_funcs: Iterable[Iterable | npt.NDArray], encoding=None) -> npt.NDArray:
    return np.column_stack(tup=input_funcs)  # type: ignore


def format_to_don(*coords: Iterable[npt.ArrayLike]) -> npt.NDArray:
    if len(coords) == 1 and isinstance(coords[0], Iterable):
        coords = coords[0]  # type: ignore

    meshes = np.meshgrid(*coords, indexing='ij')  # type: ignore

    axes = [m.flatten() for m in meshes]
    data = np.column_stack(axes)
    return data


def preprocess_raw_data(raw_npz_filename: str,
                        input_function_keys: list[str],
                        coordinate_keys: list[str],
                        processed_dataset_keys: dict[str, list[str]]) -> dict[str, npt.NDArray]:
    """
    Processed data from an npz file and groups the input functions and coordinates into arrays
    called, named according to given labels, suitable for creating the PyTorch dataset for the Kelvin problem.

    The function assumes that:
      - The input functions (sensors) are stored under keys given by input_function_keys (num_sensors, num_input_functions).
      - The coordinate arrays (for the trunk) are stored under keys given by coordinate_keys.
        A meshgrid is created and then flattened to yield a 2D array of shape
        (num_coordinate_points, num_coordinate_dimensions).
      - Operator output under the key 'g_u'.

    Args:
        npz_filename (str): Path to the .npz file.
        input_function_keys (list of str): List of keys for sensor (input function) arrays.
        coordinate_keys (list of str): List of keys for coordinate arrays.
        processed_dataset_keys (dict): Map of {'FEATURES': usually ['xb', 'xt], 'TARGETS': [g_u_1, ...]}.

    Returns:
        dict: A dictionary with the example keys:
            - 'xb': A 2D numpy array of shape (num_sensor_points, num_sensor_dimensions).
            - 'xt': A 2D numpy array of shape (num_coordinate_points, num_coordinate_dimensions).
            - 'g_u': The operator output array.
    """

    data = np.load(
        file=raw_npz_filename,
        allow_pickle=True
    )  # 'label': NDArray
    
    keys = [i for i in input_function_keys]
    branch_input = []
    missing_branch_keys = []
    for k in keys:
        if k in data:
            branch_input.append(np.asarray(data[k]))
        else:
            missing_branch_keys.append(k)
    if missing_branch_keys:
        raise KeyError(f"Missing required input_function_keys in raw dataset: {missing_branch_keys}")
    branch_input = np.column_stack(branch_input)

    coord_arrays: list[np.ndarray] = []
    for key in coordinate_keys:
        if key not in data:
            raise KeyError(f"Missing coordinate key '{key}' in raw dataset.")
        coord_arrays.append(np.asarray(data[key]))

    if len(coord_arrays) == 1:
        coord_raw = coord_arrays[0]
        if coord_raw.ndim == 1:
            coord_vec = coord_raw
        elif coord_raw.ndim == 2:
            # Legacy datasets stored x as a mesh-like matrix.
            coord_vec = np.asarray(coord_raw[:, 0])
        else:
            raise ValueError(
                f"Unsupported coordinate array shape for '{coordinate_keys[0]}': {coord_raw.shape}"
            )
        trunk_input = format_to_don((coord_vec, coord_vec))
    elif len(coord_arrays) == 3:
        # Operator query for homogeneous ground vibration:
        # (x_m, s1_n, s2_n) for all collocation/source element pairs (m, n).
        x_vec = np.asarray(coord_arrays[0], dtype=float).reshape(-1)
        s1_vec = np.asarray(coord_arrays[1], dtype=float).reshape(-1)
        s2_vec = np.asarray(coord_arrays[2], dtype=float).reshape(-1)
        if s1_vec.shape[0] != s2_vec.shape[0]:
            raise ValueError(
                "Source geometry vectors must have same length. "
                f"Got s1={s1_vec.shape[0]}, s2={s2_vec.shape[0]}."
            )
        n_field = x_vec.shape[0]
        n_source = s1_vec.shape[0]
        trunk_input = np.column_stack(
            [
                np.repeat(x_vec, n_source),
                np.tile(s1_vec, n_field),
                np.tile(s2_vec, n_field),
            ]
        )
    else:
        # Generic fallback: direct cartesian product meshgrid.
        flattened = []
        for arr in coord_arrays:
            arr_flat = np.asarray(arr).reshape(-1)
            flattened.append(arr_flat)
        trunk_input = format_to_don(tuple(flattened))

    features = {
        processed_dataset_keys['features'][0]: branch_input,
        processed_dataset_keys['features'][1]: trunk_input,
    }

    processed_data = features

    for i in processed_dataset_keys['targets']:
        if i in data:   
            processed_data[i] = np.concatenate([(data[i].real), data[i].imag], axis=-1)
        else:
            raise KeyError(
                f"Operator target '{processed_dataset_keys['targets'][0]}' must be present in the dataset keys")
    return processed_data


def run_preprocessing(problem_settings: dict[str, Any]) -> dict[str, npt.ArrayLike]:
    processed_data = preprocess_raw_data(
        raw_npz_filename=problem_settings['raw_data_path'],
        input_function_keys=problem_settings['input_function_keys'],
        coordinate_keys=problem_settings['coordinate_keys'],
        processed_dataset_keys=problem_settings['data_labels'],
    )
    return processed_data  # type: ignore
