from __future__ import annotations
import logging
import numpy as np
import numpy.typing as npt
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def input_function_encoding(input_funcs: Iterable[Iterable | npt.NDArray]) -> npt.NDArray:
    return np.column_stack(tuple(input_funcs))  # type: ignore[arg-type]


def format_to_don(*coords: Iterable[npt.ArrayLike]) -> npt.NDArray:
    if len(coords) == 1 and isinstance(coords[0], Iterable):
        coords = coords[0]  # type: ignore[assignment]

    meshes = np.meshgrid(*coords, indexing="ij")  # type: ignore[arg-type]
    axes = [m.flatten() for m in meshes]
    return np.column_stack(axes)


def preprocess_raw_data(
    raw_npz_filename: str,
    input_function_keys: list[str],
    coordinate_keys: list[str],
    processed_dataset_keys: dict[str, list[str]],
) -> dict[str, npt.NDArray]:
    data = np.load(file=raw_npz_filename, allow_pickle=True)

    input_funcs = [data[key] for key in input_function_keys]
    coords = [data[name] for name in coordinate_keys]

    branch_input = input_function_encoding(input_funcs=input_funcs)
    trunk_input = format_to_don(coords)

    processed_data = {
        processed_dataset_keys["features"][0]: branch_input,
        processed_dataset_keys["features"][1]: trunk_input,
    }

    for target_key in processed_dataset_keys["targets"]:
        if target_key not in data:
            raise KeyError(
                f"Operator target '{target_key}' must be present in dataset keys"
            )

        values = data[target_key]
        if np.iscomplexobj(values):
            values = np.stack([values.real, values.imag], axis=3)
        else:
            values = np.asarray(values)

        processed_data[target_key] = values.reshape(
            len(branch_input),
            len(trunk_input),
            -1,
        )

    return processed_data


def run_preprocessing(problem_settings: dict[str, Any]) -> dict[str, npt.NDArray]:
    return preprocess_raw_data(
        raw_npz_filename=problem_settings["raw_data_path"],
        input_function_keys=problem_settings["input_function_keys"],
        coordinate_keys=problem_settings["coordinate_keys"],
        processed_dataset_keys=problem_settings["data_labels"],
    )
