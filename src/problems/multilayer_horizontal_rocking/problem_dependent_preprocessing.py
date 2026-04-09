from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def preprocess_raw_data(
    raw_npz_filename: str,
    processed_dataset_keys: dict[str, list[str]],
) -> dict[str, npt.NDArray]:
    data = np.load(file=raw_npz_filename, allow_pickle=True)

    if "xb" in data:
        branch_input = np.asarray(data["xb"], dtype=float)
    elif "params" in data:
        branch_input = np.asarray(data["params"], dtype=float)
    else:
        raise KeyError("Raw multilayer dataset must contain 'xb' (or legacy 'params').")

    if "xt" in data:
        trunk_input = np.asarray(data["xt"], dtype=float)
    elif "r" in data and "s" in data:
        rr, ss = np.meshgrid(data["r"], data["s"], indexing="ij")
        trunk_input = np.column_stack([rr.ravel(), ss.ravel()])
    else:
        raise KeyError("Raw multilayer dataset must contain 'xt' or both 'r' and 's'.")

    if "g_u" not in data:
        raise KeyError("Raw multilayer dataset must contain 'g_u'.")
    g_u = np.asarray(data["g_u"])
    if np.iscomplexobj(g_u):
        g_u = np.concatenate([g_u.real, g_u.imag], axis=-1)

    num_samples = branch_input.shape[0]
    num_coords = trunk_input.shape[0]
    g_u = g_u.reshape(num_samples, num_coords, -1)

    return {
        processed_dataset_keys["features"][0]: branch_input,
        processed_dataset_keys["features"][1]: trunk_input,
        processed_dataset_keys["targets"][0]: g_u,
    }


def run_preprocessing(problem_settings: dict[str, Any]) -> dict[str, npt.NDArray]:
    return preprocess_raw_data(
        raw_npz_filename=problem_settings["raw_data_path"],
        processed_dataset_keys=problem_settings["data_labels"],
    )
