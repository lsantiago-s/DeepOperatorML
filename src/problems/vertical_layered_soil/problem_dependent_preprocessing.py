from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

logger = logging.getLogger(__name__)


def _load_raw_metadata(raw_npz_filename: str) -> dict[str, Any]:
    meta_path = Path(raw_npz_filename).with_suffix(".yaml")
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return payload if isinstance(payload, dict) else {}


def _update_problem_settings_from_raw_metadata(problem_settings: dict[str, Any], raw_meta: dict[str, Any]) -> None:
    input_layout = raw_meta.get("input_layout")
    if isinstance(input_layout, list) and len(input_layout) > 0:
        problem_settings["input_function_labels"] = [str(x) for x in input_layout]

    coordinate_layout = raw_meta.get("coordinate_layout")
    if isinstance(coordinate_layout, list) and len(coordinate_layout) > 0:
        problem_settings["coordinate_keys"] = [str(x) for x in coordinate_layout]

    channels = raw_meta.get("g_u_channels")
    if isinstance(channels, list) and len(channels) > 0:
        real_keys = [f"re_{str(ch).lower()}" for ch in channels]
        imag_keys = [f"im_{str(ch).lower()}" for ch in channels]
        problem_settings["output_keys"] = real_keys + imag_keys
        problem_settings["output_labels"] = (
            [f"$\\Re({str(ch)})$" for ch in channels]
            + [f"$\\Im({str(ch)})$" for ch in channels]
        )


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
    if g_u.ndim == 2:
        g_u = g_u[..., None]
    if g_u.ndim != 3:
        raise ValueError(f"Expected g_u with ndim=2 or 3, got shape {g_u.shape}.")
    if np.iscomplexobj(g_u):
        g_u = np.concatenate([g_u.real, g_u.imag], axis=-1)

    num_samples = branch_input.shape[0]
    num_coords = trunk_input.shape[0]
    if g_u.shape[0] != num_samples:
        raise ValueError(
            f"g_u sample dimension ({g_u.shape[0]}) must match xb samples ({num_samples})."
        )
    if g_u.shape[1] != num_coords:
        raise ValueError(
            f"g_u coordinate dimension ({g_u.shape[1]}) must match xt rows ({num_coords})."
        )

    return {
        processed_dataset_keys["features"][0]: branch_input,
        processed_dataset_keys["features"][1]: trunk_input,
        processed_dataset_keys["targets"][0]: g_u,
    }


def run_preprocessing(problem_settings: dict[str, Any]) -> dict[str, npt.NDArray]:
    raw_meta = _load_raw_metadata(problem_settings["raw_data_path"])
    _update_problem_settings_from_raw_metadata(problem_settings=problem_settings, raw_meta=raw_meta)
    return preprocess_raw_data(
        raw_npz_filename=problem_settings["raw_data_path"],
        processed_dataset_keys=problem_settings["data_labels"],
    )
