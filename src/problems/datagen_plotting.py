from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _disable_latex_text() -> None:
    matplotlib.rcParams["text.usetex"] = False


def _resolve_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _select_representative_indices(values: np.ndarray, max_count: int = 3) -> list[int]:
    if values.size == 0:
        return []
    if values.size <= max_count:
        return list(range(int(values.size)))
    order = np.argsort(values)
    anchors = np.linspace(0, len(order) - 1, num=max_count, dtype=int)
    indices = [int(order[pos]) for pos in anchors]
    return list(dict.fromkeys(indices))


def _save_figure(fig: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved dataset sanity plot at %s", output_path)


def _plot_kelvin_dataset(data: dict[str, np.ndarray], output_dir: Path) -> list[Path]:
    from src.problems.kelvin.plot_field import plot_field

    _disable_latex_text()

    g_u = np.asarray(data["g_u"])
    coords = {key: np.asarray(data[key]) for key in ("x", "y", "z")}

    if "q" in data:
        branch_values = np.column_stack(
            [
                np.sum(np.abs(np.asarray(data["q"])), axis=1),
                np.asarray(data["mu"]),
                np.asarray(data["nu"]),
            ]
        )
        input_labels = ["|q|_1", r"$\mu$", r"$\nu$"]
        ranking_values = branch_values[:, 0]
    else:
        branch_values = np.column_stack([np.asarray(data["mu"]), np.asarray(data["nu"])])
        input_labels = [r"$\mu$", r"$\nu$"]
        ranking_values = branch_values[:, 0]

    outputs: list[Path] = []
    for sample_idx in _select_representative_indices(ranking_values, max_count=3):
        # Kelvin's existing plotting helper slices fields as (channels, x, y, z),
        # while generated datasets store them as (x, y, z, channels).
        field = np.moveaxis(g_u[sample_idx], -1, 0)
        sample_params = branch_values[sample_idx]
        for plane in ("xy", "xz", "yz"):
            with np.errstate(divide="ignore", invalid="ignore"):
                fig = plot_field(
                    coords=coords,
                    truth_field=field,
                    pred_field=field,
                    input_function_labels=input_labels,
                    input_function_values=sample_params,
                    target_labels=["u_x", "u_y", "u_z"],
                    plot_plane=plane,
                    plotted_variable="truths",
                )
            out_path = output_dir / f"sanity_kelvin_sample{sample_idx:04d}_{plane}.png"
            _save_figure(fig, out_path)
            outputs.append(out_path)
    return outputs


def _plot_ground_vibration_dataset(data: dict[str, np.ndarray], output_dir: Path) -> list[Path]:
    from src.problems.ground_vibration.plot_influence_matrix import plot_influence_matrix

    _disable_latex_text()

    g_u = np.asarray(data["g_u"])
    x_coords = np.asarray(data["x"])
    if g_u.ndim != 3 or g_u.shape[-1] != 4:
        raise ValueError(f"Unexpected ground_vibration g_u shape: {g_u.shape}")

    n_nodes = int(x_coords.shape[0])
    param_keys = [key for key in ("c11", "c13", "c33", "c44", "ρ", "η", "a0") if key in data]
    if not param_keys:
        param_keys = ["sample_index"]

    if param_keys == ["sample_index"]:
        params = np.arange(g_u.shape[0], dtype=float)[:, None]
    else:
        params = np.column_stack([np.asarray(data[key]) for key in param_keys])
    param_labels = ["rho" if key == "ρ" else "eta" if key == "η" else key for key in param_keys]
    ranking_values = np.asarray(data["a0"], dtype=float) if "a0" in data else params[:, -1]

    outputs: list[Path] = []
    for sample_idx in _select_representative_indices(ranking_values, max_count=3):
        sample_matrix = g_u[sample_idx].reshape(n_nodes, n_nodes, 4)
        fig = plot_influence_matrix(
            U_true=sample_matrix,
            U_pred=sample_matrix,
            param_map={
                label: float(value)
                for label, value in zip(param_labels, params[sample_idx], strict=True)
            },
            target_labels=["u_xx", "u_xz", "u_zx", "u_zz"],
        )
        out_path = output_dir / f"sanity_ground_vibration_sample{sample_idx:04d}_matrix.png"
        _save_figure(fig, out_path)
        outputs.append(out_path)
    return outputs


def _plot_rajapakse_dataset(data: dict[str, np.ndarray], output_dir: Path, prefix: str) -> list[Path]:
    from src.problems.rajapakse_fixed_material.plot_field import plot_2D_field

    _disable_latex_text()

    g_u = np.asarray(data["g_u"])
    coords = {"r": np.asarray(data["r"]), "z": np.asarray(data["z"])}

    candidate_params = [
        key
        for key in (
            "delta",
            "E",
            "nu",
            "rho",
            "omega",
            "c11_over_c44",
            "c12_over_c44",
            "c13_over_c44",
            "c33_over_c44",
        )
        if key in data
    ]
    if not candidate_params:
        candidate_params = ["sample_index"]

    if candidate_params == ["sample_index"]:
        branch_values = np.arange(g_u.shape[0], dtype=float)[:, None]
    else:
        branch_values = np.column_stack([np.asarray(data[key], dtype=float) for key in candidate_params])
    ranking_values = branch_values[:, 0]

    outputs: list[Path] = []
    for sample_idx in _select_representative_indices(ranking_values, max_count=3):
        field = np.asarray(g_u[sample_idx], dtype=np.complex128)
        fig = plot_2D_field(
            coords=coords,
            truth_field=np.stack([field.real, field.imag], axis=0),
            pred_field=np.stack([field.real, field.imag], axis=0),
            input_function_labels=candidate_params,
            input_function_value=branch_values[sample_idx].tolist(),
            target_labels=["Re(u)", "Im(u)"],
        )
        out_path = output_dir / f"sanity_{prefix}_sample{sample_idx:04d}.png"
        _save_figure(fig, out_path)
        outputs.append(out_path)
    return outputs


def _plot_multilayer_dataset(data: dict[str, np.ndarray], output_dir: Path) -> list[Path]:
    from src.problems.multilayer_horizontal_rocking.plot_helper import _split_blocks

    _disable_latex_text()

    g_u = np.asarray(data["g_u"])
    a0 = np.asarray(data["a0"], dtype=float) if "a0" in data else np.arange(g_u.shape[0], dtype=float)
    case_labels = np.asarray(data["paper_case_label"]).astype(str) if "paper_case_label" in data else None
    blocks = _split_blocks(g_u)

    outputs: list[Path] = []
    for sample_idx in _select_representative_indices(a0, max_count=3):
        fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
        for ax, block_name in zip(axes.flat, ("Uxx", "Uxz", "Uzx", "Uzz"), strict=True):
            im = ax.imshow(np.abs(blocks[block_name][sample_idx]), origin="lower", cmap="magma")
            ax.set_title(block_name)
            ax.set_xlabel("source index")
            ax.set_ylabel("receiver index")
            fig.colorbar(im, ax=ax, fraction=0.046)
        title = f"Multilayer sample {sample_idx} | a0={a0[sample_idx]:.3f}"
        if case_labels is not None:
            title += f" | case={case_labels[sample_idx]}"
        fig.suptitle(title)
        out_path = output_dir / f"sanity_multilayer_sample{sample_idx:04d}.png"
        _save_figure(fig, out_path)
        outputs.append(out_path)
    return outputs


def generate_problem_dataset_plots(problem_name: str, data_path: str | Path) -> list[Path]:
    try:
        data_file = _resolve_path(data_path)
        if not data_file.exists():
            logger.warning("Skipping dataset sanity plots for %s: data file not found at %s", problem_name, data_file)
            return []

        output_dir = data_file.parent
        with np.load(data_file, allow_pickle=True) as raw_data:
            data = {key: raw_data[key] for key in raw_data.files}

        logger.info("Generating dataset sanity plots for %s from %s", problem_name, data_file)

        if problem_name == "kelvin":
            return _plot_kelvin_dataset(data=data, output_dir=output_dir)
        if problem_name == "ground_vibration":
            return _plot_ground_vibration_dataset(data=data, output_dir=output_dir)
        if problem_name in {"rajapakse_fixed_material", "rajapakse_homogeneous"}:
            return _plot_rajapakse_dataset(data=data, output_dir=output_dir, prefix=problem_name)
        if problem_name == "multilayer_horizontal_rocking":
            return _plot_multilayer_dataset(data=data, output_dir=output_dir)

        logger.info("No dataset sanity plotting hook registered for %s. Skipping.", problem_name)
        return []
    except Exception:
        logger.exception("Failed to generate dataset sanity plots for %s from %s", problem_name, data_path)
        return []
