from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml

logger = logging.getLogger(__name__)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_scalar_complex_field(matrix: np.ndarray) -> np.ndarray:
    """Convert supported matrix layouts to complex field (samples, n, n)."""
    arr = np.asarray(matrix)
    if arr.ndim == 2:
        npts = int(arr.shape[1])
        n = int(round(np.sqrt(npts)))
        if n * n != npts:
            raise ValueError(f"Expected flattened square matrix for ndim=2, got shape {arr.shape}")
        return np.asarray(arr, dtype=np.complex128).reshape(arr.shape[0], n, n)
    if arr.ndim == 4:
        return np.asarray(arr[..., 0], dtype=np.complex128)
    if arr.ndim == 3:
        return np.asarray(arr, dtype=np.complex128)
    raise ValueError(f"Expected matrix with ndim 3 or 4, got shape {arr.shape}")


def _split_blocks(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Split full U (2M x 2M) into Uxx, Uxz, Uzx, Uzz blocks."""
    u = _to_scalar_complex_field(matrix)
    n = u.shape[1]
    if n != u.shape[2] or n % 2 != 0:
        raise ValueError(f"Expected square even matrix per sample, got {u.shape}")
    m = n // 2
    return {
        "Uxx": u[:, :m, :m],
        "Uxz": u[:, :m, m:],
        "Uzx": u[:, m:, :m],
        "Uzz": u[:, m:, m:],
    }


def _relative_error_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true.reshape(y_true.shape[0], -1)
    yp = y_pred.reshape(y_pred.shape[0], -1)
    num = np.linalg.norm(yt - yp, axis=1)
    den = np.linalg.norm(yt, axis=1) + 1e-14
    return num / den


def _ensure_dirs(base_path: Path) -> dict[str, Path]:
    paths = {
        "alignment": base_path / "paper_alignment",
        "profiles": base_path / "paper_profiles",
        "heatmaps": base_path / "prediction_heatmaps",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _plot_block_mean_heatmaps(truth_blocks: dict[str, np.ndarray], pred_blocks: dict[str, np.ndarray], save_path: Path) -> None:
    block_names = ["Uxx", "Uxz", "Uzx", "Uzz"]
    fig, axes = plt.subplots(len(block_names), 3, figsize=(12, 3.2 * len(block_names)), constrained_layout=True)

    for row, block in enumerate(block_names):
        truth_abs = np.mean(np.abs(truth_blocks[block]), axis=0)
        pred_abs = np.mean(np.abs(pred_blocks[block]), axis=0)
        rel_err = np.abs(pred_abs - truth_abs) / np.maximum(np.max(truth_abs), 1e-12)

        im0 = axes[row, 0].imshow(truth_abs, origin="lower", cmap="magma")
        axes[row, 0].set_title(f"{block} mean |truth|")
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        im1 = axes[row, 1].imshow(pred_abs, origin="lower", cmap="magma")
        axes[row, 1].set_title(f"{block} mean |pred|")
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        im2 = axes[row, 2].imshow(rel_err, origin="lower", cmap="viridis")
        axes[row, 2].set_title(f"{block} mean rel. err")
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.046)

        for col in range(3):
            axes[row, col].set_xlabel("source element index")
            axes[row, col].set_ylabel("receiver element index")

    fig.suptitle("Full-matrix block alignment: truth vs prediction")
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _compute_dynamic_compliances(blocks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute surrogate compliance proxies from block matrices via area-averaged weights."""
    m = blocks["Uxx"].shape[1]
    weights = np.ones(m, dtype=float) / float(m)

    cxx = np.einsum("sij,i,j->s", blocks["Uxx"], weights, weights)
    cmm = np.einsum("sij,i,j->s", blocks["Uzz"], weights, weights)
    cxm = 0.5 * (
        np.einsum("sij,i,j->s", blocks["Uxz"], weights, weights)
        + np.einsum("sij,i,j->s", blocks["Uzx"], weights, weights)
    )
    return {"CXX": cxx, "CMM": cmm, "CXM": cxm}


def _normalize_by_reference(values: np.ndarray) -> np.ndarray:
    ref = values[0]
    if np.abs(ref) < 1e-14:
        return values
    return values / ref


def _plot_compliance_curves(
    axis_values: np.ndarray,
    truth_blocks: dict[str, np.ndarray],
    pred_blocks: dict[str, np.ndarray],
    axis_label: str,
    save_path: Path,
) -> None:
    truth_c = _compute_dynamic_compliances(truth_blocks)
    pred_c = _compute_dynamic_compliances(pred_blocks)

    curves = ["CXX", "CMM", "CXM"]
    fig, axes = plt.subplots(2, len(curves), figsize=(5.0 * len(curves), 7.0), constrained_layout=True)
    order = np.argsort(axis_values)
    x = axis_values[order]

    for col, name in enumerate(curves):
        y_true = truth_c[name][order]
        y_pred = pred_c[name][order]

        axes[0, col].plot(x, np.real(y_true), label="truth", lw=1.4)
        axes[0, col].plot(x, np.real(y_pred), label="pred", lw=1.2, linestyle="--")
        axes[0, col].set_title(f"Re({name})")
        axes[0, col].set_xlabel(axis_label)
        axes[0, col].grid(alpha=0.25)
        axes[0, col].legend()

        axes[1, col].plot(x, np.imag(y_true), label="truth", lw=1.4)
        axes[1, col].plot(x, np.imag(y_pred), label="pred", lw=1.2, linestyle="--")
        axes[1, col].set_title(f"Im({name})")
        axes[1, col].set_xlabel(axis_label)
        axes[1, col].grid(alpha=0.25)
        axes[1, col].legend()

    fig.suptitle("Dynamic compliance proxies from full influence matrix")
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _plot_reference_compliances_by_case(
    axis_values_all: np.ndarray,
    full_matrix_all: np.ndarray,
    case_labels_all: np.ndarray,
    axis_label: str,
    save_path: Path,
) -> None:
    blocks_all = _split_blocks(full_matrix_all)
    comps_all = _compute_dynamic_compliances(blocks_all)

    labels = np.asarray(case_labels_all).astype(str)
    cases = [c for c in sorted(set(labels.tolist())) if c and c.lower() != "random"]
    if not cases:
        logger.info("No paper-case labels found; skipping case-wise reference compliance plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    comp_names = ["CXX", "CMM", "CXM"]

    for case in cases:
        mask = labels == case
        if int(np.sum(mask)) < 2:
            continue
        x = np.asarray(axis_values_all[mask], dtype=float)
        order = np.argsort(x)
        x = x[order]
        for col, cname in enumerate(comp_names):
            y = comps_all[cname][mask][order]
            y_n = _normalize_by_reference(y)
            axes[0, col].plot(x, np.real(y_n), lw=1.3, label=case)
            axes[1, col].plot(x, np.imag(y_n), lw=1.3, label=case)

    for col, cname in enumerate(comp_names):
        axes[0, col].set_title(f"Re({cname}/{cname}_ref)")
        axes[1, col].set_title(f"Im({cname}/{cname}_ref)")
        axes[0, col].set_xlabel(axis_label)
        axes[1, col].set_xlabel(axis_label)
        axes[0, col].grid(alpha=0.25)
        axes[1, col].grid(alpha=0.25)
        handles0, labels0 = axes[0, col].get_legend_handles_labels()
        handles1, labels1 = axes[1, col].get_legend_handles_labels()
        if handles0:
            axes[0, col].legend()
        if handles1:
            axes[1, col].legend()

    fig.suptitle("Paper-style case comparison (normalized by lowest sampled frequency per case)")
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _plot_prediction_compliances_by_case(
    axis_values: np.ndarray,
    truth_blocks: dict[str, np.ndarray],
    pred_blocks: dict[str, np.ndarray],
    case_labels_test: np.ndarray | None,
    axis_label: str,
    save_path: Path,
) -> None:
    if case_labels_test is None:
        return
    labels = np.asarray(case_labels_test).astype(str)
    cases = [c for c in sorted(set(labels.tolist())) if c and c.lower() != "random"]
    if not cases:
        return

    truth_c = _compute_dynamic_compliances(truth_blocks)
    pred_c = _compute_dynamic_compliances(pred_blocks)
    comp_names = ["CXX", "CMM", "CXM"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for case in cases:
        mask = labels == case
        if int(np.sum(mask)) < 2:
            continue
        x = np.asarray(axis_values[mask], dtype=float)
        order = np.argsort(x)
        x = x[order]
        for col, cname in enumerate(comp_names):
            y_true = truth_c[cname][mask][order]
            y_pred = pred_c[cname][mask][order]
            y_true_n = _normalize_by_reference(y_true)
            y_pred_n = y_pred / (y_true[0] if np.abs(y_true[0]) > 1e-14 else 1.0)
            axes[col].plot(x, np.real(y_true_n), lw=1.4, label=f"{case} truth")
            axes[col].plot(x, np.real(y_pred_n), lw=1.2, linestyle="--", label=f"{case} pred")

    for col, cname in enumerate(comp_names):
        axes[col].set_title(f"Re({cname}/{cname}_ref)")
        axes[col].set_xlabel(axis_label)
        axes[col].grid(alpha=0.25)
        handles, labels = axes[col].get_legend_handles_labels()
        if handles:
            axes[col].legend(fontsize=8, ncol=2)
    fig.suptitle("Model vs reference by paper case (test split)")
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _plot_sample_full_matrix_heatmaps(
    truth_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    axis_values: np.ndarray,
    save_dir: Path,
    max_samples: int,
    axis_label: str,
) -> None:
    truth_u = _to_scalar_complex_field(truth_matrix)
    pred_u = _to_scalar_complex_field(pred_matrix)
    n_samples = truth_u.shape[0]

    chosen = list(range(min(max_samples, n_samples)))
    for sample_idx in chosen:
        true_abs = np.abs(truth_u[sample_idx])
        pred_abs = np.abs(pred_u[sample_idx])
        rel_err = np.abs(pred_abs - true_abs) / np.maximum(np.max(true_abs), 1e-12)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        im0 = axes[0].imshow(true_abs, origin="lower", cmap="magma")
        axes[0].set_title("|U| truth")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(pred_abs, origin="lower", cmap="magma")
        axes[1].set_title("|U| pred")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(rel_err, origin="lower", cmap="viridis")
        axes[2].set_title("rel. error")
        fig.colorbar(im2, ax=axes[2], fraction=0.046)

        for ax in axes:
            ax.set_xlabel("source dof index")
            ax.set_ylabel("receiver dof index")

        fig.suptitle(f"Sample {sample_idx} ({axis_label}={axis_values[sample_idx]:.3f})")
        fig.savefig(save_dir / f"sample_{sample_idx:03d}_full_matrix_heatmaps.png", dpi=180)
        plt.close(fig)


def _save_formulation_report(
    save_path: Path,
    properties_all: np.ndarray,
    axis_values: np.ndarray,
    truth_blocks: dict[str, np.ndarray],
    pred_blocks: dict[str, np.ndarray],
    axis_label: str,
    case_labels_all: np.ndarray | None = None,
) -> None:
    block_errors = {
        name: float(np.mean(_relative_error_per_sample(truth_blocks[name], pred_blocks[name])))
        for name in ["Uxx", "Uxz", "Uzx", "Uzz"]
    }

    dataset_summary = {
        "num_samples": int(properties_all.shape[0]),
        f"{axis_label}_min": float(np.min(axis_values)),
        f"{axis_label}_max": float(np.max(axis_values)),
        "eta_range_medium1": [float(np.min(properties_all[:, 0, 5])), float(np.max(properties_all[:, 0, 5]))],
        "eta_range_medium2": [float(np.min(properties_all[:, 1, 5])), float(np.max(properties_all[:, 1, 5]))],
    }
    if case_labels_all is not None:
        labels = np.asarray(case_labels_all).astype(str)
        dataset_summary["paper_case_counts"] = {
            str(label): int(np.sum(labels == label))
            for label in sorted(set(labels.tolist()))
        }

    report = {
        "formulation": "Full Influence Matrix Learning --- Non-Homogeneous Soil",
        "input_definition": {
            "dim": 15,
            "layout": [
                "c11_1",
                "c12_1",
                "c13_1",
                "c33_1",
                "c44_1",
                "eta_1",
                "rho_1",
                "c11_2",
                "c12_2",
                "c13_2",
                "c33_2",
                "c44_2",
                "eta_2",
                "rho_2",
                "a0",
            ],
        },
        "output_definition": {
            "matrix_shape": [int(truth_blocks["Uxx"].shape[1] * 2), int(truth_blocks["Uxx"].shape[2] * 2)],
            "blocks": ["Uxx", "Uxz", "Uzx", "Uzz"],
            "complex": True,
        },
        "dataset_summary": dataset_summary,
        "block_mean_relative_errors": block_errors,
        "paper_reference": {
            "labaki_2013_reported_time_s_for_Uzz_M20": 310.0,
            "note": "Reference time is Uzz-only matrix fill; current target is full coupled U matrix.",
        },
    }

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, sort_keys=False, allow_unicode=True)


def run_all_multilayer_plots(
    plots_path: Path,
    properties_all: np.ndarray,
    omega_all: np.ndarray,
    raw_u_all: np.ndarray | None,
    case_labels_all: np.ndarray | None,
    truth_test_matrix: np.ndarray,
    pred_test_matrix: np.ndarray,
    test_indices: np.ndarray,
    case_labels_test: np.ndarray | None,
    b_value: float,
    config: dict[str, Any],
    channel_names: list[str] | None = None,
) -> None:
    del b_value, channel_names

    paths = _ensure_dirs(plots_path)
    axis_label = str(config.get("frequency_axis_label", "a0"))
    max_samples = int(config.get("plot_max_samples", 3))

    axis_values_all = np.asarray(omega_all, dtype=float)
    axis_values = axis_values_all[np.asarray(test_indices, dtype=int)]

    truth_blocks = _split_blocks(truth_test_matrix)
    pred_blocks = _split_blocks(pred_test_matrix)

    _save_formulation_report(
        save_path=paths["alignment"] / "formulation_alignment.yaml",
        properties_all=np.asarray(properties_all, dtype=float),
        axis_values=axis_values_all,
        truth_blocks=truth_blocks,
        pred_blocks=pred_blocks,
        axis_label=axis_label,
        case_labels_all=case_labels_all,
    )

    _plot_block_mean_heatmaps(
        truth_blocks=truth_blocks,
        pred_blocks=pred_blocks,
        save_path=paths["alignment"] / "block_mean_heatmaps.png",
    )

    _plot_compliance_curves(
        axis_values=axis_values,
        truth_blocks=truth_blocks,
        pred_blocks=pred_blocks,
        axis_label=axis_label,
        save_path=paths["profiles"] / "dynamic_compliance_proxies.png",
    )

    if raw_u_all is not None and case_labels_all is not None:
        _plot_reference_compliances_by_case(
            axis_values_all=axis_values_all,
            full_matrix_all=np.asarray(raw_u_all),
            case_labels_all=np.asarray(case_labels_all),
            axis_label=axis_label,
            save_path=paths["profiles"] / "paper_case_reference_compliances.png",
        )
    _plot_prediction_compliances_by_case(
        axis_values=axis_values,
        truth_blocks=truth_blocks,
        pred_blocks=pred_blocks,
        case_labels_test=case_labels_test,
        axis_label=axis_label,
        save_path=paths["profiles"] / "paper_case_prediction_compliances.png",
    )

    _plot_sample_full_matrix_heatmaps(
        truth_matrix=truth_test_matrix,
        pred_matrix=pred_test_matrix,
        axis_values=axis_values,
        save_dir=paths["heatmaps"],
        max_samples=max_samples,
        axis_label=axis_label,
    )

    logger.info("Saved formulation-aligned multilayer plots under %s", plots_path)
