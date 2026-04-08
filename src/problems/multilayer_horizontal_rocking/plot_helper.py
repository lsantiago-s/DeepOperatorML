from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

logger = logging.getLogger(__name__)

PAPER_MATERIALS: dict[str, dict[str, float]] = {
    # Table 2 in Labaki et al. (2014), values in terms of c'ij = cij / c44.
    # We keep c44=1.0 so c'ij and cij are numerically equal in this normalized setup.
    "m1": {"c11": 3.0000, "c12": 1.0000, "c13": 1.0000, "c33": 3.0000, "c44": 1.0000, "eta": 0.01, "rho": 1.0},
    "m2": {"c11": 2.8284, "c12": 0.8284, "c13": 0.8284, "c33": 4.2426, "c44": 1.0000, "eta": 0.01, "rho": 1.0},
    "m3": {"c11": 2.7749, "c12": 0.7749, "c13": 0.7749, "c33": 5.5497, "c44": 1.0000, "eta": 0.01, "rho": 1.0},
}

PAPER_CASES: dict[str, list[dict[str, Any]]] = {
    # Table 1: two finite layers + half-space.
    "A": [
        {"material": "m1", "h": 0.5},
        {"material": "m1", "h": 0.5},
        {"material": "m1", "h": 0.0},
    ],
    "B": [
        {"material": "m3", "h": 0.5},
        {"material": "m2", "h": 0.5},
        {"material": "m1", "h": 0.0},
    ],
    "C": [
        {"material": "m3", "h": 0.3},
        {"material": "m2", "h": 0.7},
        {"material": "m1", "h": 0.0},
    ],
}

DEFAULT_CHANNEL_NAMES = ["urfx", "uzfx", "urmy", "uzmy", "uzz"]
PARAM_NAMES = ["c11", "c12", "c13", "c33", "c44", "eta", "rho", "h"]


def _normalized_ratios(props: np.ndarray) -> np.ndarray:
    c44 = np.maximum(np.asarray(props[..., 4], dtype=float), 1e-12)
    ratios = np.stack(
        [
            np.asarray(props[..., 0], dtype=float) / c44,
            np.asarray(props[..., 1], dtype=float) / c44,
            np.asarray(props[..., 2], dtype=float) / c44,
            np.asarray(props[..., 3], dtype=float) / c44,
        ],
        axis=-1,
    )
    return ratios


def _case_distance(sample_props: np.ndarray, case_name: str) -> float:
    """
    Compare sample properties with paper case definitions in normalized space.
    Lower is better. Distance is mean relative mismatch over aligned layers.
    """
    case_layers = PAPER_CASES[case_name]
    n_common = min(sample_props.shape[0], len(case_layers))
    if n_common == 0:
        return np.inf

    dist_vals: list[float] = []
    for i in range(n_common):
        target_mat = PAPER_MATERIALS[case_layers[i]["material"]]

        target = np.array(
            [
                target_mat["c11"],
                target_mat["c12"],
                target_mat["c13"],
                target_mat["c33"],
                target_mat["c44"],
                target_mat["eta"],
                target_mat["rho"],
                float(case_layers[i]["h"]),
            ],
            dtype=float,
        )
        sample = np.asarray(sample_props[i], dtype=float)

        # Compare in c'ij space for stiffness terms.
        sample_c44 = max(sample[4], 1e-12)
        sample_norm = sample.copy()
        sample_norm[0:5] = sample_norm[0:5] / sample_c44

        # Weighted relative mismatch.
        scales = np.array([3.0, 1.0, 1.0, 3.0, 1.0, 0.01, 1.0, 0.5], dtype=float)
        dist = np.mean(np.abs(sample_norm - target) / scales)
        dist_vals.append(float(dist))

    return float(np.mean(dist_vals))


def classify_cases(properties: np.ndarray, tolerance: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    labels: list[str] = []
    distances: list[float] = []

    for sample_props in properties:
        dists = {case: _case_distance(sample_props, case) for case in PAPER_CASES.keys()}
        best_case = min(dists, key=dists.get)
        best_dist = dists[best_case]
        if best_dist <= tolerance:
            labels.append(best_case)
        else:
            labels.append("unknown")
        distances.append(best_dist)

    return np.asarray(labels), np.asarray(distances, dtype=float)


def annulus_weights(m: int, b: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    l = (1.0 - b) / m
    s1 = b + l * np.arange(m)
    s2 = b + l * (np.arange(m) + 1)
    centers = 0.5 * (s1 + s2)
    weights = np.pi * (s2**2 - s1**2)  # area weights for uniformly distributed load
    weights = weights / np.sum(weights)
    return centers, weights


def distributed_load_profiles(
    influence: np.ndarray,
    b: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    influence: (samples, M, M, channels), complex
    returns: (r_centers, profiles) with profiles shape (samples, M, channels)
    """
    m = influence.shape[1]
    r_centers, weights = annulus_weights(m=m, b=b)
    profiles = np.einsum("sijc,j->sic", influence, weights)
    return r_centers, profiles


def _make_dirs(base_path: Path) -> dict[str, Path]:
    paths = {
        "alignment": base_path / "paper_alignment",
        "profiles": base_path / "paper_profiles",
        "heatmaps": base_path / "prediction_heatmaps",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_compatibility_report(
    save_path: Path,
    properties: np.ndarray,
    omegas: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
) -> None:
    ratios = _normalized_ratios(properties)

    report = {
        "paper_reference": {
            "materials": PAPER_MATERIALS,
            "cases": PAPER_CASES,
        },
        "dataset": {
            "num_samples": int(properties.shape[0]),
            "num_layers_plus_semispace": int(properties.shape[1]),
            "omega_min": float(np.min(omegas)),
            "omega_max": float(np.max(omegas)),
            "ratio_ranges": {
                "c11_over_c44": [float(np.min(ratios[..., 0])), float(np.max(ratios[..., 0]))],
                "c12_over_c44": [float(np.min(ratios[..., 1])), float(np.max(ratios[..., 1]))],
                "c13_over_c44": [float(np.min(ratios[..., 2])), float(np.max(ratios[..., 2]))],
                "c33_over_c44": [float(np.min(ratios[..., 3])), float(np.max(ratios[..., 3]))],
            },
            "eta_range": [float(np.min(properties[..., 5])), float(np.max(properties[..., 5]))],
            "rho_range": [float(np.min(properties[..., 6])), float(np.max(properties[..., 6]))],
            "h_range": [float(np.min(properties[..., 7])), float(np.max(properties[..., 7]))],
        },
        "case_classification": {
            "A": int(np.sum(labels == "A")),
            "B": int(np.sum(labels == "B")),
            "C": int(np.sum(labels == "C")),
            "unknown": int(np.sum(labels == "unknown")),
            "mean_distance_to_best_case": float(np.mean(distances)),
            "median_distance_to_best_case": float(np.median(distances)),
        },
        "notes": [
            "Paper reference uses vertical plate response (w, Mr, Q).",
            "Current solver channels are URFx, UZFx, URMy, UZMy, UZZ (soil influence channels).",
            "Compatibility here is measured in material-layer parameter space only.",
        ],
    }

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, sort_keys=False, allow_unicode=True)


def plot_ratio_coverage(properties: np.ndarray, save_path: Path) -> None:
    ratios = _normalized_ratios(properties).reshape(-1, 4)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    ratio_labels = ["c11/c44", "c12/c44", "c13/c44", "c33/c44"]

    material_targets = {
        "m1": [PAPER_MATERIALS["m1"]["c11"], PAPER_MATERIALS["m1"]["c12"], PAPER_MATERIALS["m1"]["c13"], PAPER_MATERIALS["m1"]["c33"]],
        "m2": [PAPER_MATERIALS["m2"]["c11"], PAPER_MATERIALS["m2"]["c12"], PAPER_MATERIALS["m2"]["c13"], PAPER_MATERIALS["m2"]["c33"]],
        "m3": [PAPER_MATERIALS["m3"]["c11"], PAPER_MATERIALS["m3"]["c12"], PAPER_MATERIALS["m3"]["c13"], PAPER_MATERIALS["m3"]["c33"]],
    }

    for i, ax in enumerate(axes.flat):
        ax.hist(ratios[:, i], bins=25, color="#4C72B0", alpha=0.85)
        for mat_name, color in zip(["m1", "m2", "m3"], ["#55A868", "#C44E52", "#8172B2"]):
            ax.axvline(material_targets[mat_name][i], color=color, linestyle="--", linewidth=1.6, label=mat_name)
        ax.set_title(ratio_labels[i])
        ax.set_xlabel("value")
        ax.set_ylabel("count")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Dataset vs paper baseline ratios (Table 2)")
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _resolve_channel_names(num_channels: int, channel_names: list[str] | None = None) -> list[str]:
    if channel_names is not None and len(channel_names) == num_channels:
        return [str(c) for c in channel_names]
    if num_channels <= len(DEFAULT_CHANNEL_NAMES):
        return DEFAULT_CHANNEL_NAMES[:num_channels]
    return [f"ch{idx}" for idx in range(num_channels)]


def plot_frequency_sweep(
    omegas: np.ndarray,
    case_labels: np.ndarray,
    profiles: np.ndarray,
    save_path: Path,
    channel_names: list[str] | None = None,
) -> None:
    """Plot |response(r~0)| vs frequency for each channel and case."""
    r0_response = np.abs(profiles[:, 0, :])  # (samples, channels)
    n_channels = r0_response.shape[1]
    names = _resolve_channel_names(n_channels, channel_names)
    ncols = 3 if n_channels > 4 else 2
    nrows = int(np.ceil(n_channels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows), constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)

    for c_idx in range(n_channels):
        ax = axes_flat[c_idx]
        for case_name, color in [("A", "#1f77b4"), ("B", "#ff7f0e"), ("C", "#2ca02c")]:
            mask = case_labels == case_name
            if not np.any(mask):
                continue
            x = omegas[mask]
            y = r0_response[mask, c_idx]
            order = np.argsort(x)
            ax.plot(x[order], y[order], "o-", markersize=3.0, linewidth=1.2, color=color, label=f"Case {case_name}")

        ax.set_title(names[c_idx])
        ax.set_xlabel("frequency (input omega)")
        ax.set_ylabel("|response| at first receiver ring")
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

    for k in range(n_channels, len(axes_flat)):
        axes_flat[k].axis("off")

    fig.suptitle("Paper-style frequency comparison by layered case")
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_radial_profiles_by_case(
    r_centers: np.ndarray,
    omegas: np.ndarray,
    case_labels: np.ndarray,
    truth_profiles: np.ndarray,
    pred_profiles: np.ndarray | None,
    target_freqs: list[float],
    save_dir: Path,
    channel_names: list[str] | None = None,
) -> None:
    """
    For each target frequency and case, pick nearest sample and plot radial profiles.
    """
    for target_freq in target_freqs:
        n_channels = truth_profiles.shape[2]
        names = _resolve_channel_names(n_channels, channel_names)
        ncols = 3 if n_channels > 4 else 2
        nrows = int(np.ceil(n_channels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows), constrained_layout=True)
        axes_flat = np.asarray(axes).reshape(-1)

        for c_idx in range(n_channels):
            ax = axes_flat[c_idx]
            for case_name, color in [("A", "#1f77b4"), ("B", "#ff7f0e"), ("C", "#2ca02c")]:
                mask = case_labels == case_name
                if not np.any(mask):
                    continue
                case_indices = np.where(mask)[0]
                nearest = case_indices[np.argmin(np.abs(omegas[case_indices] - target_freq))]

                y_true = np.abs(truth_profiles[nearest, :, c_idx])
                ax.plot(r_centers, y_true, color=color, linewidth=1.8, label=f"Case {case_name} truth")

                if pred_profiles is not None and nearest < pred_profiles.shape[0]:
                    y_pred = np.abs(pred_profiles[nearest, :, c_idx])
                    ax.plot(r_centers, y_pred, color=color, linestyle="--", linewidth=1.4, label=f"Case {case_name} pred")

            ax.set_title(names[c_idx])
            ax.set_xlabel("normalized radius r/a")
            ax.set_ylabel("|response| under distributed load")
            ax.grid(alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)

        for k in range(n_channels, len(axes_flat)):
            axes_flat[k].axis("off")

        fig.suptitle(f"Paper-style radial profiles by case (target omega={target_freq:.3f})")
        out_file = save_dir / f"radial_profiles_omega_{target_freq:.3f}.png"
        fig.savefig(out_file, dpi=180)
        plt.close(fig)


def plot_prediction_heatmaps(
    truth_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    case_labels: np.ndarray,
    save_dir: Path,
    max_samples: int = 3,
    channel_names: list[str] | None = None,
) -> None:
    """
    truth/pred heatmaps for representative samples; one figure per sample.
    matrix shape: (samples, M, M, channels), complex.
    """
    chosen: list[int] = []
    for case_name in ["A", "B", "C"]:
        idxs = np.where(case_labels == case_name)[0]
        if len(idxs) > 0:
            chosen.append(int(idxs[0]))

    if len(chosen) == 0:
        chosen = list(range(min(max_samples, truth_matrix.shape[0])))
    else:
        chosen = chosen[:max_samples]

    n_channels = truth_matrix.shape[-1]
    names = _resolve_channel_names(n_channels, channel_names)

    for sample_idx in chosen:
        fig, axes = plt.subplots(n_channels, 3, figsize=(12, max(3.0 * n_channels, 6.0)), constrained_layout=True)
        axes_arr = np.asarray(axes)
        if axes_arr.ndim == 1:
            axes_arr = axes_arr[None, :]

        for c_idx in range(n_channels):
            true_abs = np.abs(truth_matrix[sample_idx, :, :, c_idx])
            pred_abs = np.abs(pred_matrix[sample_idx, :, :, c_idx])
            denom = np.maximum(true_abs.max(), 1e-12)
            rel_err = np.abs(pred_abs - true_abs) / denom

            im0 = axes_arr[c_idx, 0].imshow(true_abs, origin="lower", cmap="magma")
            axes_arr[c_idx, 0].set_title(f"{names[c_idx]} |truth|")
            fig.colorbar(im0, ax=axes_arr[c_idx, 0], fraction=0.046)

            im1 = axes_arr[c_idx, 1].imshow(pred_abs, origin="lower", cmap="magma")
            axes_arr[c_idx, 1].set_title(f"{names[c_idx]} |pred|")
            fig.colorbar(im1, ax=axes_arr[c_idx, 1], fraction=0.046)

            im2 = axes_arr[c_idx, 2].imshow(rel_err, origin="lower", cmap="viridis")
            axes_arr[c_idx, 2].set_title(f"{names[c_idx]} rel. err")
            fig.colorbar(im2, ax=axes_arr[c_idx, 2], fraction=0.046)

            for col in range(3):
                axes_arr[c_idx, col].set_xlabel("source ring index")
                axes_arr[c_idx, col].set_ylabel("receiver ring index")

        case_tag = case_labels[sample_idx] if sample_idx < len(case_labels) else "unknown"
        fig.suptitle(f"Sample {sample_idx} (case={case_tag}) - truth/prediction heatmaps")
        fig.savefig(save_dir / f"sample_{sample_idx:03d}_truth_pred_heatmaps.png", dpi=180)
        plt.close(fig)


def run_all_multilayer_plots(
    plots_path: Path,
    properties_all: np.ndarray,
    omega_all: np.ndarray,
    truth_test_matrix: np.ndarray,
    pred_test_matrix: np.ndarray,
    test_indices: np.ndarray,
    b_value: float,
    config: dict[str, Any],
    channel_names: list[str] | None = None,
) -> None:
    paths = _make_dirs(plots_path)

    case_tol = float(config.get("paper_case_distance_tolerance", 0.35))
    target_freqs = [float(v) for v in config.get("paper_target_frequencies", [0.001, 4.0])]
    do_alignment = bool(config.get("plot_paper_alignment", True))
    do_profiles = bool(config.get("plot_paper_profiles", True))
    do_heatmaps = bool(config.get("plot_prediction_heatmaps", True))

    labels_all, distances_all = classify_cases(properties=properties_all, tolerance=case_tol)
    if do_alignment:
        save_compatibility_report(
            save_path=paths["alignment"] / "paper_baseline_compatibility.yaml",
            properties=properties_all,
            omegas=omega_all,
            labels=labels_all,
            distances=distances_all,
        )
        plot_ratio_coverage(
            properties=properties_all,
            save_path=paths["alignment"] / "paper_ratio_coverage.png",
        )

    if len(test_indices) == 0:
        logger.warning("No test indices found; skipping paper-style profile plots.")
        return

    test_indices = np.asarray(test_indices, dtype=int)
    properties_test = properties_all[test_indices]
    omega_test = omega_all[test_indices]
    case_labels_test, _ = classify_cases(properties=properties_test, tolerance=case_tol)

    r_centers, truth_profiles = distributed_load_profiles(truth_test_matrix, b=b_value)
    _, pred_profiles = distributed_load_profiles(pred_test_matrix, b=b_value)

    if do_profiles:
        plot_frequency_sweep(
            omegas=omega_test,
            case_labels=case_labels_test,
            profiles=truth_profiles,
            save_path=paths["profiles"] / "frequency_sweep_by_case.png",
            channel_names=channel_names,
        )

        plot_radial_profiles_by_case(
            r_centers=r_centers,
            omegas=omega_test,
            case_labels=case_labels_test,
            truth_profiles=truth_profiles,
            pred_profiles=pred_profiles,
            target_freqs=target_freqs,
            save_dir=paths["profiles"],
            channel_names=channel_names,
        )

    if do_heatmaps:
        plot_prediction_heatmaps(
            truth_matrix=truth_test_matrix,
            pred_matrix=pred_test_matrix,
            case_labels=case_labels_test,
            save_dir=paths["heatmaps"],
            max_samples=int(config.get("plot_max_samples", 3)),
            channel_names=channel_names,
        )
