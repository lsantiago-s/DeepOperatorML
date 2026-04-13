from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(raw_data_path: Path) -> dict[str, np.ndarray]:
    data = np.load(raw_data_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _plot_branch_heatmap(xb: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
    im = ax.imshow(xb, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("Input feature index")
    ax.set_ylabel("Sample index")
    ax.set_title("Branch input matrix (xb)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(out / "branch_input_heatmap.png", dpi=180)
    plt.close(fig)


def _plot_layer_properties(properties: np.ndarray, out: Path) -> None:
    # properties shape: (samples, N+1, 8)
    n_rows = properties.shape[1]
    names = ["c11", "c12", "c13", "c33", "c44", "eta", "rho", "h"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)

    for idx, name in enumerate(names):
        ax = axes[idx // 4, idx % 4]
        for row in range(n_rows):
            vals = properties[:, row, idx]
            ax.plot(np.sort(vals), label=f"row {row+1}", lw=1.2)
        ax.set_title(name)
        ax.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Sorted property values per layer/half-space row")
    fig.savefig(out / "layer_property_profiles.png", dpi=180)
    plt.close(fig)


def _plot_profile_tensor(profiles: np.ndarray, z_grid: np.ndarray | None, out: Path) -> None:
    if profiles.ndim != 3 or profiles.shape[1] < 1:
        return

    names = ["c11", "c12", "c13", "c33", "c44", "rho", "eta"][: profiles.shape[1]]
    rows = int(np.ceil(len(names) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(14, 3.6 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, 3)

    x = np.arange(profiles.shape[2], dtype=float) if z_grid is None else np.asarray(z_grid, dtype=float)
    x_label = "depth index" if z_grid is None else "z/a"
    for idx, name in enumerate(names):
        ax = axes[idx // 3, idx % 3]
        im = ax.imshow(profiles[:, idx, :], aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"{name}(z)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("sample")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for idx in range(len(names), rows * 3):
        axes[idx // 3, idx % 3].axis("off")

    fig.suptitle("Depth-profile encoding across samples")
    fig.savefig(out / "profile_encoding_heatmaps.png", dpi=180)
    plt.close(fig)


def _plot_sample_matrix(g_u: np.ndarray, out: Path) -> None:
    if g_u.ndim == 3 and g_u.shape[-1] == 4:
        n = int(round(np.sqrt(g_u.shape[1])))
        if n * n != g_u.shape[1]:
            return
        ch = g_u[0].reshape(n, n, 4)
        block_names = ["Uxx", "Uxz", "Uzx", "Uzz"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for ax, block_idx in zip(axes.flat, range(4), strict=True):
            im = ax.imshow(np.abs(ch[..., block_idx]), origin="lower", cmap="magma")
            ax.set_title(f"|{block_names[block_idx]}| sample 0")
            ax.set_xlabel("source element")
            ax.set_ylabel("receiver element")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.savefig(out / "sample0_blocks_abs.png", dpi=180)
        plt.close(fig)
        return

    if g_u.ndim != 2:
        return
    n = int(round(np.sqrt(g_u.shape[1])))
    if n * n != g_u.shape[1]:
        return
    u0 = g_u[0].reshape(n, n)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im0 = axes[0].imshow(np.abs(u0), origin="lower", cmap="magma")
    axes[0].set_title("|U| sample 0")
    axes[0].set_xlabel("source dof")
    axes[0].set_ylabel("receiver dof")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(np.angle(u0), origin="lower", cmap="twilight")
    axes[1].set_title("arg(U) sample 0")
    axes[1].set_xlabel("source dof")
    axes[1].set_ylabel("receiver dof")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.savefig(out / "sample0_full_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sanity plots for vertical_layered_soil raw dataset.")
    parser.add_argument("--raw-data", required=True, help="Path to raw .npz")
    parser.add_argument(
        "--output-dir",
        default="output/vertical_layered_soil/data_sanity",
        help="Directory to save sanity plots",
    )
    args = parser.parse_args()

    raw_data_path = Path(args.raw_data).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load(raw_data_path)

    xb = np.asarray(data["xb"], dtype=float)
    properties = np.asarray(data["properties"], dtype=float)
    g_u = np.asarray(data["g_u"])
    profiles = np.asarray(data["profiles"], dtype=float) if "profiles" in data else None
    z_grid = np.asarray(data["z"], dtype=float) if "z" in data else None

    _plot_branch_heatmap(xb=xb, out=out_dir)
    _plot_layer_properties(properties=properties, out=out_dir)
    if profiles is not None:
        _plot_profile_tensor(profiles=profiles, z_grid=z_grid, out=out_dir)
    _plot_sample_matrix(g_u=g_u, out=out_dir)

    outputs = [
        "branch_input_heatmap.png",
        "layer_property_profiles.png",
        "profile_encoding_heatmaps.png" if profiles is not None else None,
        "sample0_blocks_abs.png" if (g_u.ndim == 3 and g_u.shape[-1] == 4) else "sample0_full_matrix.png",
    ]
    report = {
        "raw_data": str(raw_data_path),
        "num_samples": int(xb.shape[0]),
        "xb_shape": list(xb.shape),
        "properties_shape": list(properties.shape),
        "g_u_shape": list(g_u.shape),
        "profiles_shape": list(profiles.shape) if profiles is not None else None,
        "outputs": [x for x in outputs if x is not None],
    }
    with open(out_dir / "sanity_report.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, sort_keys=False)

    print(f"Saved sanity plots to {out_dir}")


if __name__ == "__main__":
    main()
