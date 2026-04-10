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


def _plot_sample_matrix(g_u: np.ndarray, out: Path) -> None:
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

    _plot_branch_heatmap(xb=xb, out=out_dir)
    _plot_layer_properties(properties=properties, out=out_dir)
    _plot_sample_matrix(g_u=g_u, out=out_dir)

    report = {
        "raw_data": str(raw_data_path),
        "num_samples": int(xb.shape[0]),
        "xb_shape": list(xb.shape),
        "properties_shape": list(properties.shape),
        "g_u_shape": list(g_u.shape),
        "outputs": [
            "branch_input_heatmap.png",
            "layer_property_profiles.png",
            "sample0_full_matrix.png",
        ],
    }
    with open(out_dir / "sanity_report.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, sort_keys=False)

    print(f"Saved sanity plots to {out_dir}")


if __name__ == "__main__":
    main()
