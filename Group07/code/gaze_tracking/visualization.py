from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from gaze_tracking.aoi import AOI


def save_visualizations(
    gaze_log: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    aois: list[AOI],
    output_dir: Path,
    width: int,
    height: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_trajectory(gaze_log, aois, output_dir / "trajectory.png", width, height)
    save_heatmap(gaze_log, output_dir / "gaze_heatmap.png", width, height)
    save_transition_heatmap(transition_matrix, output_dir / "transition_matrix.png")


def save_trajectory(
    gaze_log: pd.DataFrame,
    aois: list[AOI],
    path: Path,
    width: int,
    height: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title("Gaze Trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for aoi in aois:
        rect = plt.Rectangle(
            (aoi.x1, aoi.y1),
            aoi.x2 - aoi.x1,
            aoi.y2 - aoi.y1,
            fill=False,
            linewidth=0.8,
            color="gray",
        )
        ax.add_patch(rect)
        ax.text((aoi.x1 + aoi.x2) / 2, (aoi.y1 + aoi.y2) / 2, aoi.id, ha="center", va="center")

    if not gaze_log.empty:
        ax.plot(gaze_log["gaze_x"], gaze_log["gaze_y"], marker="o", markersize=2, linewidth=0.8)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_heatmap(gaze_log: pd.DataFrame, path: Path, width: int, height: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title("Gaze Heatmap")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if len(gaze_log) >= 3 and gaze_log["gaze_x"].nunique() > 1 and gaze_log["gaze_y"].nunique() > 1:
        sns.kdeplot(
            data=gaze_log,
            x="gaze_x",
            y="gaze_y",
            fill=True,
            cmap="mako",
            levels=30,
            thresh=0.05,
            ax=ax,
        )
    elif not gaze_log.empty:
        ax.scatter(gaze_log["gaze_x"], gaze_log["gaze_y"], s=10, alpha=0.65)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_transition_heatmap(transition_matrix: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("AOI Transition Matrix")

    if transition_matrix.empty:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center")
        ax.axis("off")
    else:
        sns.heatmap(transition_matrix, annot=True, fmt="d", cmap="viridis", ax=ax)
        ax.set_xlabel("Target AOI")
        ax.set_ylabel("Source AOI")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
