from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from gaze_tracking.calibration import fit_poly2_calibration
from gaze_tracking.gaze_estimator import GazeEstimator


def run_interactive_calibration(
    output_dir: Path,
    calibration_path: Path,
    screen_width: int = 1280,
    screen_height: int = 720,
    camera_index: int = 0,
    samples_per_point: int = 30,
    model_path: Path | None = None,
    device: str = "auto",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    points = build_calibration_targets(screen_width, screen_height)
    estimator = GazeEstimator(
        model_path=model_path,
        device=device,
        smooth_window=3,
    )
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {camera_index}")

    point_index = 0
    is_sampling = False
    current_samples: list[dict] = []
    accepted_samples: list[dict] = []

    try:
        cv2.namedWindow("Gaze Calibration", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Gaze Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while point_index < len(points):
            ok, frame = cap.read()
            if not ok:
                break

            estimate, _ = estimator.estimate(frame)
            target_x, target_y = points[point_index]

            if is_sampling and estimate is not None:
                current_samples.append(
                    {
                        "point_index": point_index + 1,
                        "target_x": target_x,
                        "target_y": target_y,
                        "yaw_deg": estimate.yaw_deg,
                        "pitch_deg": estimate.pitch_deg,
                        "confidence": estimate.confidence,
                        "timestamp_ms": time.perf_counter() * 1000.0,
                    }
                )
                if len(current_samples) >= samples_per_point:
                    accepted_samples.append(_summarize_point_samples(current_samples))
                    point_index += 1
                    current_samples = []
                    is_sampling = False
                    continue

            canvas = _build_calibration_canvas(
                screen_width,
                screen_height,
                points,
                point_index,
                len(current_samples),
                samples_per_point,
                is_sampling,
                estimate is not None,
            )
            cv2.imshow("Gaze Calibration", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                current_samples = []
                is_sampling = False
            if key == ord(" "):
                current_samples = []
                is_sampling = True
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(accepted_samples) != len(points):
        raise RuntimeError(
            f"calibration cancelled or incomplete: collected {len(accepted_samples)} of {len(points)} points"
        )

    calibration = fit_poly2_calibration(accepted_samples, screen_width, screen_height)
    calibration.save(calibration_path)
    pd.DataFrame(accepted_samples).to_csv(output_dir / "calibration_samples.csv", index=False)


def build_calibration_targets(width: int, height: int) -> list[tuple[int, int]]:
    margin_x = int(width * 0.08)
    margin_y = int(height * 0.08)
    return _grid_targets(width, height, rows=4, cols=4, margin_x=margin_x, margin_y=margin_y)


def _summarize_point_samples(samples: list[dict]) -> dict:
    yaw_deg = float(np.median([item["yaw_deg"] for item in samples]))
    pitch_deg = float(np.median([item["pitch_deg"] for item in samples]))
    confidence = float(np.mean([item["confidence"] for item in samples]))
    first = samples[0]
    return {
        "point_index": int(first["point_index"]),
        "target_x": float(first["target_x"]),
        "target_y": float(first["target_y"]),
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "confidence": confidence,
        "sample_count": len(samples),
    }


def _grid_targets(
    width: int,
    height: int,
    rows: int,
    cols: int,
    margin_x: int,
    margin_y: int,
) -> list[tuple[int, int]]:
    xs = np.linspace(margin_x, width - margin_x, cols)
    ys = np.linspace(margin_y, height - margin_y, rows)
    return [(int(round(x)), int(round(y))) for y in ys for x in xs]


def _build_calibration_canvas(
    width: int,
    height: int,
    points: list[tuple[int, int]],
    point_index: int,
    sample_count: int,
    samples_per_point: int,
    is_sampling: bool,
    has_gaze: bool,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)

    for idx, point in enumerate(points):
        color = (160, 160, 160)
        radius = 8
        thickness = 2
        if idx < point_index:
            color = (80, 180, 80)
            radius = 9
            thickness = -1
        elif idx == point_index:
            color = (40, 40, 230)
            radius = 16
            thickness = 3
        cv2.circle(canvas, point, radius, color, thickness)
        cv2.line(canvas, (point[0] - 24, point[1]), (point[0] + 24, point[1]), color, 2)
        cv2.line(canvas, (point[0], point[1] - 24), (point[0], point[1] + 24), color, 2)

    progress = f"Point {point_index + 1}/{len(points)} | samples {sample_count}/{samples_per_point}"
    if is_sampling and has_gaze:
        state = "Sampling... keep looking at the red target"
    elif is_sampling:
        state = "Waiting for gaze detection"
    else:
        state = "Look at the red target, press SPACE"
    gaze_state = "gaze detected" if has_gaze else "no gaze detected"

    cv2.putText(canvas, progress, (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (30, 30, 30), 2)
    cv2.putText(canvas, state, (32, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (30, 30, 30), 2)
    cv2.putText(canvas, "SPACE: collect | r: retry point | q: quit", (32, height - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2)
    cv2.putText(canvas, gaze_state, (width - 250, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 120, 80) if has_gaze else (50, 80, 230), 2)

    if sample_count > 0:
        bar_w = int((width - 64) * sample_count / max(1, samples_per_point))
        cv2.rectangle(canvas, (32, 112), (32 + bar_w, 132), (80, 180, 80), -1)
        cv2.rectangle(canvas, (32, 112), (width - 32, 132), (120, 120, 120), 2)

    return canvas
