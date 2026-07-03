from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from gaze_tracking.analyzer import analyze_gaze_log
from gaze_tracking.aoi import AOI, build_grid_aois, find_aoi
from gaze_tracking.calibration import load_calibration
from gaze_tracking.visualization import save_visualizations


def run_webcam(
    output_dir: Path,
    screen_width: int = 1280,
    screen_height: int = 720,
    camera_index: int = 0,
    rows: int = 3,
    cols: int = 3,
    min_fixation_ms: float = 120.0,
    calibration_path: Path | None = None,
    model_path: Path | None = None,
    device: str = "auto",
    max_frames: int | None = None,
) -> None:
    from gaze_tracking.gaze_estimator import GazeEstimator

    output_dir.mkdir(parents=True, exist_ok=True)
    aois = build_grid_aois(screen_width, screen_height, rows=rows, cols=cols)
    calibration = load_calibration(calibration_path, screen_width, screen_height)
    estimator = GazeEstimator(
        model_path=model_path,
        device=device,
    )
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {camera_index}")

    start = time.perf_counter()
    records: list[dict] = []
    gaze_path: deque[tuple[float, float]] = deque(maxlen=120)
    screen_history: deque[tuple[float, float]] = deque(maxlen=5)
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            estimate, landmarks = estimator.estimate(frame)
            timestamp_ms = (time.perf_counter() - start) * 1000.0

            if estimate is not None:
                calibrated_x, calibrated_y = calibration.apply(estimate.yaw_deg, estimate.pitch_deg)
                screen_history.append((calibrated_x, calibrated_y))
                calibrated_x = float(np.mean([p[0] for p in screen_history]))
                calibrated_y = float(np.mean([p[1] for p in screen_history]))
                calibrated_x = float(np.clip(calibrated_x, 0, screen_width - 1))
                calibrated_y = float(np.clip(calibrated_y, 0, screen_height - 1))
                aoi_id = find_aoi(calibrated_x, calibrated_y, aois)
                gaze_path.append((calibrated_x, calibrated_y))
                records.append(
                    {
                        "timestamp_ms": timestamp_ms,
                        "gaze_x": calibrated_x,
                        "gaze_y": calibrated_y,
                        "yaw_deg": estimate.yaw_deg,
                        "pitch_deg": estimate.pitch_deg,
                        "aoi_id": aoi_id,
                        "confidence": estimate.confidence,
                    }
                )
                _draw_status(frame, calibrated_x, calibrated_y, aoi_id, len(records), estimate.confidence)
                attention_canvas = _build_attention_canvas(
                    aois, screen_width, screen_height, calibrated_x, calibrated_y, aoi_id, list(gaze_path)
                )
                cv2.imshow("Attention Plane", attention_canvas)
            else:
                _draw_no_face_status(frame)

            _draw_face_landmarks(frame, landmarks)
            cv2.imshow("Gaze Tracking - press q to stop", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    gaze_log = pd.DataFrame(records, columns=_gaze_log_columns())
    save_analysis_outputs(gaze_log, aois, output_dir, screen_width, screen_height, min_fixation_ms)


def save_analysis_outputs(
    gaze_log: pd.DataFrame,
    aois: list[AOI],
    output_dir: Path,
    width: int,
    height: int,
    min_fixation_ms: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gaze_log_path = output_dir / "gaze_log.csv"
    if gaze_log.empty:
        gaze_log = pd.DataFrame(columns=_gaze_log_columns())
    gaze_log.to_csv(gaze_log_path, index=False)

    result = analyze_gaze_log(gaze_log, min_duration_ms=min_fixation_ms)
    result.fixation_table.to_csv(output_dir / "fixations.csv", index=False)
    result.aoi_summary.to_csv(output_dir / "aoi_summary.csv", index=False)
    result.transition_matrix.to_csv(output_dir / "transition_matrix.csv")
    save_visualizations(gaze_log, result.transition_matrix, aois, output_dir, width, height)


def _gaze_log_columns() -> list[str]:
    return [
        "timestamp_ms",
        "gaze_x",
        "gaze_y",
        "yaw_deg",
        "pitch_deg",
        "aoi_id",
        "confidence",
    ]


def _draw_status(
    frame,
    gaze_x: float,
    gaze_y: float,
    aoi_id: str,
    count: int,
    confidence: float,
) -> None:
    text = f"AOI: {aoi_id} | gaze=({gaze_x:.0f},{gaze_y:.0f}) | conf={confidence:.2f} | samples={count}"
    cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 230, 40), 2)


def _draw_no_face_status(frame) -> None:
    cv2.putText(
        frame,
        "No face detected. Face the camera and keep good lighting.",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (40, 180, 255),
        2,
    )


def _draw_face_landmarks(frame, landmarks: list[tuple[int, int]]) -> None:
    for point in landmarks:
        cv2.circle(frame, point, 3, (0, 255, 255), -1)


def _build_attention_canvas(
    aois: list[AOI],
    width: int,
    height: int,
    gaze_x: float,
    gaze_y: float,
    aoi_id: str,
    gaze_path: list[tuple[float, float]],
) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    for aoi in aois:
        color = (80, 80, 80)
        thickness = 2
        if aoi.id == aoi_id:
            color = (80, 180, 80)
            thickness = 4

        cv2.rectangle(canvas, (aoi.x1, aoi.y1), (aoi.x2, aoi.y2), color, thickness)
        cv2.putText(
            canvas,
            aoi.id,
            (aoi.x1 + 18, aoi.y1 + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color,
            2,
        )

    path_points = [
        (int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))) for x, y in gaze_path
    ]
    for start, end in zip(path_points, path_points[1:]):
        cv2.line(canvas, start, end, (180, 120, 40), 3)

    gx = int(np.clip(gaze_x, 0, width - 1))
    gy = int(np.clip(gaze_y, 0, height - 1))
    cv2.circle(canvas, (gx, gy), 14, (30, 30, 230), -1)
    cv2.circle(canvas, (gx, gy), 24, (30, 30, 230), 2)
    cv2.putText(
        canvas,
        f"Current AOI: {aoi_id}",
        (20, height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
    )
    return cv2.resize(canvas, (960, 540), interpolation=cv2.INTER_AREA)
