from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from gaze_tracking.aoi import AOI
from gaze_tracking.calibration import load_calibration


@dataclass(frozen=True)
class DigitTarget:
    digit: str
    aoi: AOI


def run_digit_selection(
    output_dir: Path,
    screen_width: int = 1280,
    screen_height: int = 720,
    camera_index: int = 0,
    calibration_path: Path | None = None,
    model_path: Path | None = None,
    device: str = "auto",
    digits: str = "123456789",
    dwell_ms: float = 2000.0,
    digit_font_scale: float = 4.0,
    target_padding: int = 72,
    outside_margin: float = 160.0,
    random_seed: int | None = None,
    max_frames: int | None = None,
) -> None:
    from gaze_tracking.gaze_estimator import GazeEstimator

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(random_seed)
    digit_list = _normalize_digits(digits)
    calibration = load_calibration(calibration_path, screen_width, screen_height)
    estimator = GazeEstimator(model_path=model_path, device=device)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {camera_index}")

    targets = _build_digit_targets(
        digit_list,
        screen_width,
        screen_height,
        digit_font_scale,
        target_padding,
        rng,
    )
    start = time.perf_counter()
    records: list[dict] = []
    selections: list[dict] = []
    screen_history: deque[tuple[float, float]] = deque(maxlen=5)
    gaze_path: deque[tuple[float, float]] = deque(maxlen=90)
    current_digit: str | None = None
    hit_started_ms: float | None = None
    selected_this_layout = False
    last_selection: dict | None = None
    reshuffle_at_ms: float | None = None
    frame_count = 0

    try:
        window_name = "Digit Gaze Selection - press q to stop"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            estimate, _ = estimator.estimate(frame)
            timestamp_ms = (time.perf_counter() - start) * 1000.0

            if reshuffle_at_ms is not None and timestamp_ms >= reshuffle_at_ms:
                targets = _build_digit_targets(
                    digit_list,
                    screen_width,
                    screen_height,
                    digit_font_scale,
                    target_padding,
                    rng,
                )
                current_digit = None
                hit_started_ms = None
                selected_this_layout = False
                reshuffle_at_ms = None

            gaze_x: float | None = None
            gaze_y: float | None = None
            visible_gaze: tuple[float, float] | None = None
            status = "NO_GAZE"
            target_digit: str | None = None
            selected_digit: str | None = None
            dwell_elapsed_ms = 0.0

            if estimate is not None:
                raw_x, raw_y = calibration.apply(estimate.yaw_deg, estimate.pitch_deg, clamp=False)
                gaze_x, gaze_y = raw_x, raw_y

                if _is_outside_screen(raw_x, raw_y, screen_width, screen_height, outside_margin):
                    screen_history.clear()
                    status = "OUT_SCREEN"
                else:
                    screen_history.append(
                        (
                            float(np.clip(raw_x, 0, screen_width - 1)),
                            float(np.clip(raw_y, 0, screen_height - 1)),
                        )
                    )
                    gaze_x = float(np.mean([point[0] for point in screen_history]))
                    gaze_y = float(np.mean([point[1] for point in screen_history]))
                    visible_gaze = (gaze_x, gaze_y)
                    gaze_path.append(visible_gaze)

                    target = _find_digit_target(gaze_x, gaze_y, targets)
                    if target is None:
                        status = "BLANK"
                    else:
                        status = "DIGIT"
                        target_digit = target.digit

                if status != "DIGIT":
                    current_digit = None
                    hit_started_ms = None
                elif target_digit != current_digit:
                    current_digit = target_digit
                    hit_started_ms = timestamp_ms
                elif hit_started_ms is not None:
                    # 连续看同一个数字达到设定时间后才算选中。
                    dwell_elapsed_ms = timestamp_ms - hit_started_ms

                if (
                    target_digit is not None
                    and hit_started_ms is not None
                    and dwell_elapsed_ms >= dwell_ms
                    and not selected_this_layout
                ):
                    selected_digit = target_digit
                    target = next(item for item in targets if item.digit == target_digit)
                    selection = {
                        "timestamp_ms": timestamp_ms,
                        "digit": selected_digit,
                        "gaze_x": gaze_x,
                        "gaze_y": gaze_y,
                        "dwell_ms": dwell_elapsed_ms,
                        "target_x1": target.aoi.x1,
                        "target_y1": target.aoi.y1,
                        "target_x2": target.aoi.x2,
                        "target_y2": target.aoi.y2,
                    }
                    selections.append(selection)
                    last_selection = selection
                    selected_this_layout = True
                    reshuffle_at_ms = timestamp_ms + 900.0
            else:
                screen_history.clear()
                current_digit = None
                hit_started_ms = None

            records.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "gaze_x": gaze_x,
                    "gaze_y": gaze_y,
                    "yaw_deg": None if estimate is None else estimate.yaw_deg,
                    "pitch_deg": None if estimate is None else estimate.pitch_deg,
                    "status": status,
                    "target_digit": target_digit,
                    "dwell_elapsed_ms": dwell_elapsed_ms,
                    "confidence": 0.0 if estimate is None else estimate.confidence,
                }
            )

            canvas = _build_digit_canvas(
                screen_width,
                screen_height,
                targets,
                visible_gaze,
                list(gaze_path),
                status,
                target_digit,
                dwell_elapsed_ms,
                dwell_ms,
                digit_font_scale,
                last_selection,
            )
            cv2.imshow(window_name, canvas)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    pd.DataFrame(records, columns=_gaze_log_columns()).to_csv(
        output_dir / "digit_gaze_log.csv",
        index=False,
    )
    pd.DataFrame(selections, columns=_selection_log_columns()).to_csv(
        output_dir / "digit_selection_log.csv",
        index=False,
    )


def _gaze_log_columns() -> list[str]:
    return [
        "timestamp_ms",
        "gaze_x",
        "gaze_y",
        "yaw_deg",
        "pitch_deg",
        "status",
        "target_digit",
        "dwell_elapsed_ms",
        "confidence",
    ]


def _selection_log_columns() -> list[str]:
    return [
        "timestamp_ms",
        "digit",
        "gaze_x",
        "gaze_y",
        "dwell_ms",
        "target_x1",
        "target_y1",
        "target_x2",
        "target_y2",
    ]


def _normalize_digits(digits: str) -> list[str]:
    result = []
    for char in digits:
        if char.isdigit() and char not in result:
            result.append(char)
    if not result:
        raise ValueError("--digits must contain at least one digit")
    return result


def _build_digit_targets(
    digits: list[str],
    width: int,
    height: int,
    font_scale: float,
    padding: int,
    rng: random.Random,
) -> list[DigitTarget]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(3, int(round(font_scale * 2)))
    text_sizes = [cv2.getTextSize(digit, font, font_scale, thickness)[0] for digit in digits]
    target_width = max(size[0] for size in text_sizes) + padding * 2
    target_height = max(size[1] for size in text_sizes) + padding * 2
    positions = _grid_positions(len(digits), target_width, target_height, width, height, rng)

    targets = []
    for digit, (x1, y1) in zip(digits, positions):
        aoi = AOI(
            id=f"DIGIT:{digit}",
            x1=x1,
            y1=y1,
            x2=x1 + target_width,
            y2=y1 + target_height,
        )
        targets.append(DigitTarget(digit=digit, aoi=aoi))
    return targets


def _grid_positions(
    count: int,
    target_width: int,
    target_height: int,
    width: int,
    height: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    margin = 32
    available_width = width - margin * 2
    available_height = height - margin * 2

    for columns in range(1, count + 1):
        rows = int(np.ceil(count / columns))
        if columns * target_width > available_width or rows * target_height > available_height:
            continue

        xs = np.linspace(margin, width - margin - target_width, columns)
        ys = np.linspace(margin, height - margin - target_height, rows)
        positions = [(int(round(x)), int(round(y))) for y in ys for x in xs]
        rng.shuffle(positions)
        return positions[:count]

    raise ValueError("Digit targets do not fit on screen. Reduce font scale or target padding.")


def _find_digit_target(x: float, y: float, targets: list[DigitTarget]) -> DigitTarget | None:
    for target in targets:
        if target.aoi.contains(x, y):
            return target
    return None


def _is_outside_screen(x: float, y: float, width: int, height: int, margin: float) -> bool:
    return x < -margin or y < -margin or x >= width + margin or y >= height + margin


def _build_digit_canvas(
    width: int,
    height: int,
    targets: list[DigitTarget],
    gaze: tuple[float, float] | None,
    gaze_path: list[tuple[float, float]],
    status: str,
    target_digit: str | None,
    dwell_elapsed_ms: float,
    dwell_ms: float,
    digit_font_scale: float,
    last_selection: dict | None,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for target in targets:
        is_hit = target.digit == target_digit
        box_color = (80, 180, 80) if is_hit else (190, 190, 190)
        thickness = 4 if is_hit else 2
        cv2.rectangle(
            canvas,
            (target.aoi.x1, target.aoi.y1),
            (target.aoi.x2, target.aoi.y2),
            box_color,
            thickness,
        )

        digit_thickness = max(3, int(round(digit_font_scale * 2)))
        text_size, _ = cv2.getTextSize(target.digit, font, digit_font_scale, digit_thickness)
        center_x = (target.aoi.x1 + target.aoi.x2) // 2
        center_y = (target.aoi.y1 + target.aoi.y2) // 2
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2
        cv2.putText(
            canvas,
            target.digit,
            (text_x, text_y),
            font,
            digit_font_scale,
            (20, 20, 20),
            digit_thickness,
            cv2.LINE_AA,
        )

    if len(gaze_path) > 1:
        points = [(int(x), int(y)) for x, y in gaze_path]
        for start, end in zip(points, points[1:]):
            cv2.line(canvas, start, end, (180, 120, 40), 2)

    if gaze is not None:
        gx = int(np.clip(gaze[0], 0, width - 1))
        gy = int(np.clip(gaze[1], 0, height - 1))
        cv2.circle(canvas, (gx, gy), 12, (30, 30, 230), -1)
        cv2.circle(canvas, (gx, gy), 22, (30, 30, 230), 2)

    progress = min(1.0, dwell_elapsed_ms / max(1.0, dwell_ms)) if status == "DIGIT" else 0.0
    cv2.rectangle(canvas, (32, 32), (width - 32, 58), (180, 180, 180), 2)
    cv2.rectangle(canvas, (32, 32), (32 + int((width - 64) * progress), 58), (80, 180, 80), -1)

    status_text = _status_text(status, target_digit, dwell_elapsed_ms, dwell_ms)
    cv2.putText(canvas, status_text, (32, 100), font, 0.85, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "q: stop", (32, height - 34), font, 0.72, (80, 80, 80), 2, cv2.LINE_AA)

    if last_selection is not None:
        text = f"Selected: {last_selection['digit']}"
        text_size, _ = cv2.getTextSize(text, font, 1.3, 3)
        cv2.putText(
            canvas,
            text,
            (width - text_size[0] - 34, height - 34),
            font,
            1.3,
            (40, 130, 40),
            3,
            cv2.LINE_AA,
        )
    return canvas


def _status_text(
    status: str,
    target_digit: str | None,
    dwell_elapsed_ms: float,
    dwell_ms: float,
) -> str:
    if status == "DIGIT":
        return f"Looking at digit {target_digit}: {dwell_elapsed_ms / 1000.0:.1f}/{dwell_ms / 1000.0:.1f}s"
    if status == "BLANK":
        return "Looking at blank area: no input"
    if status == "OUT_SCREEN":
        return "Looking outside screen: OUT"
    return "No reliable gaze detected"
