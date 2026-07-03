from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MODEL_NAME = "l2cs"
INPUT_FEATURES = ("yaw_deg", "pitch_deg")


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    x_coeffs: tuple[float, ...]
    y_coeffs: tuple[float, ...]
    points: tuple[dict, ...] = ()

    def apply(self, yaw_deg: float, pitch_deg: float, clamp: bool = True) -> tuple[float, float]:
        features = _poly2_features(yaw_deg, pitch_deg)
        screen_x = float(np.dot(np.array(self.x_coeffs), features))
        screen_y = float(np.dot(np.array(self.y_coeffs), features))

        if clamp:
            screen_x = min(max(screen_x, 0.0), float(self.width - 1))
            screen_y = min(max(screen_y, 0.0), float(self.height - 1))
        return screen_x, screen_y

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": MODEL_NAME,
            "method": "poly2",
            "input_features": list(INPUT_FEATURES),
            "width": self.width,
            "height": self.height,
            "x_coeffs": list(self.x_coeffs),
            "y_coeffs": list(self.y_coeffs),
            "points": list(self.points),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration(path: Path | None, width: int, height: int) -> Calibration:
    target = path or Path("outputs/calibration.json")
    if not target.exists():
        raise RuntimeError(
            f"Calibration file not found: {target}\n"
            "Run: python code\\main.py --mode calibrate"
        )

    data = json.loads(target.read_text(encoding="utf-8"))

    # 兼容项目之前生成的 L2CS 校准文件，但不接受旧虹膜方案的系数。
    model = str(data.get("model", data.get("backend", ""))).lower()
    feature_value = data.get("input_features", data.get("feature_names", []))
    input_features = (
        tuple(str(item) for item in feature_value)
        if isinstance(feature_value, (list, tuple))
        else ()
    )
    x_coeffs = _read_coeffs(data.get("x_coeffs"))
    y_coeffs = _read_coeffs(data.get("y_coeffs"))

    if (
        model != MODEL_NAME
        or str(data.get("method", "")) != "poly2"
        or input_features != INPUT_FEATURES
        or len(x_coeffs) != 6
        or len(y_coeffs) != 6
    ):
        raise ValueError(
            f"Calibration file {target} is not a valid L2CS 16-point calibration.\n"
            "Run: python code\\main.py --mode calibrate"
        )

    return Calibration(
        width=int(data.get("width", width)),
        height=int(data.get("height", height)),
        x_coeffs=x_coeffs,
        y_coeffs=y_coeffs,
        points=tuple(data.get("points", [])),
    )


def fit_poly2_calibration(samples: list[dict], width: int, height: int) -> Calibration:
    if len(samples) < 6:
        raise ValueError("At least 6 calibration samples are required")

    raw_angles = np.array(
        [[float(item["yaw_deg"]), float(item["pitch_deg"])] for item in samples],
        dtype=float,
    )
    target_x = np.array([float(item["target_x"]) for item in samples], dtype=float)
    target_y = np.array([float(item["target_y"]) for item in samples], dtype=float)
    design = np.vstack([_poly2_features(yaw, pitch) for yaw, pitch in raw_angles])

    # 分别拟合屏幕横坐标和纵坐标。
    x_coeffs, *_ = np.linalg.lstsq(design, target_x, rcond=None)
    y_coeffs, *_ = np.linalg.lstsq(design, target_y, rcond=None)

    return Calibration(
        width=width,
        height=height,
        x_coeffs=tuple(float(value) for value in x_coeffs),
        y_coeffs=tuple(float(value) for value in y_coeffs),
        points=tuple(samples),
    )


def _poly2_features(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    return np.array(
        [1.0, yaw_deg, pitch_deg, yaw_deg * pitch_deg, yaw_deg * yaw_deg, pitch_deg * pitch_deg],
        dtype=float,
    )


def _read_coeffs(value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(float(item) for item in value)
