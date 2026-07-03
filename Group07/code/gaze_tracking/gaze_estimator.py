from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GazeEstimate:
    yaw_deg: float
    pitch_deg: float
    confidence: float


class GazeEstimator:
    """Use L2CS-Net to estimate horizontal and vertical gaze angles."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        device: str = "auto",
        smooth_window: int = 6,
    ):
        self.model_path = Path(model_path) if model_path is not None else Path("models/L2CSNet_gaze360.pkl")
        self.device_name = device
        self.history: deque[tuple[float, float]] = deque(maxlen=smooth_window)
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                "L2CS-Net weights were not found. Place L2CSNet_gaze360.pkl at "
                f"{self.model_path.resolve()}."
            )

        try:
            import torch
            from l2cs import Pipeline
        except ImportError as exc:
            raise ImportError(
                "L2CS-Net is not installed. Run: pip install -r code\\requirements.txt"
            ) from exc

        if self.device_name == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif self.device_name == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but PyTorch cannot access CUDA.")
            device = torch.device("cuda:0")
        elif self.device_name == "cpu":
            device = torch.device("cpu")
        else:
            raise ValueError(f"unsupported device: {self.device_name}")

        return Pipeline(weights=self.model_path, arch="ResNet50", device=device)

    def estimate(self, frame_bgr: np.ndarray) -> tuple[GazeEstimate | None, list[tuple[int, int]]]:
        try:
            results = self.pipeline.step(frame_bgr)
        except ValueError:
            return None, []

        yaw_values = np.asarray(getattr(results, "yaw", []), dtype=float).reshape(-1)
        pitch_values = np.asarray(getattr(results, "pitch", []), dtype=float).reshape(-1)
        if yaw_values.size == 0 or pitch_values.size == 0:
            return None, []

        scores = np.asarray(getattr(results, "scores", []), dtype=float).reshape(-1)
        best_index = int(np.argmax(scores)) if scores.size else 0
        best_index = min(best_index, yaw_values.size - 1, pitch_values.size - 1)

        # L2CS-Net 输出弧度，这里转成更容易记录和校准的角度。
        yaw_deg = float(np.degrees(yaw_values[best_index]))
        pitch_deg = float(np.degrees(pitch_values[best_index]))
        confidence = float(scores[best_index]) if best_index < scores.size else 1.0

        self.history.append((yaw_deg, pitch_deg))
        smooth_yaw = float(np.mean([point[0] for point in self.history]))
        smooth_pitch = float(np.mean([point[1] for point in self.history]))

        landmarks = _landmarks_as_points(getattr(results, "landmarks", None), best_index)
        return GazeEstimate(smooth_yaw, smooth_pitch, confidence), landmarks


def _landmarks_as_points(landmarks: object, index: int) -> list[tuple[int, int]]:
    if landmarks is None:
        return []

    array = np.asarray(landmarks)
    if array.size == 0:
        return []

    try:
        face_landmarks = array[index]
    except IndexError:
        return []

    points = np.asarray(face_landmarks).reshape(-1, 2)
    return [(int(x), int(y)) for x, y in points if np.isfinite(x) and np.isfinite(y)]
