from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gaze_tracking.aoi import build_grid_aois, find_aoi
from gaze_tracking.webcam_app import save_analysis_outputs


def main() -> None:
    width = 1280
    height = 720
    output_dir = Path("outputs/sample")
    aois = build_grid_aois(width, height, rows=3, cols=3)

    # A small synthetic path: center -> right -> bottom-left -> center.
    path = [
        (640, 360, 900),
        (1000, 360, 700),
        (250, 600, 800),
        (640, 360, 600),
    ]

    records = []
    timestamp_ms = 0.0
    rng = np.random.default_rng(2026)

    for x, y, duration_ms in path:
        sample_count = max(2, int(duration_ms / 33))
        for _ in range(sample_count):
            gx = float(np.clip(x + rng.normal(0, 20), 0, width - 1))
            gy = float(np.clip(y + rng.normal(0, 16), 0, height - 1))
            records.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "gaze_x": gx,
                    "gaze_y": gy,
                    "yaw_deg": (gx / width - 0.5) * 40.0,
                    "pitch_deg": (gy / height - 0.5) * 30.0,
                    "aoi_id": find_aoi(gx, gy, aois),
                    "confidence": 0.95,
                }
            )
            timestamp_ms += 33.0

    gaze_log = pd.DataFrame(records)
    save_analysis_outputs(gaze_log, aois, output_dir, width, height, min_fixation_ms=120.0)
    print(f"Sample outputs saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
