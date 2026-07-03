from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gaze_tracking.aoi import build_grid_aois


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaze tracking and visual attention analysis")
    parser.add_argument("--mode", choices=["webcam", "analyze", "calibrate", "digits"], default="webcam")
    parser.add_argument("--input", type=Path, help="Input gaze_log.csv for analyze mode")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--min-fixation-ms", type=float, default=120.0)
    parser.add_argument("--calibration", type=Path, default=Path("outputs/calibration.json"))
    parser.add_argument("--samples-per-point", type=int, default=30)
    parser.add_argument("--model-path", type=Path, default=Path("models/L2CSNet_gaze360.pkl"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--digits", default="123456789", help="Digits shown in digits mode")
    parser.add_argument("--dwell-ms", type=float, default=2000.0, help="Dwell time needed to select a digit")
    parser.add_argument("--digit-font-scale", type=float, default=4.0)
    parser.add_argument("--target-padding", type=int, default=72)
    parser.add_argument("--outside-margin", type=float, default=160.0)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Automatically stop webcam/digits mode after this many frames; 0 means run until q",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output

    if args.mode == "calibrate":
        from gaze_tracking.calibration_app import run_interactive_calibration

        run_interactive_calibration(
            output_dir=output_dir,
            calibration_path=args.calibration,
            screen_width=args.width,
            screen_height=args.height,
            camera_index=args.camera,
            samples_per_point=args.samples_per_point,
            model_path=args.model_path,
            device=args.device,
        )
        print(f"Saved calibration to {args.calibration.resolve()}")
        print(f"Saved calibration samples to {(output_dir / 'calibration_samples.csv').resolve()}")
        return

    if args.mode == "webcam":
        from gaze_tracking.webcam_app import run_webcam

        run_webcam(
            output_dir=output_dir,
            screen_width=args.width,
            screen_height=args.height,
            camera_index=args.camera,
            rows=args.rows,
            cols=args.cols,
            min_fixation_ms=args.min_fixation_ms,
            calibration_path=args.calibration,
            model_path=args.model_path,
            device=args.device,
            max_frames=args.max_frames if args.max_frames > 0 else None,
        )
        print(f"Saved results to {output_dir.resolve()}")
        return

    if args.mode == "digits":
        from gaze_tracking.digit_selection_app import run_digit_selection

        run_digit_selection(
            output_dir=output_dir,
            screen_width=args.width,
            screen_height=args.height,
            camera_index=args.camera,
            calibration_path=args.calibration,
            model_path=args.model_path,
            device=args.device,
            digits=args.digits,
            dwell_ms=args.dwell_ms,
            digit_font_scale=args.digit_font_scale,
            target_padding=args.target_padding,
            outside_margin=args.outside_margin,
            random_seed=args.random_seed,
            max_frames=args.max_frames if args.max_frames > 0 else None,
        )
        print(f"Saved digit selection results to {output_dir.resolve()}")
        return

    if args.input is None:
        raise SystemExit("--input is required in analyze mode")

    gaze_log = pd.read_csv(args.input)
    aois = build_grid_aois(args.width, args.height, rows=args.rows, cols=args.cols)
    from gaze_tracking.webcam_app import save_analysis_outputs

    save_analysis_outputs(gaze_log, aois, output_dir, args.width, args.height, args.min_fixation_ms)
    print(f"Saved analysis results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
