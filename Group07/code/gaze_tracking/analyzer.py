from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class AnalysisResult:
    fixation_table: pd.DataFrame
    aoi_summary: pd.DataFrame
    transition_matrix: pd.DataFrame


def build_fixations(
    gaze_log: pd.DataFrame,
    min_duration_ms: float = 120.0,
) -> pd.DataFrame:
    """Merge consecutive samples in the same AOI into fixation candidates."""
    required = {"timestamp_ms", "gaze_x", "gaze_y", "aoi_id"}
    missing = required - set(gaze_log.columns)
    if missing:
        raise ValueError(f"gaze log is missing columns: {sorted(missing)}")

    if gaze_log.empty:
        return pd.DataFrame(
            columns=[
                "fixation_id",
                "aoi_id",
                "start_ms",
                "end_ms",
                "duration_ms",
                "mean_x",
                "mean_y",
                "sample_count",
            ]
        )

    rows: list[dict] = []
    current_aoi = None
    start_idx = 0
    fixation_id = 1

    sorted_log = gaze_log.sort_values("timestamp_ms").reset_index(drop=True)

    for idx, sample in sorted_log.iterrows():
        aoi = sample["aoi_id"]
        if current_aoi is None:
            current_aoi = aoi
            start_idx = idx
            continue

        if aoi != current_aoi:
            fixation_id = _append_fixation(
                rows, sorted_log, start_idx, idx - 1, current_aoi, fixation_id, min_duration_ms
            )
            current_aoi = aoi
            start_idx = idx

    _append_fixation(
        rows,
        sorted_log,
        start_idx,
        len(sorted_log) - 1,
        current_aoi,
        fixation_id,
        min_duration_ms,
    )

    return pd.DataFrame(rows)


def analyze_gaze_log(gaze_log: pd.DataFrame, min_duration_ms: float = 120.0) -> AnalysisResult:
    fixations = build_fixations(gaze_log, min_duration_ms=min_duration_ms)
    aoi_summary = summarize_aois(fixations)
    transition_matrix = build_transition_matrix(fixations)
    return AnalysisResult(fixations, aoi_summary, transition_matrix)


def summarize_aois(fixations: pd.DataFrame) -> pd.DataFrame:
    if fixations.empty:
        return pd.DataFrame(
            columns=[
                "aoi_id",
                "fixation_count",
                "total_duration_ms",
                "first_fixation_duration_ms",
                "mean_fixation_duration_ms",
            ]
        )

    summary = (
        fixations.groupby("aoi_id")
        .agg(
            fixation_count=("fixation_id", "count"),
            total_duration_ms=("duration_ms", "sum"),
            first_fixation_duration_ms=("duration_ms", "first"),
            mean_fixation_duration_ms=("duration_ms", "mean"),
        )
        .reset_index()
        .sort_values("total_duration_ms", ascending=False)
    )
    return summary


def build_transition_matrix(fixations: pd.DataFrame) -> pd.DataFrame:
    if fixations.empty:
        return pd.DataFrame()

    sequence = fixations["aoi_id"].tolist()
    aois = sorted(set(sequence))
    matrix = pd.DataFrame(0, index=aois, columns=aois, dtype=int)

    for source, target in zip(sequence, sequence[1:]):
        if source != target:
            matrix.loc[source, target] += 1

    return matrix


def _append_fixation(
    rows: list[dict],
    gaze_log: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    aoi_id: str,
    fixation_id: int,
    min_duration_ms: float,
) -> int:
    segment = gaze_log.iloc[start_idx : end_idx + 1]
    start_ms = float(segment["timestamp_ms"].iloc[0])
    end_ms = float(segment["timestamp_ms"].iloc[-1])
    duration_ms = max(0.0, end_ms - start_ms)

    if duration_ms >= min_duration_ms and aoi_id != "OUT":
        rows.append(
            {
                "fixation_id": fixation_id,
                "aoi_id": aoi_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": duration_ms,
                "mean_x": float(segment["gaze_x"].mean()),
                "mean_y": float(segment["gaze_y"].mean()),
                "sample_count": int(len(segment)),
            }
        )
        return fixation_id + 1
    return fixation_id
