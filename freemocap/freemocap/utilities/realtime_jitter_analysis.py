import csv
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

JITTER_INTERVALS_SECONDS = (
    (5.0, 9.0),
    (11.0, 15.0),
    (20.0, 24.0),
    (54.0, 58.0),
    (67.0, 71.0),
    (79.0, 83.0),
)


def _recording_prefix(preview_output_path: Path) -> str:
    suffix = "_realtime_preview"
    stem = preview_output_path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def _total_jitter(pose: np.ndarray) -> tuple[float, int]:
    """Sum 3-D frame-to-frame movement across every valid pose landmark."""
    if pose.shape[0] < 2:
        return np.nan, 0
    valid_samples = np.isfinite(pose).all(axis=2)
    valid_transitions = valid_samples[1:] & valid_samples[:-1]
    distances = np.linalg.norm(np.diff(pose, axis=0), axis=2)
    valid_distances = distances[valid_transitions]
    if valid_distances.size == 0:
        return np.nan, 0
    return float(np.sum(valid_distances)), int(valid_distances.size)


def save_realtime_pose_data_and_jitter_report(
    preview_output_path: Path,
    timestamps: np.ndarray,
    raw_pose: np.ndarray,
    filtered_pose: np.ndarray,
) -> None:
    """Save RT pose samples and jitter for the requested stationary intervals."""
    preview_output_path = Path(preview_output_path)
    output_folder = preview_output_path.parent
    prefix = _recording_prefix(preview_output_path)

    timestamps = np.asarray(timestamps, dtype=float)
    raw_pose = np.asarray(raw_pose, dtype=float)
    filtered_pose = np.asarray(filtered_pose, dtype=float)
    expected_shape = (timestamps.size, 33, 3)
    if raw_pose.shape != expected_shape or filtered_pose.shape != expected_shape:
        raise ValueError(
            "Pose arrays must both have shape "
            f"{expected_shape}; received {raw_pose.shape} and {filtered_pose.shape}"
        )

    timestamp_path = output_folder / f"{prefix}_pose_timestamps.npy"
    raw_pose_path = output_folder / f"{prefix}_raw_mediapipe_pose.npy"
    filtered_pose_path = output_folder / f"{prefix}_kalman_pose.npy"
    report_path = output_folder / f"{prefix}_jitter_report.csv"
    np.save(timestamp_path, timestamps)
    np.save(raw_pose_path, raw_pose)
    np.save(filtered_pose_path, filtered_pose)

    fieldnames = (
        "interval",
        "start_seconds",
        "end_seconds",
        "sample_count",
        "raw_mediapipe_total_jitter",
        "raw_mediapipe_valid_landmark_transitions",
        "kalman_total_jitter",
        "kalman_valid_landmark_transitions",
        "coordinate_units",
    )
    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()
        for start_seconds, end_seconds in JITTER_INTERVALS_SECONDS:
            interval_mask = (
                (timestamps >= start_seconds) & (timestamps <= end_seconds)
            )
            interval_label = f"{start_seconds:.0f}-{end_seconds:.0f}s"
            raw_interval = raw_pose[interval_mask]
            filtered_interval = filtered_pose[interval_mask]
            raw_total, raw_transition_count = _total_jitter(raw_interval)
            filtered_total, filtered_transition_count = _total_jitter(
                filtered_interval
            )
            writer.writerow(
                {
                    "interval": interval_label,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "sample_count": len(raw_interval),
                    "raw_mediapipe_total_jitter": raw_total,
                    "raw_mediapipe_valid_landmark_transitions": raw_transition_count,
                    "kalman_total_jitter": filtered_total,
                    "kalman_valid_landmark_transitions": filtered_transition_count,
                    "coordinate_units": "MediaPipe normalized XYZ",
                }
            )

    logger.info(
        "Saved real-time pose arrays and jitter report to: %s",
        output_folder,
    )
