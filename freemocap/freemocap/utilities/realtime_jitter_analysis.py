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


def save_filter_comparison_data_and_jitter_report(
    preview_output_path: Path,
    timestamps: np.ndarray,
    raw_pose: np.ndarray,
    one_euro_pose: np.ndarray,
    kalman_pose: np.ndarray,
    kalman_one_euro_pose: np.ndarray,
) -> None:
    """Save matching independent and combined filter comparison streams."""
    preview_output_path = Path(preview_output_path)
    output_folder = preview_output_path.parent
    prefix = _recording_prefix(preview_output_path)

    timestamps = np.asarray(timestamps, dtype=float)
    raw_pose = np.asarray(raw_pose, dtype=float)
    one_euro_pose = np.asarray(one_euro_pose, dtype=float)
    kalman_pose = np.asarray(kalman_pose, dtype=float)
    kalman_one_euro_pose = np.asarray(kalman_one_euro_pose, dtype=float)
    expected_shape = (timestamps.size, 33, 3)
    poses = {
        "raw_mediapipe": raw_pose,
        "one_euro": one_euro_pose,
        "kalman": kalman_pose,
        "kalman_one_euro": kalman_one_euro_pose,
    }
    invalid_shapes = {
        name: pose.shape for name, pose in poses.items() if pose.shape != expected_shape
    }
    if invalid_shapes:
        raise ValueError(
            f"Pose arrays must have shape {expected_shape}; received {invalid_shapes}"
        )

    np.save(output_folder / f"{prefix}_pose_timestamps.npy", timestamps)
    for pose_name, pose in poses.items():
        np.save(output_folder / f"{prefix}_{pose_name}_pose.npy", pose)

    report_path = output_folder / f"{prefix}_filter_comparison_jitter_report.csv"
    fieldnames = (
        "interval",
        "start_seconds",
        "end_seconds",
        "sample_count",
        "raw_mediapipe_total_jitter",
        "raw_mediapipe_valid_landmark_transitions",
        "one_euro_total_jitter",
        "one_euro_valid_landmark_transitions",
        "kalman_total_jitter",
        "kalman_valid_landmark_transitions",
        "kalman_one_euro_total_jitter",
        "kalman_one_euro_valid_landmark_transitions",
        "coordinate_units",
    )
    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()
        for start_seconds, end_seconds in JITTER_INTERVALS_SECONDS:
            interval_mask = (
                (timestamps >= start_seconds) & (timestamps <= end_seconds)
            )
            interval_poses = {
                name: pose[interval_mask] for name, pose in poses.items()
            }
            metrics = {
                name: _total_jitter(pose)
                for name, pose in interval_poses.items()
            }
            writer.writerow(
                {
                    "interval": f"{start_seconds:.0f}-{end_seconds:.0f}s",
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "sample_count": int(interval_mask.sum()),
                    "raw_mediapipe_total_jitter": metrics["raw_mediapipe"][0],
                    "raw_mediapipe_valid_landmark_transitions": metrics["raw_mediapipe"][1],
                    "one_euro_total_jitter": metrics["one_euro"][0],
                    "one_euro_valid_landmark_transitions": metrics["one_euro"][1],
                    "kalman_total_jitter": metrics["kalman"][0],
                    "kalman_valid_landmark_transitions": metrics["kalman"][1],
                    "kalman_one_euro_total_jitter": metrics["kalman_one_euro"][0],
                    "kalman_one_euro_valid_landmark_transitions": metrics["kalman_one_euro"][1],
                    "coordinate_units": "MediaPipe normalized XYZ",
                }
            )

    logger.info("Saved filter-comparison jitter report to: %s", report_path)
