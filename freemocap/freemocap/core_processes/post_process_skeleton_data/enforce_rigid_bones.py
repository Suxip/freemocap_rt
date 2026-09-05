from copy import deepcopy
from typing import Dict, List, Union

import numpy as np

from freemocap.data_layer.skeleton_models.segments import Segment
from freemocap.data_layer.skeleton_models.skeleton import Skeleton


def calculate_bone_lengths_and_statistics(
        marker_data: Dict[str, np.ndarray], segment_connections: Dict[str, Segment]
) -> Dict[str, Dict[str, Union[np.ndarray, float]]]:
    """
    Calculates bone lengths and a forward-only running length estimate.

    Parameters:
    - marker_data: A dictionary containing marker trajectories with marker names as keys and
      3D positions as values (numpy arrays).
    - segment_connections: A dictionary defining the segments (bones) with segment names as keys
      and dictionaries with 'proximal' and 'distal' markers as values.

    Returns:
    - A dictionary containing observed lengths and causal target-length estimates.
    """
    bone_statistics = {}

    for segment_name, segment in segment_connections.items():
        proximal_positions = marker_data[segment.proximal]
        distal_positions = marker_data[segment.distal]

        lengths = np.linalg.norm(distal_positions - proximal_positions, axis=1)
        # Estimate the target length from the current and previous frames only. An
        # expanding mean is deliberately used instead of the whole-recording
        # median so an early output can never change when later frames arrive.
        valid = np.isfinite(lengths)
        cumulative_sum = np.cumsum(np.where(valid, lengths, 0.0))
        cumulative_count = np.cumsum(valid)
        causal_length_estimate = np.divide(
            cumulative_sum,
            cumulative_count,
            out=np.full_like(lengths, np.nan, dtype=float),
            where=cumulative_count > 0,
        )

        bone_statistics[segment_name] = {
            "lengths": lengths,
            "causal_length_estimate": causal_length_estimate,
        }

    return bone_statistics


def enforce_rigid_bones(
        marker_data: Dict[str, np.ndarray],
        segment_connections: Dict[str, Segment],
        bone_lengths_and_statistics: Dict[str, Dict[str, Union[np.ndarray, float]]],
        joint_hierarchy: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    """
    Enforces rigid bones using the target estimate available at each frame.

    Parameters:
    - marker_data: The original marker positions.
    - segment_connections: Information about how segments (bones) are connected.
    - bone_lengths_and_statistics: The desired bone lengths and statistics for each segment.
    - joint_hierarchy: The hierarchy of joints, indicating parent-child relationships.

    Returns:
    - A dictionary of adjusted marker positions.
    """
    rigid_marker_data = deepcopy(marker_data)

    for segment_name, stats in bone_lengths_and_statistics.items():
        desired_lengths = stats["causal_length_estimate"]
        lengths = stats["lengths"]

        segment = segment_connections[segment_name]
        proximal_marker, distal_marker = segment.proximal, segment.distal

        for frame_index, current_length in enumerate(lengths):
            desired_length = desired_lengths[frame_index]
            if np.isfinite(current_length) and np.isfinite(desired_length) and current_length != desired_length:
                proximal_position = marker_data[proximal_marker][frame_index]
                distal_position = marker_data[distal_marker][frame_index]
                direction = distal_position - proximal_position
                try:
                    direction /= np.linalg.norm(direction)  # Normalize to unit vector
                except ZeroDivisionError:
                    direction /= 1e-5  # Set to a small value if the direction is zero
                adjustment = (desired_length - current_length) * direction

                rigid_marker_data[distal_marker][frame_index] += adjustment

                adjust_children(distal_marker, frame_index, adjustment, rigid_marker_data, joint_hierarchy)

    return rigid_marker_data


def adjust_children(
        parent_marker: str,
        frame_index: int,
        adjustment: np.ndarray,
        marker_data: Dict[str, np.ndarray],
        joint_hierarchy: Dict[str, List[str]],
):
    """
    Recursively adjusts the positions of child markers based on the adjustment of the parent marker.
    """
    if parent_marker in joint_hierarchy:
        for child_marker in joint_hierarchy[parent_marker]:
            marker_data[child_marker][frame_index] += adjustment
            adjust_children(child_marker, frame_index, adjustment, marker_data, joint_hierarchy)


def merge_rigid_marker_data(rigid_marker_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Merges the center of mass data from multiple segments into a single array.

    Parameters:
    - segment_com_data: A dictionary where each key is a segment name and the value is the center of mass data for that segment.

    Returns:
    - A numpy array containing the merged center of mass data.
    """
    # TODO: We could use this more broadly as a skeleton method

    rigid_marker_data_list = list(rigid_marker_data.values())

    return np.stack(rigid_marker_data_list, axis=1)


def enforce_rigid_bones_from_skeleton(skeleton: Skeleton) -> np.ndarray:
    """
    Calculates bone lengths and statistics from skeleton data and enforces rigid bones.

    Parameters:
    - skeleton: The Skeleton instance containing segment information, joint hierarchy, and marker data.

    Returns:
    - A numpy array of adjusted marker positions.
    """
    # TODO: Should this be a method of Skeleton?
    if not skeleton.segments:
        raise ValueError("Segments must be defined before rigid bones can be enforced.")

    if not skeleton.joint_hierarchy:
        raise ValueError("Joint hierarchy must be defined before rigid bones can be enforced.")

    bone_lengths_and_statistcs = calculate_bone_lengths_and_statistics(
        marker_data=skeleton.marker_data, segment_connections=skeleton.segments
    )

    rigid_marker_data = enforce_rigid_bones(
        marker_data=skeleton.marker_data,
        segment_connections=skeleton.segments,
        bone_lengths_and_statistics=bone_lengths_and_statistcs,
        joint_hierarchy=skeleton.joint_hierarchy,
    )

    return merge_rigid_marker_data(rigid_marker_data=rigid_marker_data)
