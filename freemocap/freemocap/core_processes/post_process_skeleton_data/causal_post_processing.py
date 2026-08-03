"""Stateful, forward-only post-processing for 3-D skeleton trajectories."""

import numpy as np


class OneEuroFilter:
    """Causal One Euro filter applied independently to every marker coordinate."""

    def __init__(self, shape: tuple[int, int], sampling_rate: float, min_cutoff: float,
                 beta: float, derivative_cutoff: float):
        self.dt = 1.0 / sampling_rate
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.previous_raw = np.full(shape, np.nan)
        self.previous_filtered = np.full(shape, np.nan)
        self.previous_derivative = np.zeros(shape, dtype=float)

    def _alpha(self, cutoff):
        time_constant = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + time_constant / self.dt)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=float)
        output = np.full_like(frame, np.nan)
        valid = np.isfinite(frame)
        first = valid & ~np.isfinite(self.previous_filtered)
        output[first] = frame[first]

        continuing = valid & ~first
        if np.any(continuing):
            raw_derivative = (frame[continuing] - self.previous_raw[continuing]) / self.dt
            derivative_alpha = self._alpha(self.derivative_cutoff)
            derivative = (
                derivative_alpha * raw_derivative
                + (1.0 - derivative_alpha) * self.previous_derivative[continuing]
            )
            cutoff = self.min_cutoff + self.beta * np.abs(derivative)
            alpha = self._alpha(cutoff)
            output[continuing] = (
                alpha * frame[continuing]
                + (1.0 - alpha) * self.previous_filtered[continuing]
            )
            self.previous_derivative[continuing] = derivative

        self.previous_raw[valid] = frame[valid]
        self.previous_filtered[valid] = output[valid]
        return output


def causally_post_process_skeleton(skeleton_data: np.ndarray, parameters) -> np.ndarray:
    """Process a complete array sequentially without inspecting future frames."""
    data = np.asarray(skeleton_data, dtype=float)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError("skeleton_data must have shape (frames, markers, 3)")
    if data.shape[0] == 0:
        return data.copy()

    shape = data.shape[1:]
    one_euro_parameters = parameters.one_euro_filter_parameters
    one_euro_filter = OneEuroFilter(
        shape=shape,
        sampling_rate=parameters.framerate,
        min_cutoff=one_euro_parameters.min_cutoff,
        beta=one_euro_parameters.beta,
        derivative_cutoff=one_euro_parameters.derivative_cutoff,
    )

    output = np.empty_like(data, dtype=float)
    for frame_number, frame in enumerate(data):
        processed_frame = frame.copy()
        if parameters.run_one_euro_filter:
            processed_frame = one_euro_filter.process_frame(processed_frame)
        output[frame_number] = processed_frame
    return output
