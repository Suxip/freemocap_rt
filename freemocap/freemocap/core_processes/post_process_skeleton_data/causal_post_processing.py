"""Stateful, forward-only post-processing for 3-D skeleton trajectories."""

import numpy as np


class KalmanGapFiller:
    """Constant-velocity Kalman filter used to predict short missing-data gaps."""

    def __init__(self, shape: tuple[int, int], sampling_rate: float, process_noise: float,
                 measurement_noise: float, max_gap_to_fill: int):
        self.dt = 1.0 / sampling_rate
        self.transition = np.array([[1.0, self.dt], [0.0, 1.0]])
        self.observation = np.array([1.0, 0.0])
        self.process_covariance = process_noise * np.array(
            [[self.dt ** 4 / 4.0, self.dt ** 3 / 2.0],
             [self.dt ** 3 / 2.0, self.dt ** 2]]
        )
        self.measurement_noise = measurement_noise
        self.max_gap_to_fill = max_gap_to_fill
        self.state = np.zeros((*shape, 2), dtype=float)
        self.covariance = np.zeros((*shape, 2, 2), dtype=float)
        self.initialized = np.zeros(shape, dtype=bool)
        self.missing_count = np.zeros(shape, dtype=int)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=float)
        output = np.full_like(frame, np.nan)
        valid = np.isfinite(frame)

        new_values = valid & ~self.initialized
        self.state[new_values, 0] = frame[new_values]
        self.state[new_values, 1] = 0.0
        self.covariance[new_values] = np.eye(2)
        self.initialized[new_values] = True

        active = self.initialized & ~new_values
        if np.any(active):
            predicted_state = self.state[active] @ self.transition.T
            predicted_covariance = (
                self.transition @ self.covariance[active] @ self.transition.T
                + self.process_covariance
            )
            self.state[active] = predicted_state
            self.covariance[active] = predicted_covariance

        measured = valid & self.initialized & ~new_values
        if np.any(measured):
            innovation = frame[measured] - self.state[measured, 0]
            innovation_covariance = self.covariance[measured, 0, 0] + self.measurement_noise
            gain = self.covariance[measured, :, 0] / innovation_covariance[:, None]
            self.state[measured] += gain * innovation[:, None]
            identity_minus_kh = np.eye(2)[None, :, :] - gain[:, :, None] * self.observation[None, None, :]
            self.covariance[measured] = identity_minus_kh @ self.covariance[measured]

        self.missing_count[valid] = 0
        self.missing_count[~valid] += 1
        # Preserve real observations and use the Kalman prediction only where
        # the observation is missing.
        output[valid] = frame[valid]
        may_predict = ~valid & self.initialized & (self.missing_count <= self.max_gap_to_fill)
        output[may_predict] = self.state[may_predict, 0]
        return output


def causally_post_process_skeleton(skeleton_data: np.ndarray, parameters) -> np.ndarray:
    """Process a complete array sequentially without inspecting future frames."""
    data = np.asarray(skeleton_data, dtype=float)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError("skeleton_data must have shape (frames, markers, 3)")
    if data.shape[0] == 0:
        return data.copy()

    shape = data.shape[1:]
    kalman_parameters = parameters.kalman_filter_parameters
    gap_filler = KalmanGapFiller(
        shape=shape,
        sampling_rate=parameters.framerate,
        process_noise=kalman_parameters.process_noise,
        measurement_noise=kalman_parameters.measurement_noise,
        max_gap_to_fill=parameters.max_gap_to_fill,
    )
    output = np.empty_like(data, dtype=float)
    for frame_number, frame in enumerate(data):
        output[frame_number] = gap_filler.process_frame(frame)
    return output
