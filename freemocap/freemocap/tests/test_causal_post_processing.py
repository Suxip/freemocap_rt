import numpy as np

from freemocap.core_processes.post_process_skeleton_data.causal_post_processing import (
    causally_post_process_skeleton,
)
from freemocap.data_layer.recording_models.post_processing_parameter_models import (
    PostProcessingParametersModel,
)


def _sample_data() -> np.ndarray:
    time = np.arange(40, dtype=float)
    data = np.stack((time, 2.0 * time, -time), axis=1)[:, None, :]
    data[5:8, 0, 0] = np.nan
    data[15:30, 0, 1] = np.nan
    data[:3, 0, 2] = np.nan
    return data


def test_causal_post_processing_is_prefix_invariant():
    data = _sample_data()
    parameters = PostProcessingParametersModel(max_gap_to_fill=5)
    complete_result = causally_post_process_skeleton(data, parameters)

    for stop in range(1, len(data) + 1):
        prefix_result = causally_post_process_skeleton(data[:stop], parameters)
        np.testing.assert_allclose(prefix_result, complete_result[:stop], equal_nan=True)


def test_kalman_only_fills_up_to_maximum_gap():
    data = _sample_data()
    parameters = PostProcessingParametersModel(max_gap_to_fill=2, run_one_euro_filter=False)
    result = causally_post_process_skeleton(data, parameters)

    assert np.isfinite(result[5:7, 0, 0]).all()
    assert np.isnan(result[7, 0, 0])
    assert np.isnan(result[:3, 0, 2]).all()


def test_one_euro_output_changes_with_current_measurement():
    data = np.zeros((5, 1, 3), dtype=float)
    data[-1] = 10.0
    result = causally_post_process_skeleton(data, PostProcessingParametersModel())

    assert np.all(result[-1] > 0.0)
    assert np.all(result[-1] < 10.0)
