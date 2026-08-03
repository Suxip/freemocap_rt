# Responsible for orchestrating post-processing pipeline

import logging
import multiprocessing
from pathlib import Path
from typing import Union

import numpy as np

from freemocap.core_processes.post_process_skeleton_data.causal_post_processing import (
    causally_post_process_skeleton,
)
from freemocap.data_layer.recording_models.post_processing_parameter_models import ProcessingParameterModel
from freemocap.system.logging.configure_logging import log_view_logging_format_string
from freemocap.system.logging.queue_logger import DirectQueueHandler
from freemocap.system.paths_and_filenames.file_and_folder_names import LOG_VIEW_PROGRESS_BAR_STRING

logger = logging.getLogger(__name__)


def save_numpy_array_to_disk(array_to_save: np.ndarray, file_name: str, save_directory: Union[str, Path]):
    if not file_name.endswith(".npy"):
        file_name += ".npy"
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    np.save(str(Path(save_directory) / file_name), array_to_save)


def post_process_data(
    recording_processing_parameter_model: ProcessingParameterModel,
    raw_skel3d_frame_marker_xyz: np.ndarray,
    queue: multiprocessing.Queue,
) -> np.ndarray:
    """Fill missing skeleton coordinates with a causal Kalman filter."""
    if queue:
        handler = DirectQueueHandler(queue)
        handler.setFormatter(logging.Formatter(fmt=log_view_logging_format_string, datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(handler)

    logger.info("Starting causal Kalman post-processing")
    logger.info(LOG_VIEW_PROGRESS_BAR_STRING)
    processed = causally_post_process_skeleton(
        skeleton_data=raw_skel3d_frame_marker_xyz,
        parameters=recording_processing_parameter_model.post_processing_parameters_model,
    )
    logger.info("Done with causal Kalman gap filling")
    return processed
