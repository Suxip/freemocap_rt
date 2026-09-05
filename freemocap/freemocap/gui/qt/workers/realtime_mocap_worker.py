import logging
import threading

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from skellytracker.trackers.mediapipe_tracker.mediapipe_holistic_tracker import MediapipeHolisticTracker

from freemocap.core_processes.post_process_skeleton_data.causal_post_processing import (
    KalmanGapFiller,
    OneEuroFilter,
)

logger = logging.getLogger(__name__)


class RealtimeMocapWorker(QThread):
    """Track only the newest camera frame so the GUI cannot accumulate lag."""

    frame_processed_signal = Signal(QImage, object, object)
    processing_error_signal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._condition = threading.Condition()
        self._latest_image = None
        self._should_stop = False
        self._finish_when_idle = False

    def submit_frame(self, image: QImage) -> None:
        with self._condition:
            self._latest_image = image.copy()
            self._condition.notify()

    def stop(self) -> bool:
        with self._condition:
            self._should_stop = True
            self._condition.notify()
        return self.wait(10000)

    def finish_after_pending_frame(self) -> None:
        """Finish after processing the newest frame already submitted."""
        with self._condition:
            self._finish_when_idle = True
            self._condition.notify()

    def run(self) -> None:
        try:
            tracker = MediapipeHolisticTracker(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                static_image_mode=False,
                smooth_landmarks=False,
            )
            gap_filler = KalmanGapFiller(
                shape=(33, 3),
                sampling_rate=30.0,
                process_noise=1.0,
                measurement_noise=10.0,
                max_gap_to_fill=10,
            )
            one_euro_filter = OneEuroFilter(
                shape=(33, 3),
                sampling_rate=30.0,
                min_cutoff=1.0,
                beta=0.007,
                derivative_cutoff=1.0,
            )
            while True:
                with self._condition:
                    while (
                        self._latest_image is None
                        and not self._should_stop
                        and not self._finish_when_idle
                    ):
                        self._condition.wait()
                    if self._should_stop:
                        return
                    if self._latest_image is None and self._finish_when_idle:
                        return
                    image = self._latest_image
                    self._latest_image = None

                rgb_image = self._qimage_to_rgb_array(image)
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                tracked_objects = tracker.process_image(bgr_image)
                pose_landmarks = tracked_objects["pose_landmarks"].extra["landmarks"]
                raw_pose_xyz = self._pose_to_array(pose_landmarks)
                filtered_pose_xyz = one_euro_filter.process_frame(
                    gap_filler.process_frame(raw_pose_xyz.copy())
                )
                annotated_rgb = cv2.cvtColor(tracker.annotated_image, cv2.COLOR_BGR2RGB)
                annotated_qimage = self._rgb_array_to_qimage(annotated_rgb)
                self.frame_processed_signal.emit(
                    annotated_qimage,
                    raw_pose_xyz,
                    filtered_pose_xyz,
                )
        except Exception as error:
            logger.exception("Real-time motion capture failed")
            self.processing_error_signal.emit(str(error))

    @staticmethod
    def _qimage_to_rgb_array(image: QImage) -> np.ndarray:
        converted = image.convertToFormat(QImage.Format.Format_RGB888)
        height, width = converted.height(), converted.width()
        array = np.frombuffer(converted.bits(), dtype=np.uint8).reshape(height, converted.bytesPerLine())
        return array[:, : width * 3].reshape(height, width, 3).copy()

    @staticmethod
    def _rgb_array_to_qimage(image: np.ndarray) -> QImage:
        height, width, channels = image.shape
        return QImage(
            image.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()

    @staticmethod
    def _pose_to_array(pose_landmarks) -> np.ndarray:
        if pose_landmarks is None:
            return np.full((33, 3), np.nan)
        return np.asarray([[landmark.x, landmark.y, landmark.z] for landmark in pose_landmarks.landmark], dtype=float)
