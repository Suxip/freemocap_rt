import logging
import time
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

logger = logging.getLogger(__name__)


class ImportedVideoFrameSource(QThread):
    """Replay a preset video in real time as QImages for the live mocap worker."""

    frame_ready_signal = Signal(QImage)
    playback_error_signal = Signal(str)

    def __init__(self, video_path: Path, parent=None):
        super().__init__(parent=parent)
        self.video_path = Path(video_path)

    def stop(self) -> bool:
        self.requestInterruption()
        return self.wait(5000)

    def run(self) -> None:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            message = f"Could not open imported video: {self.video_path}"
            logger.error(message)
            self.playback_error_signal.emit(message)
            return

        frames_per_second = capture.get(cv2.CAP_PROP_FPS)
        if frames_per_second <= 0:
            frames_per_second = 30.0
            logger.warning(
                "Imported video did not report a valid frame rate; replaying at 30 FPS"
            )

        frame_index = 0
        playback_started_at = time.monotonic()
        try:
            while not self.isInterruptionRequested():
                success, bgr_frame = capture.read()
                if not success:
                    return

                target_time = playback_started_at + frame_index / frames_per_second
                while not self.isInterruptionRequested():
                    remaining_seconds = target_time - time.monotonic()
                    if remaining_seconds <= 0:
                        break
                    self.msleep(max(1, min(20, round(remaining_seconds * 1000))))
                if self.isInterruptionRequested():
                    return

                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb_frame.shape
                image = QImage(
                    rgb_frame.data,
                    width,
                    height,
                    channels * width,
                    QImage.Format.Format_RGB888,
                ).copy()
                self.frame_ready_signal.emit(image)
                frame_index += 1
        except Exception as error:
            logger.exception("Failed while replaying imported video")
            self.playback_error_signal.emit(str(error))
        finally:
            capture.release()
