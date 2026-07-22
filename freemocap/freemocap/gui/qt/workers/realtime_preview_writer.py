import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_SIZE = (1920, 1080)


class RealtimePreviewWriter(QThread):
    """Encode Data Viewer composite frames without blocking the GUI thread."""

    preview_saved_signal = Signal(str)
    writing_error_signal = Signal(str)

    def __init__(
        self,
        output_path: Path,
        frames_per_second: float = 30.0,
        output_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.output_path = Path(output_path)
        self.frames_per_second = frames_per_second
        self.output_size = output_size
        if self.output_size[0] <= 0 or self.output_size[1] <= 0:
            raise ValueError("output_size must contain positive dimensions")
        if self.output_size[0] % 2 or self.output_size[1] % 2:
            raise ValueError("output_size dimensions must be even for MP4 encoding")
        self._frames = queue.Queue()
        self._stop_requested = threading.Event()
        self._recording_started_at = time.monotonic()
        self._stopped_at = None

    def submit_frame(self, image: QImage) -> None:
        if not self._stop_requested.is_set():
            self._frames.put((time.monotonic() - self._recording_started_at, image.copy()))

    def stop(self) -> bool:
        self._stopped_at = time.monotonic() - self._recording_started_at
        self._stop_requested.set()
        return self.wait(15000)

    def run(self) -> None:
        writer = None
        last_frame = None
        written_frame_count = 0
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            while not self._stop_requested.is_set() or not self._frames.empty():
                try:
                    elapsed_seconds, image = self._frames.get(timeout=0.05)
                except queue.Empty:
                    continue

                frame = self._qimage_to_bgr_array(image)
                frame = self._letterbox_to_output_size(frame, self.output_size)
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(self.output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.frames_per_second,
                        self.output_size,
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open video writer for {self.output_path}")

                target_frame_index = max(
                    written_frame_count,
                    round(elapsed_seconds * self.frames_per_second),
                )
                while written_frame_count < target_frame_index:
                    writer.write(frame if last_frame is None else last_frame)
                    written_frame_count += 1
                writer.write(frame)
                written_frame_count += 1
                last_frame = frame

            if writer is not None and last_frame is not None and self._stopped_at is not None:
                final_frame_count = round(self._stopped_at * self.frames_per_second)
                while written_frame_count < final_frame_count:
                    writer.write(last_frame)
                    written_frame_count += 1
        except Exception as error:
            logger.exception("Failed to save the real-time preview")
            self.writing_error_signal.emit(str(error))
            return
        finally:
            if writer is not None:
                writer.release()

        if writer is not None:
            logger.info(f"Saved real-time preview to {self.output_path}")
            self.preview_saved_signal.emit(str(self.output_path))

    @staticmethod
    def _qimage_to_bgr_array(image: QImage) -> np.ndarray:
        converted = image.convertToFormat(QImage.Format.Format_RGB888)
        height, width = converted.height(), converted.width()
        rgb = np.frombuffer(converted.bits(), dtype=np.uint8).reshape(height, converted.bytesPerLine())
        rgb = rgb[:, : width * 3].reshape(height, width, 3).copy()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _letterbox_to_output_size(frame: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
        """Fit a frame in a fixed canvas without cropping or changing its aspect ratio."""
        output_width, output_height = output_size
        input_height, input_width = frame.shape[:2]
        scale = min(output_width / input_width, output_height / input_height)
        resized_width = max(1, round(input_width * scale))
        resized_height = max(1, round(input_height * scale))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)

        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        x_offset = (output_width - resized_width) // 2
        y_offset = (output_height - resized_height) // 2
        canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
        return canvas
