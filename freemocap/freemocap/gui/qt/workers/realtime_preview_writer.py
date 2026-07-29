import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

try:
    from mediapipe.python.solutions.pose import POSE_CONNECTIONS
except ImportError:
    POSE_CONNECTIONS = ()

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_SIZE = (1920, 1080)


class RealtimePreviewWriter(QThread):
    """Build and encode the three-panel real-time preview off the GUI thread."""

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
        # Matplotlib is not thread-safe alongside the live Qt canvas. Render
        # this static matching background once, before the worker starts, then
        # draw changing raw landmarks onto it with OpenCV in the worker.
        self._raw_plot_template = self._create_raw_pose_plot_template()

    def submit_frame(
        self,
        annotated_image: QImage,
        raw_pose_xyz: np.ndarray,
        filtered_plot_image: QImage,
    ) -> None:
        if not self._stop_requested.is_set():
            self._frames.put(
                (
                    time.monotonic() - self._recording_started_at,
                    annotated_image.copy(),
                    np.asarray(raw_pose_xyz, dtype=float).copy(),
                    filtered_plot_image.copy(),
                )
            )

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
                    (
                        elapsed_seconds,
                        annotated_image,
                        raw_pose_xyz,
                        filtered_plot_image,
                    ) = self._frames.get(timeout=0.05)
                except queue.Empty:
                    continue

                raw_plot_frame = self._render_raw_pose(
                    self._raw_plot_template,
                    raw_pose_xyz,
                )
                frame = self._compose_three_panel_frame(
                    annotated_frame=self._qimage_to_bgr_array(annotated_image),
                    raw_plot_frame=raw_plot_frame,
                    filtered_plot_frame=self._qimage_to_bgr_array(filtered_plot_image),
                    output_size=self.output_size,
                )
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
    def _create_raw_pose_plot_template():
        figure = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axes = figure.add_subplot(111, projection="3d")
        axes.set_title("Raw MediaPipe pose")
        axes.set_proj_type("ortho")
        axes.view_init(elev=0, azim=-90, roll=0)
        axes.set_xlim(-0.75, 0.75)
        axes.set_ylim(-0.75, 0.75)
        axes.set_zlim(0.0, 1.5)
        axes.set_xlabel("X")
        axes.set_ylabel("Depth")
        axes.set_zlabel("Height")
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        background = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        return background, axes.bbox.bounds

    @staticmethod
    def _render_raw_pose(raw_plot_template, pose_xyz: np.ndarray) -> np.ndarray:
        background, axes_bounds = raw_plot_template
        frame = background.copy()
        if pose_xyz.shape != (33, 3) or not np.isfinite(pose_xyz).any():
            return frame

        # With the same front orthographic view as the filtered plot, X maps
        # horizontally and height maps vertically. MediaPipe depth points into
        # the screen, so it does not change this particular camera projection.
        x = pose_xyz[:, 0] - 0.5
        height = 1.0 - pose_xyz[:, 1]
        axes_x, axes_y, axes_width, axes_height = axes_bounds
        image_height = frame.shape[0]
        pixel_x = axes_x + (x + 0.75) / 1.5 * axes_width
        pixel_y = image_height - (axes_y + height / 1.5 * axes_height)
        pixel_points = np.column_stack((pixel_x, pixel_y))
        valid = np.isfinite(pixel_points).all(axis=1)
        line_color = (180, 119, 31)

        for connection in POSE_CONNECTIONS:
            start, end = tuple(connection)
            if valid[start] and valid[end]:
                cv2.line(
                    frame,
                    tuple(np.rint(pixel_points[start]).astype(int)),
                    tuple(np.rint(pixel_points[end]).astype(int)),
                    line_color,
                    2,
                    cv2.LINE_AA,
                )
        for point in pixel_points[valid]:
            cv2.circle(
                frame,
                tuple(np.rint(point).astype(int)),
                2,
                line_color,
                -1,
                cv2.LINE_AA,
            )
        return frame

    @classmethod
    def _compose_three_panel_frame(
        cls,
        annotated_frame: np.ndarray,
        raw_plot_frame: np.ndarray,
        filtered_plot_frame: np.ndarray,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        output_width, output_height = output_size
        panel_width = output_width // 3
        output = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        for panel_index, panel_frame in enumerate(
            (annotated_frame, raw_plot_frame, filtered_plot_frame)
        ):
            fitted = cls._fit_inside_panel(panel_frame, panel_width, output_height)
            fitted_height, fitted_width = fitted.shape[:2]
            x_offset = panel_index * panel_width + (panel_width - fitted_width) // 2
            y_offset = (output_height - fitted_height) // 2
            output[
                y_offset : y_offset + fitted_height,
                x_offset : x_offset + fitted_width,
            ] = fitted
        return output

    @staticmethod
    def _fit_inside_panel(
        frame: np.ndarray,
        panel_width: int,
        panel_height: int,
    ) -> np.ndarray:
        input_height, input_width = frame.shape[:2]
        scale = min(panel_width / input_width, panel_height / input_height)
        resized_width = max(1, round(input_width * scale))
        resized_height = max(1, round(input_height * scale))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
