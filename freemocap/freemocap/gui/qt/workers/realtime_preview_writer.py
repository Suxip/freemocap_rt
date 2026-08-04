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
from mpl_toolkits.mplot3d import proj3d

from freemocap.gui.qt.utilities.realtime_pose_plot import configure_realtime_pose_axes
from freemocap.utilities.realtime_jitter_analysis import (
    save_realtime_pose_data_and_jitter_report,
)

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
        self._raw_plot_template_size = (
            self._raw_plot_template[0].shape[1],
            self._raw_plot_template[0].shape[0],
        )
        self._raw_plot_template_dpi = 100.0
        self._video_title_overlay = self._create_title_overlay(
            width=self._raw_plot_template_size[0],
            height=self._raw_plot_template_size[1],
            dpi=self._raw_plot_template_dpi,
            title="Video Recording",
        )

    def submit_frame(
        self,
        annotated_image: QImage,
        raw_pose_xyz: np.ndarray,
        filtered_pose_xyz: np.ndarray,
        filtered_plot_image: QImage,
        filtered_plot_dpi: float,
    ) -> None:
        if not self._stop_requested.is_set():
            filtered_plot_size = (
                filtered_plot_image.width(),
                filtered_plot_image.height(),
            )
            if (
                filtered_plot_size != self._raw_plot_template_size
                or filtered_plot_dpi != self._raw_plot_template_dpi
            ):
                self._raw_plot_template = self._create_raw_pose_plot_template(
                    width=filtered_plot_size[0],
                    height=filtered_plot_size[1],
                    dpi=filtered_plot_dpi,
                )
                self._raw_plot_template_size = filtered_plot_size
                self._raw_plot_template_dpi = filtered_plot_dpi
                self._video_title_overlay = self._create_title_overlay(
                    width=filtered_plot_size[0],
                    height=filtered_plot_size[1],
                    dpi=filtered_plot_dpi,
                    title="Video Recording",
                )
            self._frames.put(
                (
                    time.monotonic() - self._recording_started_at,
                    annotated_image.copy(),
                    np.asarray(raw_pose_xyz, dtype=float).copy(),
                    np.asarray(filtered_pose_xyz, dtype=float).copy(),
                    filtered_plot_image.copy(),
                    self._raw_plot_template,
                    self._video_title_overlay,
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
        pose_timestamps = []
        raw_pose_frames = []
        filtered_pose_frames = []
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            while not self._stop_requested.is_set() or not self._frames.empty():
                try:
                    (
                        elapsed_seconds,
                        annotated_image,
                        raw_pose_xyz,
                        filtered_pose_xyz,
                        filtered_plot_image,
                        raw_plot_template,
                        video_title_overlay,
                    ) = self._frames.get(timeout=0.05)
                except queue.Empty:
                    continue

                pose_timestamps.append(elapsed_seconds)
                raw_pose_frames.append(raw_pose_xyz)
                filtered_pose_frames.append(filtered_pose_xyz)

                raw_plot_frame = self._render_raw_pose(
                    raw_plot_template,
                    raw_pose_xyz,
                )
                filtered_plot_frame = self._qimage_to_bgr_array(filtered_plot_image)
                annotated_frame = self._place_on_white_canvas_to_match(
                    self._qimage_to_bgr_array(annotated_image),
                    filtered_plot_frame=filtered_plot_frame,
                    title_overlay=video_title_overlay,
                )
                frame = self._compose_three_panel_frame(
                    annotated_frame=annotated_frame,
                    raw_plot_frame=raw_plot_frame,
                    filtered_plot_frame=filtered_plot_frame,
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
            save_realtime_pose_data_and_jitter_report(
                preview_output_path=self.output_path,
                timestamps=np.asarray(pose_timestamps, dtype=float),
                raw_pose=np.asarray(raw_pose_frames, dtype=float),
                filtered_pose=np.asarray(filtered_pose_frames, dtype=float),
            )
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
    def _create_raw_pose_plot_template(
        width: int = 640,
        height: int = 480,
        dpi: float = 100.0,
    ):
        figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        canvas = FigureCanvasAgg(figure)
        axes = figure.add_subplot(111, projection="3d")
        configure_realtime_pose_axes(axes, title="Raw MediaPipe pose")
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        background = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        return background, axes

    @staticmethod
    def _create_title_overlay(
        width: int,
        height: int,
        dpi: float,
        title: str,
    ) -> np.ndarray:
        figure = Figure(
            figsize=(width / dpi, height / dpi),
            dpi=dpi,
            facecolor=(0.0, 0.0, 0.0, 0.0),
        )
        canvas = FigureCanvasAgg(figure)
        axes = figure.add_subplot(111, projection="3d")
        configure_realtime_pose_axes(axes, title=title)
        axes.set_axis_off()
        axes.patch.set_alpha(0.0)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba()).copy()

    @staticmethod
    def _render_raw_pose(raw_plot_template, pose_xyz: np.ndarray) -> np.ndarray:
        background, axes = raw_plot_template
        frame = background.copy()
        if pose_xyz.shape != (33, 3) or not np.isfinite(pose_xyz).any():
            return frame

        # Apply the same coordinate mapping and Matplotlib 3D projection used by
        # the filtered graph. Only the final raster drawing uses OpenCV.
        x = pose_xyz[:, 0] - 0.5
        depth = -pose_xyz[:, 2]
        height = 1.0 - pose_xyz[:, 1]
        projected_x, projected_y, _ = proj3d.proj_transform(
            x,
            depth,
            height,
            axes.get_proj(),
        )
        display_points = axes.transData.transform(
            np.column_stack((projected_x, projected_y))
        )
        image_height = frame.shape[0]
        pixel_points = display_points
        pixel_points[:, 1] = image_height - pixel_points[:, 1]
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

    @classmethod
    def _place_on_white_canvas_to_match(
        cls,
        frame: np.ndarray,
        filtered_plot_frame: np.ndarray,
        title_overlay: np.ndarray,
    ) -> np.ndarray:
        """Center a video frame on a graph-sized white canvas without distortion."""
        canvas_height, canvas_width = filtered_plot_frame.shape[:2]
        canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
        fitted = cls._fit_inside_panel(frame, canvas_width, canvas_height)
        fitted_height, fitted_width = fitted.shape[:2]
        x_offset = (canvas_width - fitted_width) // 2
        y_offset = (canvas_height - fitted_height) // 2
        canvas[
            y_offset : y_offset + fitted_height,
            x_offset : x_offset + fitted_width,
        ] = fitted
        overlay_bgr = cv2.cvtColor(title_overlay, cv2.COLOR_RGBA2BGR)
        alpha = title_overlay[:, :, 3:4].astype(np.float32) / 255.0
        canvas = (
            overlay_bgr.astype(np.float32) * alpha
            + canvas.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
        return canvas
