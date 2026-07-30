import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSplitter, QStackedLayout, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from skelly_viewer import SkellyViewer

from freemocap.gui.qt.utilities.realtime_pose_plot import configure_realtime_pose_axes

try:
    from mediapipe.python.solutions.pose import POSE_CONNECTIONS
except ImportError:
    POSE_CONNECTIONS = ()


class RealtimeDataViewer(QWidget):
    """Offline SkellyViewer plus a live annotated-video and pose display."""

    composite_frame_ready_signal = Signal(QImage, object, QImage, float)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._offline_viewer = SkellyViewer()
        self._live_viewer = QWidget()
        self._live_video_label = QLabel("Start a motion capture recording to view live data")
        self._live_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._live_video_label.setMinimumSize(320, 240)
        self._figure = Figure()
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._axes = self._figure.add_subplot(111, projection="3d")
        self._initialize_plot()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._live_video_label)
        splitter.addWidget(self._canvas)
        live_layout = QVBoxLayout(self._live_viewer)
        live_layout.addWidget(splitter)

        self._stack = QStackedLayout(self)
        self._stack.addWidget(self._offline_viewer)
        self._stack.addWidget(self._live_viewer)
        self.show_offline()

    def start_live(self) -> None:
        self._live_video_label.setText("Waiting for the first tracked frame…")
        self._stack.setCurrentWidget(self._live_viewer)

    def stop_live(self) -> None:
        self._live_video_label.setText("Recording stopped")

    def show_offline(self) -> None:
        self._stack.setCurrentWidget(self._offline_viewer)

    def update_live_frame(
        self,
        image: QImage,
        raw_pose_xyz: np.ndarray,
        filtered_pose_xyz: np.ndarray,
    ) -> None:
        pixmap = QPixmap.fromImage(image)
        self._live_video_label.setPixmap(
            pixmap.scaled(
                self._live_video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._update_pose(filtered_pose_xyz)
        filtered_plot_image = self._matplotlib_canvas_to_image(self._canvas)
        effective_dpi = (
            filtered_plot_image.width() / self._figure.get_figwidth()
        )
        self.composite_frame_ready_signal.emit(
            image.copy(),
            raw_pose_xyz.copy(),
            filtered_plot_image,
            effective_dpi,
        )

    def show_live_error(self, message: str) -> None:
        self._live_video_label.setText(f"Real-time processing error: {message}")

    def load_skeleton_data(self, mediapipe_skeleton_npy_path):
        self._offline_viewer.load_skeleton_data(mediapipe_skeleton_npy_path)
        self.show_offline()

    def generate_video_display(self, video_folder_path):
        self._offline_viewer.generate_video_display(video_folder_path)

    def _initialize_plot(self) -> None:
        configure_realtime_pose_axes(self._axes, title="Real-time filtered pose")
        self._points = self._axes.scatter([], [], [], s=12)
        self._bones = [self._axes.plot([], [], [], linewidth=2)[0] for _ in POSE_CONNECTIONS]

    def _update_pose(self, pose_xyz: np.ndarray) -> None:
        if pose_xyz.shape != (33, 3) or not np.isfinite(pose_xyz).any():
            return
        # Convert MediaPipe image coordinates to a centered, upright display.
        x = pose_xyz[:, 0] - 0.5
        depth = -pose_xyz[:, 2]
        height = 1.0 - pose_xyz[:, 1]
        self._points._offsets3d = (x, depth, height)
        for line, connection in zip(self._bones, POSE_CONNECTIONS):
            start, end = tuple(connection)
            line.set_data_3d(
                [x[start], x[end]],
                [depth[start], depth[end]],
                [height[start], height[end]],
            )
        # Render now so the composite capture contains this frame's graph rather
        # than the graph from the previous event-loop iteration.
        self._canvas.draw()

    @staticmethod
    def _matplotlib_canvas_to_image(canvas: FigureCanvasQTAgg) -> QImage:
        rgba = np.asarray(canvas.buffer_rgba())
        height, width, channels = rgba.shape
        return QImage(
            rgba.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGBA8888,
        ).copy()
