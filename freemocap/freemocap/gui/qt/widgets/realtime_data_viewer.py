import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSplitter, QStackedLayout, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from skelly_viewer import SkellyViewer

try:
    from mediapipe.python.solutions.pose import POSE_CONNECTIONS
except ImportError:
    POSE_CONNECTIONS = ()


class RealtimeDataViewer(QWidget):
    """Offline SkellyViewer plus a live annotated-video and pose display."""

    composite_frame_ready_signal = Signal(QImage)

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

    def update_live_frame(self, image: QImage, pose_xyz: np.ndarray) -> None:
        pixmap = QPixmap.fromImage(image)
        self._live_video_label.setPixmap(
            pixmap.scaled(
                self._live_video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._update_pose(pose_xyz)
        # Capture the same side-by-side widget the user sees. The writer owns a
        # copy so the GUI can safely repaint immediately afterward.
        self.composite_frame_ready_signal.emit(self._live_viewer.grab().toImage().copy())

    def show_live_error(self, message: str) -> None:
        self._live_video_label.setText(f"Real-time processing error: {message}")

    def load_skeleton_data(self, mediapipe_skeleton_npy_path):
        self._offline_viewer.load_skeleton_data(mediapipe_skeleton_npy_path)
        self.show_offline()

    def generate_video_display(self, video_folder_path):
        self._offline_viewer.generate_video_display(video_folder_path)

    def _initialize_plot(self) -> None:
        self._axes.set_title("Live pose")
        self._axes.set_xlim(-0.75, 0.75)
        self._axes.set_ylim(-0.75, 0.75)
        self._axes.set_zlim(0.0, 1.5)
        self._axes.set_xlabel("X")
        self._axes.set_ylabel("Depth")
        self._axes.set_zlabel("Height")
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
