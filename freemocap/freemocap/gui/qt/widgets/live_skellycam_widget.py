from typing import Dict

from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QImage
from skellycam import SkellyCamWidget


class LiveSkellyCamWidget(SkellyCamWidget):
    """SkellyCam widget that also exposes frames to the real-time pipeline."""

    live_image_signal = Signal(str, QImage)

    @Slot(str, QImage, dict)
    def _handle_image_update(self, camera_id: str, q_image: QImage, frame_diagnostics_dictionary: Dict):
        super()._handle_image_update(camera_id, q_image, frame_diagnostics_dictionary)
        # SkellyCam reuses buffers internally, so consumers receive an owned copy.
        self.live_image_signal.emit(str(camera_id), q_image.copy())
