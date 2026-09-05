import logging
import multiprocessing
import shutil
import threading
from pathlib import Path
from typing import Union, List, Callable

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QMainWindow,
    QFileDialog,
    QWidget,
    QHBoxLayout,
)
from skellycam import (
    SkellyCamParameterTreeWidget,
)
from tqdm import tqdm

from freemocap.data_layer.generate_jupyter_notebook.generate_jupyter_notebook import (
    generate_jupyter_notebook,
)
from freemocap.data_layer.recording_models.post_processing_parameter_models import (
    ProcessingParameterModel,
)
from freemocap.data_layer.recording_models.recording_info_model import (
    RecordingInfoModel,
)
from freemocap.gui.qt.actions_and_menus.actions import Actions
from freemocap.gui.qt.actions_and_menus.menu_bar import MenuBar
from freemocap.gui.qt.style_sheet.css_file_watcher import CSSFileWatcher
from freemocap.gui.qt.style_sheet.scss_file_watcher import SCSSFileWatcher
from freemocap.gui.qt.style_sheet.set_css_style_sheet import apply_css_style_sheet
from freemocap.gui.qt.utilities.copy_timestamps_folder import copy_directory_if_contains_timestamps
from freemocap.gui.qt.utilities.get_qt_app import get_qt_app
from freemocap.gui.qt.utilities.save_and_load_gui_state import (
    GuiState,
    load_gui_state,
    save_gui_state,
)
from freemocap.gui.qt.utilities.update_most_recent_recording_toml import (
    update_most_recent_recording_toml,
)
from freemocap.gui.qt.widgets.active_recording_widget import ActiveRecordingInfoWidget
from freemocap.gui.qt.widgets.camera_controller_group_box import CameraControllerGroupBox
from freemocap.gui.qt.widgets.central_tab_widget import CentralTabWidget
from freemocap.gui.qt.widgets.control_panel.control_panel_dock_widget import (
    ControlPanelWidget,
)
from freemocap.gui.qt.widgets.control_panel.export_data_control_panel import VisualizationControlPanel
from freemocap.gui.qt.widgets.control_panel.process_mocap_data_panel.process_motion_capture_data_panel import (
    ProcessMotionCaptureDataPanel,
)
from freemocap.gui.qt.widgets.directory_view_widget import DirectoryViewWidget
from freemocap.gui.qt.widgets.home_widget import (
    HomeWidget,
)
from freemocap.gui.qt.widgets.import_videos_wizard import ImportVideosWizard
from freemocap.gui.qt.widgets.log_view_widget import LogViewWidget
from freemocap.gui.qt.widgets.live_skellycam_widget import LiveSkellyCamWidget
from freemocap.gui.qt.widgets.opencv_conflict_dialog import OpencvConflictDialog
from freemocap.gui.qt.widgets.realtime_data_viewer import RealtimeDataViewer
from freemocap.gui.qt.widgets.release_notes_dialogs.tabbed_release_notes_dialog import TabbedReleaseNotesDialog
from freemocap.gui.qt.widgets.set_data_folder_dialog import SetDataFolderDialog
from freemocap.gui.qt.widgets.welcome_screen_dialog import WelcomeScreenDialog
from freemocap.gui.qt.workers.download_sample_data_thread_worker import DownloadDataThreadWorker
from freemocap.gui.qt.workers.export_to_blender_thread_worker import ExportToBlenderThreadWorker
from freemocap.gui.qt.workers.imported_video_frame_source import ImportedVideoFrameSource
from freemocap.gui.qt.workers.realtime_mocap_worker import RealtimeMocapWorker
from freemocap.gui.qt.workers.realtime_preview_writer import RealtimePreviewWriter
# reboot GUI method based on this - https://stackoverflow.com/a/56563926/14662833
from freemocap.system.open_file import open_file
from freemocap.system.paths_and_filenames.file_and_folder_names import (
    PATH_TO_FREEMOCAP_LOGO_SVG,
)
from freemocap.system.paths_and_filenames.path_getters import (
    get_recording_session_folder_path,
    get_css_stylesheet_path,
    get_scss_stylesheet_path,
    get_blender_file_path,
    get_most_recent_recording_path,
    get_gui_state_json_path,
)
from freemocap.system.user_data.pipedream_pings import PipedreamPings
from freemocap.utilities.remove_empty_directories import remove_empty_directories

EXIT_CODE_REBOOT = -123456789

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(
            self,
            freemocap_data_folder_path: Union[str, Path],
            pipedream_pings: PipedreamPings,
            parent=None,
    ):
        super().__init__(parent=parent)
        self._log_view_widget = LogViewWidget(parent=self)  # start this first so it will grab the setup logs
        logger.info("Initializing FreeMoCap MainWindow")

        self.setMinimumSize(1280, 720)
        self.setWindowIcon(QIcon(PATH_TO_FREEMOCAP_LOGO_SVG))
        self.setWindowTitle("freemocap \U0001f480 \U00002728")

        dummy_widget = QWidget()
        self._layout = QHBoxLayout()
        dummy_widget.setLayout(self._layout)
        self.setCentralWidget(dummy_widget)

        self._css_file_watcher = self._set_up_stylesheet()

        self._freemocap_data_folder_path = freemocap_data_folder_path
        self._pipedream_pings = pipedream_pings

        self._gui_state: GuiState = load_gui_state(get_gui_state_json_path())

        self._kill_thread_event = multiprocessing.Event()
        self._realtime_worker = None
        self._realtime_preview_writer = None
        self._imported_video_frame_source = None
        self._imported_realtime_processing_active = False
        self._realtime_processing_active = False
        self._realtime_camera_id = None

        self._active_recording_info_widget = ActiveRecordingInfoWidget(parent=self)
        self._active_recording_info_widget.new_active_recording_selected_signal.connect(
            self._handle_new_active_recording_selected
        )
        self._directory_view_widget = self._create_directory_view_widget()
        self._directory_view_widget.expand_directory_to_path(get_recording_session_folder_path())
        self._directory_view_widget.new_active_recording_selected_signal.connect(
            self._active_recording_info_widget.set_active_recording
        )

        self._actions = Actions(freemocap_main_window=self)

        self._menu_bar = MenuBar(actions=self._actions, parent=self)
        self.setMenuBar(self._menu_bar)

        self.statusBar().showMessage(
            "Watch the terminal output for status updates, we're working on integrating better status updates into the GUI"
        )

        self._central_tab_widget = self._create_central_tab_widget()
        self.setCentralWidget(self._central_tab_widget)

        self._tools_dock_widget = self._create_tools_dock_widget()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._tools_dock_widget)

        self._control_panel_widget = self._create_control_panel_widget(log_update=self._log_view_widget.add_log)
        self._tools_dock_widget.setWidget(self._control_panel_widget)

        self._connect_calibration_updates()

        log_view_dock_widget = QDockWidget("Log View", self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_view_dock_widget)
        log_view_dock_widget.setWidget(self._log_view_widget)
        log_view_dock_widget.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        logger.debug("Finished initializing FreeMoCap MainWindow")

    def _create_tools_dock_widget(self):
        tools_dock_widget = QDockWidget("Control Panel", self)
        tools_dock_widget.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        tools_dock_widget.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        return tools_dock_widget

    def _handle_videos_saved_to_this_folder_signal(self, folder_path: str):
        logger.debug(f"Videos saved to this folder signal received: {folder_path}")

        self._active_recording_info_widget.set_active_recording(recording_folder_path=folder_path)

        if (
                self._controller_group_box.auto_process_videos_checked
                and self._controller_group_box.mocap_videos_radio_button_checked
        ):
            logger.info("'Auto process videos' checkbox is checked - triggering 'Process Motion Capture Data' button")
            self._process_motion_capture_data_panel.process_motion_capture_data_button.click()
        elif self._controller_group_box.calibration_videos_radio_button_checked:
            logger.info("Processing calibration videos")
            self._process_motion_capture_data_panel.calibrate_from_active_recording(
                charuco_square_size_mm=float(self._controller_group_box.charuco_square_size),
                use_charuco_as_groundplane=self._controller_group_box.use_charuco_as_groundplane,
                charuco_board_name=self._controller_group_box.charuco_board_name,
            )

    def _handle_processing_finished_signal(self):
        self._update_skelly_viewer_widget()
        if self._controller_group_box.auto_open_in_blender_checked and not self._kill_thread_event.is_set():
            logger.info("'Auto Open in Blender' checkbox is checked - triggering 'Create Blender Scene'")
            self._export_active_recording_to_blender()
        if self._controller_group_box.generate_jupyter_notebook_checked and not self._kill_thread_event.is_set():
            self._generate_jupyter_notebook()
        logger.info("Processing finished")

    def handle_start_new_session_action(self):
        # self._central_tab_widget.set_welcome_tab_enabled(True)
        self._central_tab_widget.set_camera_view_tab_enabled(True)
        self._central_tab_widget.setCurrentIndex(1)
        self._controller_group_box.show()
        self._skellycam_widget.detect_available_cameras()

    def _create_realtime_mocap_worker(self) -> RealtimeMocapWorker:
        """Create the single processing pipeline used by live and imported frames."""
        worker = RealtimeMocapWorker(parent=self)
        worker.frame_processed_signal.connect(
            self._skelly_viewer_widget.update_live_frame
        )
        worker.processing_error_signal.connect(
            self._skelly_viewer_widget.show_live_error
        )
        return worker

    def _start_realtime_processing(self):
        if not self._controller_group_box.mocap_videos_radio_button_checked:
            return
        self._stop_imported_video_processing()
        self._stop_realtime_worker()
        if self._realtime_worker is not None:
            logger.error("The previous real-time worker is still stopping; cannot start another one yet")
            return
        self._realtime_worker = self._create_realtime_mocap_worker()
        self._realtime_worker.start()
        self._start_realtime_preview_writer()
        self._realtime_camera_id = None
        self._realtime_processing_active = True
        self._skelly_viewer_widget.start_live()
        self._central_tab_widget.setCurrentIndex(2)
        logger.info("Started real-time tracking and visualization")

    def _start_imported_video_realtime_processing(self, video_path: Path) -> None:
        self._stop_imported_video_processing()
        self._stop_realtime_worker()
        if self._realtime_worker is not None:
            logger.error(
                "The previous real-time worker is still stopping; "
                "cannot process the imported video yet"
            )
            return

        self._realtime_processing_active = False
        self._realtime_camera_id = None
        self._realtime_worker = self._create_realtime_mocap_worker()
        self._realtime_worker.finished.connect(
            self._handle_imported_realtime_worker_finished
        )
        self._realtime_worker.start()
        self._start_realtime_preview_writer()

        self._imported_video_frame_source = ImportedVideoFrameSource(
            video_path=video_path,
            parent=self,
        )
        self._imported_video_frame_source.frame_ready_signal.connect(
            self._realtime_worker.submit_frame
        )
        self._imported_video_frame_source.playback_error_signal.connect(
            self._skelly_viewer_widget.show_live_error
        )
        self._imported_video_frame_source.finished.connect(
            self._handle_imported_video_source_finished
        )
        self._imported_realtime_processing_active = True
        self._skelly_viewer_widget.start_live()
        self._central_tab_widget.setCurrentIndex(2)
        self.statusBar().showMessage(
            f"Running real-time processing on imported video: {video_path.name}"
        )
        self._imported_video_frame_source.start()
        logger.info("Started real-time processing for imported video: %s", video_path)

    def _handle_imported_video_source_finished(self) -> None:
        source = self._imported_video_frame_source
        self._imported_video_frame_source = None
        if source is not None:
            source.deleteLater()
        if (
            self._imported_realtime_processing_active
            and self._realtime_worker is not None
        ):
            self._realtime_worker.finish_after_pending_frame()

    def _handle_imported_realtime_worker_finished(self) -> None:
        if not self._imported_realtime_processing_active:
            return
        self._imported_realtime_processing_active = False
        self._stop_imported_video_source()
        self._stop_realtime_preview_writer()
        self._skelly_viewer_widget.stop_live()
        if self._realtime_worker is not None:
            self._realtime_worker.deleteLater()
            self._realtime_worker = None
        active_recording = self._active_recording_info_widget.get_active_recording_info()
        if active_recording is None:
            saved_message = "Finished real-time processing of imported video"
        else:
            saved_message = (
                "Finished imported-video processing. Saved recording to: "
                f"{active_recording.path}"
            )
        self.statusBar().showMessage(saved_message)
        logger.info(saved_message)

    def _stop_imported_video_source(self) -> None:
        if self._imported_video_frame_source is None:
            return
        source = self._imported_video_frame_source
        if not source.stop():
            logger.warning("Imported video frame source did not stop within 5 seconds")
            return
        source.deleteLater()
        self._imported_video_frame_source = None

    def _stop_imported_video_processing(self) -> None:
        if not self._imported_realtime_processing_active:
            self._stop_imported_video_source()
            return
        self._imported_realtime_processing_active = False
        self._stop_imported_video_source()
        self._stop_realtime_worker()
        self._stop_realtime_preview_writer()
        self._skelly_viewer_widget.stop_live()

    def _stop_realtime_processing(self):
        self._realtime_processing_active = False
        self._stop_realtime_worker()
        self._stop_realtime_preview_writer()
        self._skelly_viewer_widget.stop_live()
        logger.info("Stopped real-time tracking and visualization")

    def _stop_realtime_worker(self):
        if self._realtime_worker is not None:
            if not self._realtime_worker.stop():
                logger.warning("Real-time worker did not stop within 10 seconds")
                return
            self._realtime_worker.deleteLater()
            self._realtime_worker = None

    def _start_realtime_preview_writer(self):
        self._stop_realtime_preview_writer()
        active_recording = self._active_recording_info_widget.get_active_recording_info()
        if active_recording is None:
            logger.warning("Cannot save real-time preview because there is no active recording")
            return
        output_folder = Path(active_recording.realtime_preview_folder_path)
        output_path = output_folder / f"{active_recording.name}_realtime_preview.mp4"
        self._realtime_preview_writer = RealtimePreviewWriter(output_path=output_path, parent=self)
        self._realtime_preview_writer.preview_saved_signal.connect(
            lambda path: logger.info(f"Real-time preview saved: {path}")
        )
        self._realtime_preview_writer.writing_error_signal.connect(
            lambda error: logger.error(f"Real-time preview could not be saved: {error}")
        )
        self._realtime_preview_writer.start()

    def _write_realtime_preview_frame(
        self,
        image,
        raw_pose_xyz,
        one_euro_pose_xyz,
        kalman_pose_xyz,
        kalman_one_euro_pose_xyz,
        filtered_plot_image,
        filtered_plot_dpi,
    ):
        if self._realtime_preview_writer is not None:
            self._realtime_preview_writer.submit_frame(
                image,
                raw_pose_xyz,
                one_euro_pose_xyz,
                kalman_pose_xyz,
                kalman_one_euro_pose_xyz,
                filtered_plot_image,
                filtered_plot_dpi,
            )

    def _stop_realtime_preview_writer(self):
        if self._realtime_preview_writer is None:
            return
        if not self._realtime_preview_writer.stop():
            logger.warning("Real-time preview writer did not stop within 15 seconds")
            return
        self._realtime_preview_writer.deleteLater()
        self._realtime_preview_writer = None

    @Slot(str, object)
    def _handle_live_camera_image(self, camera_id: str, image):
        if not self._realtime_processing_active or self._realtime_worker is None:
            return
        # The live preview uses the first active camera. All cameras continue to
        # be recorded by SkellyCam for the normal synchronized offline pipeline.
        if self._realtime_camera_id is None:
            self._realtime_camera_id = camera_id
        if camera_id == self._realtime_camera_id:
            self._realtime_worker.submit_frame(image)

    def update(self):
        super().update()

        try:
            if not self._skellycam_widget.is_recording:
                self._controller_group_box.update_recording_name_string()
        except Exception as e:
            logger.exception(e)

    def _set_up_stylesheet(self):
        apply_css_style_sheet(self, get_css_stylesheet_path())
        SCSSFileWatcher(
            path_to_scss_file=get_scss_stylesheet_path(), path_to_css_file=get_css_stylesheet_path(), parent=self
        )
        css_file_watcher = CSSFileWatcher(path_to_css_file=get_css_stylesheet_path(), parent=self)
        return css_file_watcher

    def _create_new_synchronized_videos_folder(self) -> str:
        new_recording_folder_path = self._controller_group_box.get_new_recording_path()
        logger.info(f"Creating new recording folder at: {new_recording_folder_path}")
        self._active_recording_info_widget.set_active_recording(recording_folder_path=new_recording_folder_path)
        return self._active_recording_info_widget.active_recording_info.synchronized_videos_folder_path

    def _create_central_tab_widget(self):
        self._home_widget = HomeWidget(actions=self._actions, gui_state=self._gui_state, parent=self)

        self._skellycam_widget = LiveSkellyCamWidget(
            self._create_new_synchronized_videos_folder,
            parent=self,
        )
        self._skellycam_widget.videos_saved_to_this_folder_signal.connect(
            self._handle_videos_saved_to_this_folder_signal
        )

        self._controller_group_box = CameraControllerGroupBox(
            skellycam_widget=self._skellycam_widget, gui_state=self._gui_state, parent=self
        )

        self._skelly_viewer_widget = RealtimeDataViewer(parent=self)

        center_tab_widget = CentralTabWidget(
            parent=self,
            skelly_cam_widget=self._skellycam_widget,
            camera_controller_widget=self._controller_group_box,
            welcome_to_freemocap_widget=self._home_widget,
            skelly_viewer_widget=self._skelly_viewer_widget,
            directory_view_widget=self._directory_view_widget,
            active_recording_info_widget=self._active_recording_info_widget,
        )

        center_tab_widget.set_welcome_tab_enabled(True)
        center_tab_widget.set_camera_view_tab_enabled(True)
        center_tab_widget.set_visualize_data_tab_enabled(True)

        self._skellycam_widget.live_image_signal.connect(self._handle_live_camera_image)
        self._skelly_viewer_widget.composite_frame_ready_signal.connect(self._write_realtime_preview_frame)
        self._controller_group_box.motion_capture_recording_started.connect(self._start_realtime_processing)
        self._controller_group_box.recording_stopped.connect(self._stop_realtime_processing)

        return center_tab_widget

    def _create_directory_view_widget(self):
        return DirectoryViewWidget(
            gui_state=self._gui_state,
            get_active_recording_info_callable=self._active_recording_info_widget.get_active_recording_info,
        )

    def _create_control_panel_widget(self, log_update: Callable):
        self._camera_configuration_parameter_tree_widget = SkellyCamParameterTreeWidget(self._skellycam_widget)

        self._process_motion_capture_data_panel = ProcessMotionCaptureDataPanel(
            recording_processing_parameters=ProcessingParameterModel(),
            get_active_recording_info=self._active_recording_info_widget.get_active_recording_info,
            gui_state=self._gui_state,
            kill_thread_event=self._kill_thread_event,
            log_update=log_update,
        )
        self._process_motion_capture_data_panel.processing_finished_signal.connect(
            self._handle_processing_finished_signal
        )

        self._visualization_control_panel = VisualizationControlPanel(parent=self, gui_state=self._gui_state)
        self._visualization_control_panel.export_to_blender_button.clicked.connect(
            self._export_active_recording_to_blender
        )

        self._visualization_control_panel.generate_jupyter_notebook_button.clicked.connect(
            self._generate_jupyter_notebook
        )

        return ControlPanelWidget(
            camera_configuration_parameter_tree_widget=self._camera_configuration_parameter_tree_widget,
            process_motion_capture_data_panel=self._process_motion_capture_data_panel,
            visualize_data_widget=self._visualization_control_panel,
            parent=self,
        )

    def _export_active_recording_to_blender(self):
        logger.debug("Exporting active recording to Blender...")
        recording_path = self._active_recording_info_widget.get_active_recording_path()

        if self._visualization_control_panel.blender_executable_path is None:
            logger.error("Blender executable path is None!")
            return

        if not recording_path:
            logger.error("Recording path is None!")
            return

        self._export_to_blender_thread_worker = ExportToBlenderThreadWorker(
            recording_path=recording_path,
            blender_file_path=Path(get_blender_file_path(recording_path)),
            blender_executable_path=Path(self._visualization_control_panel.blender_executable_path),
            kill_thread_event=self._kill_thread_event,
        )
        self._export_to_blender_thread_worker.start()
        self._export_to_blender_thread_worker.success.connect(self._handle_export_to_blender_finished)

    @Slot()
    def _handle_export_to_blender_finished(self, success_value: bool) -> None:
        if success_value is False:
            logger.error("Blender export failed!")
        elif self._controller_group_box.auto_open_in_blender_checked:
            if Path(self._active_recording_info_widget.active_recording_info.blender_file_path).exists():
                open_file(self._active_recording_info_widget.active_recording_info.blender_file_path)
            else:
                logger.error(
                    "Blender file does not exist! Did something go wrong in the `export_to_blender` call above?"
                )

    def _generate_jupyter_notebook(self):
        logger.info("Exporting active recording to a Jupyter notebook...")
        recording_path = self._active_recording_info_widget.get_active_recording_path()
        # TODO: Need to include jupyter notebook in recording files that we keep track of (2023-05-15)
        if recording_path:
            generate_jupyter_notebook(path_to_recording=recording_path)

    def _handle_new_active_recording_selected(self, recording_info_model: RecordingInfoModel):
        logger.info(f"New active recording selected: {recording_info_model.path}")

        self._pipedream_pings.update_pings_dict(
            key=recording_info_model.name, value=self._active_recording_info_widget.active_recording_info.status_check
        )

        path = Path(recording_info_model.path)
        path.mkdir(parents=True, exist_ok=True)

        if str(path.parent) != get_recording_session_folder_path():
            self._directory_view_widget.set_folder_as_root(path.parent)
        else:
            self._directory_view_widget.set_folder_as_root(path)

        if Path(recording_info_model.synchronized_videos_folder_path).exists():
            self._directory_view_widget.expand_directory_to_path(recording_info_model.synchronized_videos_folder_path)
        else:
            self._directory_view_widget.expand_directory_to_path(recording_info_model.path)

        self._active_recording_info_widget.update_parameter_tree()
        # self._recording_name_label.setText(f"Recording Name: {recording_info_model.name}")
        self._update_skelly_viewer_widget()
        self._directory_view_widget.handle_new_active_recording_selected()

        try:
            self._process_motion_capture_data_panel.update_calibration_path()
        except (
                AttributeError
        ):  # Active Recording and Data Panel widgets rely on each other, so we're guaranteed to hit this every time the app opens
            logger.debug("Process motion capture data panel not created yet, skipping claibraiton setting")
        except Exception as e:
            logger.error(e)

        update_most_recent_recording_toml(recording_info_model=recording_info_model)

    def _update_skelly_viewer_widget(self):
        active_recording_info = self._active_recording_info_widget.active_recording_info

        if active_recording_info.data3d_status_check:
            self._skelly_viewer_widget.load_skeleton_data(
                mediapipe_skeleton_npy_path=active_recording_info.data_3d_npy_file_path
            )

        if active_recording_info.data2d_status_check:
            self._skelly_viewer_widget.generate_video_display(
                video_folder_path=active_recording_info.annotated_videos_folder_path
            )

        elif active_recording_info.synchronized_videos_status_check:
            self._skelly_viewer_widget.generate_video_display(
                video_folder_path=active_recording_info.synchronized_videos_folder_path
            )

    def kill_running_threads_and_processes(self):
        logger.info("Killing running threads and processes... ")
        self._stop_imported_video_processing()
        self._realtime_processing_active = False
        self._stop_realtime_worker()
        self._stop_realtime_preview_writer()
        try:
            self._skellycam_widget.close()
        except Exception as e:
            logger.error(f"Error killing running threads and processes: {e}")

        self._kill_thread_event.set()

    def handle_load_most_recent_recording(self):
        logger.info("`Load Most Recent Recording` QAction triggered")
        most_recent_recording_path = get_most_recent_recording_path()

        if most_recent_recording_path is None:
            logger.error("`get_most_recent_recording_path()` return `None`!")
            return

        self._active_recording_info_widget.set_active_recording(recording_folder_path=get_most_recent_recording_path())
        self._central_tab_widget.setCurrentIndex(2)
        self._control_panel_widget.tab_widget.setCurrentWidget(self._process_motion_capture_data_panel)

    def open_load_existing_recording_dialog(self):
        # from this tutorial - https://www.youtube.com/watch?v=gg5TepTc2Jg&t=649s
        logger.info("Opening `Load Existing Recording` dialog... ")
        user_selected_directory = QFileDialog.getExistingDirectory(
            self,
            "Select a recording folder",
            str(get_recording_session_folder_path()),
        )
        if len(user_selected_directory) == 0:
            logger.info("User cancelled `Load Existing Recording` dialog")
            return

        logger.info(f"User selected recording path:{user_selected_directory}")

        self._active_recording_info_widget.set_active_recording(recording_folder_path=user_selected_directory)
        self._central_tab_widget.setCurrentIndex(2)

    def reset_to_default_gui_settings(self):
        self._gui_state = GuiState()

        self._home_widget._send_pings_checkbox.setChecked(self._gui_state.send_user_pings)
        self._controller_group_box._auto_process_videos_checkbox.setChecked(self._gui_state.auto_process_videos_on_save)
        self._controller_group_box._generate_jupyter_notebook_checkbox.setChecked(
            self._gui_state.generate_jupyter_notebook
        )
        self._controller_group_box._auto_open_in_blender_checkbox.setChecked(self._gui_state.auto_open_in_blender)
        self._controller_group_box._charuco_square_size_line_edit.setText(str(self._gui_state.charuco_square_size))
        self._process_motion_capture_data_panel._calibration_control_panel._charuco_square_size_line_edit.setText(
            str(self._gui_state.charuco_square_size)
        )
        self._visualization_control_panel._blender_executable_label.setText(str(self._gui_state.blender_path))
        self._visualization_control_panel._blender_executable_path = str(self._gui_state.blender_path)

        save_gui_state(self._gui_state)

        self._active_recording_info_widget.set_active_recording(recording_folder_path=get_most_recent_recording_path())

    def open_import_videos_dialog(self):
        # from this tutorial - https://www.youtube.com/watch?v=gg5TepTc2Jg&t=649s
        logger.info("Opening `Import Videos` dialog... ")

        import_videos_path = QFileDialog.getExistingDirectory(
            self,
            "Select a folder containing synchronized videos (each video must have *exactly* the same number of frames)",
            str(Path.home()),
        )

        if len(import_videos_path) == 0:
            logger.info("User cancelled `Import Videos` dialog")
            return

        self._import_videos_window = ImportVideosWizard(
            parent=self,
            import_videos_path=import_videos_path,
            kill_thread_event=self._kill_thread_event,
        )
        self._import_videos_window.folder_to_save_videos_to_selected.connect(self._handle_import_videos)
        self._import_videos_window.exec()

    def open_welcome_screen_dialog(self):
        logger.info("Opening `Welcome to Freemocap` dialog... ")

        self._welcome_screen_dialog = WelcomeScreenDialog(
            gui_state=self._gui_state, kill_thread_event=self._kill_thread_event, parent=self
        )

        self._welcome_screen_dialog.exec()

    def open_release_notes_popup(self):
        logger.info("Opening `Release Notes` dialog... ")

        self._gui_state.shown_latest_release_notes = True

        save_gui_state(gui_state=self._gui_state, file_pathstring=get_gui_state_json_path())

        dialog = TabbedReleaseNotesDialog(
            kill_thread_event=threading.Event(),
            gui_state=self._gui_state,
            dark_mode=True,  # Set to True for dark mode, False for light mode
        )
        dialog.exec()

    def open_opencv_conflict_dialog(self):
        self._opencv_conflict_dialog = OpencvConflictDialog(
            gui_state=self._gui_state, kill_thread_event=self._kill_thread_event, parent=self
        )

        self._opencv_conflict_dialog.exec()

    def open_settings_dialog(self):
        self._settings_dialog = SetDataFolderDialog(
            gui_state=self._gui_state, kill_thread_event=self._kill_thread_event, parent=self
        )

        self._settings_dialog.exec()

        if self._settings_dialog.result():
            self.reboot_gui()

    def download_data(self, dataset_name: str):
        logger.info("Downloading sample data")
        self.download_data_thread_worker = DownloadDataThreadWorker(dataset_name=dataset_name)
        self.download_data_thread_worker.start()
        self.download_data_thread_worker.finished.connect(self._handle_download_data_finished)

    @Slot(str)
    def _handle_download_data_finished(self, downloaded_data_path: str):
        logger.info("Setting downloaded data as active recording... ")
        self._active_recording_info_widget.set_active_recording(recording_folder_path=downloaded_data_path)

    @Slot(list, str, bool)
    def _handle_import_videos(self, video_paths: List[str], folder_to_save_videos: str, synchronization_bool: bool):
        folder_to_save_videos = Path(folder_to_save_videos)
        folder_to_save_videos.mkdir(parents=True, exist_ok=True)

        if len(video_paths) == 0:
            logger.error("No videos to import!")
            return

        imported_video_paths = []
        if not synchronization_bool:
            for video_path in tqdm(
                    video_paths,
                    desc="Importing videos...",
                    colour=[255, 128, 0],
                    unit="video",
                    unit_scale=True,
                    leave=False,
            ):
                if not Path(video_path).exists():
                    logger.error(f"{video_path} does not exist!")
                    return

                destination_path = folder_to_save_videos / Path(video_path).name
                logger.info(f"Copying video from {video_path} to {destination_path}")

                shutil.copy(video_path, destination_path)
                imported_video_paths.append(destination_path)
        else:
            imported_video_paths = [Path(video_path) for video_path in video_paths]

        timestamps_copied = copy_directory_if_contains_timestamps(
            source_dir=Path(video_paths[0]).parent, destination_dir=folder_to_save_videos
        )

        if timestamps_copied:
            logger.info(f"Copied timestamps from {Path(video_paths[0]).parent} to {folder_to_save_videos}")
        else:
            logger.info(f"No timestamps found in {Path(video_paths[0]).parent}")

        self._active_recording_info_widget.set_active_recording(
            recording_folder_path=Path(folder_to_save_videos).parent
        )
        logger.info(
            "Imported recording saved using the standard recording structure at: %s",
            Path(folder_to_save_videos).parent,
        )
        self._start_imported_video_realtime_processing(imported_video_paths[0])

    def _connect_calibration_updates(self):
        self._process_motion_capture_data_panel._calibration_control_panel.control_panel_calibration_updated.connect(
            self._controller_group_box.charuco_option_updated
        )
        self._controller_group_box.controller_group_box_calibration_updated.connect(
            self._process_motion_capture_data_panel._calibration_control_panel.charuco_option_updated
        )

    def reboot_gui(self):
        logger.info("Rebooting GUI... ")
        get_qt_app().exit(EXIT_CODE_REBOOT)

    def closeEvent(self, a0) -> None:
        logger.info("Main window `closeEvent` detected")
        self._stop_imported_video_processing()

        if self._home_widget.consent_to_send_usage_information:
            self._pipedream_pings.update_pings_dict(key="gui_closed", value=True)
            if self._active_recording_info_widget.active_recording_info is not None:
                self._pipedream_pings.update_pings_dict(
                    key="active recording status on close",
                    value=self._active_recording_info_widget.active_recording_info.status_check,
                )
            self._pipedream_pings.send_pipedream_ping()

        try:
            remove_empty_directories(get_recording_session_folder_path())
        except Exception as e:
            logger.error(f"Error while removing empty directories: {e}")

        try:
            self._skellycam_widget.close()
        except Exception as e:
            logger.error(f"Error while closing the viewer widget: {e}")
        super().closeEvent(a0)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    main_window = MainWindow(pipedream_pings=PipedreamPings())
    main_window.show()
    app.exec()
    for process in multiprocessing.active_children():
        logger.info(f"Terminating process: {process}")
        process.terminate()
    sys.exit()
