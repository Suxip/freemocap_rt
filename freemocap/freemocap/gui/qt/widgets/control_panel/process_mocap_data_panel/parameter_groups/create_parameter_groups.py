from pyqtgraph.parametertree import Parameter
from skellytracker.trackers.mediapipe_tracker.mediapipe_model_info import (
    MediapipeTrackingParams,
)

from freemocap.data_layer.recording_models.post_processing_parameter_models import (
    ProcessingParameterModel,
    AniposeTriangulate3DParametersModel,
    PostProcessingParametersModel,
    KalmanFilterParametersModel,
)

KALMAN_PROCESS_NOISE = "Kalman Process Noise"
KALMAN_MEASUREMENT_NOISE = "Kalman Measurement Noise"
MAX_GAP_TO_FILL = "Maximum Gap to Fill"

POST_PROCESSING_FRAME_RATE = "Framerate"

CAUSAL_POST_PROCESSING_TREE_NAME = "Causal Post Processing"

OUTLIER_REJECTION_TREE_NAME = "Outlier Rejection"

USE_OUTLIER_REJECTION_METHOD = "Use Outlier Rejection Method?"

OUTLIER_REJECTION_MAXIMUM_CAMERAS_TO_DROP = "Maximum Cameras to Drop"

OUTLIER_REJECTION_MINIMUM_CAMERAS_FOR_TRIANGULATION = "Minimum Cameras for Triangulation"

OUTLIER_REJECTION_TARGET_REPROJECTION_ERROR = "Target Reprojection Error"

ANIPOSE_CONFIDENCE_CUTOFF = "Confidence Threshold Cut-off"

FLATTEN_SINGLE_CAMERA_DATA = "Flatten Single Camera Data (Recommended)"

ANIPOSE_TREE_NAME = "Anipose Triangulation"

YOLO_CROP_TREE_NAME = "YOLO Crop"

USE_YOLO_CROP_METHOD = "Use YOLO Crop Method"

YOLO_MODEL_SIZE = "YOLO Model Size"

BOUNDING_BOX_BUFFER_METHOD = "Buffer Bounding Box:"

BOUNDING_BOX_BUFFER_PERCENTAGE = "Bounding Box Buffer Percentage"

STATIC_IMAGE_MODE = "Static Image Mode"

MINIUMUM_TRACKING_CONFIDENCE = "Minimum Tracking Confidence"

MINIMUM_DETECTION_CONFIDENCE = "Minimum Detection Confidence"

MEDIAPIPE_MODEL_COMPLEXITY = "Model Complexity"

MEDIAPIPE_TREE_NAME = "Mediapipe"

RUN_IMAGE_TRACKING_NAME = "Run 2d image tracking?"

RUN_3D_TRIANGULATION_NAME = "Run 3d triangulation?"

NUMBER_OF_PROCESSES_PARAMETER_NAME = "Max Number of Processes to Use"


# TODO: figure out how to generalize this
def create_mediapipe_parameter_group(
        parameter_model: MediapipeTrackingParams,
) -> Parameter:
    mediapipe_model_complexity_list = [
        "0 (Fastest/Least accurate)",
        "1 (Middle ground)",
        "2 (Slowest/Most accurate)",
    ]
    return Parameter.create(
        name=MEDIAPIPE_TREE_NAME,
        type="group",
        expanded=False, # collapsed by default

        children=[

            dict(
                name=MEDIAPIPE_MODEL_COMPLEXITY,
                type="list",
                limits=mediapipe_model_complexity_list,
                value=mediapipe_model_complexity_list[parameter_model.mediapipe_model_complexity],
                tip="Which Mediapipe model to use - higher complexity is slower but more accurate. "
                    "Variable name in `mediapipe` code: `mediapipe_model_complexity`",
            ),
            dict(
                name=MINIMUM_DETECTION_CONFIDENCE,
                type="float",
                value=parameter_model.min_detection_confidence,
                step=0.05,
                limits=(0.0, 1.0),
                tip="Minimum confidence for a skeleton detection to be considered valid. "
                    "Variable name in `mediapipe` code: `min_detection_confidence`."
                    "NOTE - Never trust a machine learning model's estimates of their own confidence!",
            ),
            dict(
                name=MINIUMUM_TRACKING_CONFIDENCE,
                type="float",
                value=parameter_model.min_tracking_confidence,
                step=0.05,
                limits=(0.0, 1.0),
                tip="Minimum confidence needed to use the previous frame's skeleton estiamte to predict the next one"
                    "Variable name in `mediapipe` code: `min_tracking_confidence`.",
            ),
            dict(
                name=STATIC_IMAGE_MODE,
                type="bool",
                value=parameter_model.static_image_mode,
                tip="If true, the model will process each image independently, without tracking across frames."
                    "I think this is equivalent to setting `min_tracking_confidence` to 0.0"
                    "Variable name in `mediapipe` code: `static_image_mode`",
            ),
            dict(
                name=YOLO_CROP_TREE_NAME,
                type="group",
                expanded=False, # collapsed by default
                children=[
                    dict(
                        name=USE_YOLO_CROP_METHOD,
                        type="bool",
                        value=parameter_model.use_yolo_crop_method,
                        tip="If true, `skellytracker` will use YOLO to pre-crop the person from the image before running the `mediapipe` tracker",
                    ),
                    dict(
                        name=YOLO_MODEL_SIZE,
                        type="list",
                        limits=["nano", "small", "medium", "large", "extra_large", "high_res"],
                        value=parameter_model.yolo_model_size,
                        tip="Smaller models are faster but may be less accurate",
                    ),
                    dict(
                        name=BOUNDING_BOX_BUFFER_METHOD,
                        type="list",
                        limits=["By box size", "By image size"],
                        value=parameter_model.buffer_size_method,
                        tip="Buffer bounding box by percentage of either box size or image size",
                    ),
                    dict(
                        name=BOUNDING_BOX_BUFFER_PERCENTAGE,
                        type="int",
                        value=parameter_model.bounding_box_buffer_percentage,
                        limits=(0, 100),
                        step=1,
                        tip="Percentage to increase size of bounding box",
                    ),
                ],
            ),
        ],
    )


# def create_3d_triangulation_parameter_group(
#         parameter_model: AniposeTriangulate3DParametersModel = None,
# ) -> Parameter:
#     if parameter_model is None:
#         parameter_model = AniposeTriangulate3DParametersModel()
#
#     return Parameter.create(
#         name=ANIPOSE_TREE_NAME,
#         type="group",
#         children=[
#
#             dict(
#                 name=FLATTEN_SINGLE_CAMERA_DATA,
#                 type="bool",
#                 value=parameter_model.flatten_single_camera_data,
#                 tip="If true, flatten the data from single camera recordings.",
#             ),
#             dict(
#                 name=OUTLIER_REJECTION_TREE_NAME,
#                 type="group",
#                 children=[
#                     dict(
#                         name=USE_OUTLIER_REJECTION_METHOD,
#                         type="bool",
#                         value=parameter_model.use_triangulate_outlier_rejection,
#                         tip="If true, use `anipose`'s `triangulate_using_outlier_rejection` method.",
#                     ),
#                     dict(
#                         name=OUTLIER_REJECTION_MAXIMUM_CAMERAS_TO_DROP,
#                         type="int",
#                         value=parameter_model.maximum_cameras_to_drop,
#                         limits=(0, 100),
#                         step=1,
#                         tip="Maximum amount of cameras permitted to drop.",
#                     ),
#                     dict(
#                         name=OUTLIER_REJECTION_TARGET_REPROJECTION_ERROR,
#                         type="float",
#                         value=parameter_model.target_reprojection_error,
#                         limits=(0.0, 1.0),
#                         step=0.001,
#                         tip="The target reprojection error that stops the outlier rejection search.\n"
#                             "If a camera combination achieves an error below this value, it is accepted and further dropped-camera iterations are skipped.",
#                     ),
#                 ],
#             ),
#         ],
#     )


def create_post_processing_parameter_group(
        parameter_model: PostProcessingParametersModel = None,
) -> Parameter:
    if parameter_model is None:
        parameter_model = PostProcessingParametersModel()

    return Parameter.create(
        name=CAUSAL_POST_PROCESSING_TREE_NAME,
        type="group",
        expanded=False, # collapsed by default
        children=[
            dict(
                name=POST_PROCESSING_FRAME_RATE,
                type="float",
                value=parameter_model.framerate,
                tip="Framerate of the recording " "TODO - Calculate this from the recorded timestamps....",
            ),
            dict(
                name=KALMAN_PROCESS_NOISE,
                type="float",
                value=parameter_model.kalman_filter_parameters.process_noise,
                tip="Higher values allow the Kalman motion model to change faster.",
            ),
            dict(
                name=KALMAN_MEASUREMENT_NOISE,
                type="float",
                value=parameter_model.kalman_filter_parameters.measurement_noise,
                tip="Higher values place less trust in observed marker positions.",
            ),
            dict(
                name=MAX_GAP_TO_FILL,
                type="int",
                value=parameter_model.max_gap_to_fill,
                limits=(0, 10000),
                tip="Maximum consecutive missing frames predicted by the causal Kalman filter.",
            ),
        ],
        tip="Forward-only Kalman gap filling; no future frames are used.",
    )


def extract_parameter_model_from_parameter_tree(
        parameter_object: Parameter,
) -> ProcessingParameterModel:
    parameter_values_dictionary = extract_processing_parameter_model_from_tree(parameter_object=parameter_object)

    return ProcessingParameterModel(
        tracking_parameters_model=MediapipeTrackingParams(
            mediapipe_model_complexity=get_integer_from_mediapipe_model_complexity(
                parameter_values_dictionary[MEDIAPIPE_MODEL_COMPLEXITY]
            ),
            min_detection_confidence=parameter_values_dictionary[MINIMUM_DETECTION_CONFIDENCE],
            min_tracking_confidence=parameter_values_dictionary[MINIUMUM_TRACKING_CONFIDENCE],
            static_image_mode=parameter_values_dictionary[STATIC_IMAGE_MODE],
            run_image_tracking=parameter_values_dictionary[RUN_IMAGE_TRACKING_NAME],
            num_processes=parameter_values_dictionary[NUMBER_OF_PROCESSES_PARAMETER_NAME],
            use_yolo_crop_method=parameter_values_dictionary[USE_YOLO_CROP_METHOD],
            yolo_model_size=parameter_values_dictionary[YOLO_MODEL_SIZE],
            buffer_size_method=get_bounding_box_buffer_method_from_string(
                parameter_values_dictionary[BOUNDING_BOX_BUFFER_METHOD]
            ),
            bounding_box_buffer_percentage=parameter_values_dictionary[BOUNDING_BOX_BUFFER_PERCENTAGE],
        ),
        anipose_triangulate_3d_parameters_model=AniposeTriangulate3DParametersModel(
            flatten_single_camera_data=parameter_values_dictionary[FLATTEN_SINGLE_CAMERA_DATA],
            use_triangulate_outlier_rejection=parameter_values_dictionary[USE_OUTLIER_REJECTION_METHOD],
            minimum_cameras_for_triangulation=parameter_values_dictionary[OUTLIER_REJECTION_MINIMUM_CAMERAS_FOR_TRIANGULATION],
            maximum_cameras_to_drop=parameter_values_dictionary[OUTLIER_REJECTION_MAXIMUM_CAMERAS_TO_DROP],
            target_reprojection_error=parameter_values_dictionary[OUTLIER_REJECTION_TARGET_REPROJECTION_ERROR],
            run_3d_triangulation=parameter_values_dictionary[RUN_3D_TRIANGULATION_NAME],
        ),
        post_processing_parameters_model=PostProcessingParametersModel(
            framerate=parameter_values_dictionary[POST_PROCESSING_FRAME_RATE],
            kalman_filter_parameters=KalmanFilterParametersModel(
                process_noise=parameter_values_dictionary[KALMAN_PROCESS_NOISE],
                measurement_noise=parameter_values_dictionary[KALMAN_MEASUREMENT_NOISE],
            ),
            max_gap_to_fill=parameter_values_dictionary[MAX_GAP_TO_FILL],
        ),
    )


def get_integer_from_mediapipe_model_complexity(mediapipe_model_complexity_value: str):
    mediapipe_model_complexity_dictionary = {
        "0 (Fastest/Least accurate)": 0,
        "1 (Middle ground)": 1,
        "2 (Slowest/Most accurate)": 2,
    }
    return mediapipe_model_complexity_dictionary[mediapipe_model_complexity_value]


def get_bounding_box_buffer_method_from_string(buffer_method_string: str) -> str:
    bounding_box_buffer_method_dict = {
        "By box size": "buffer_by_box_size",
        "By image size": "buffer_by_image_size",
    }
    return bounding_box_buffer_method_dict[buffer_method_string]


def extract_processing_parameter_model_from_tree(parameter_object, value_dictionary: dict = None):
    if value_dictionary is None:
        value_dictionary = {}

    for child in parameter_object.children():
        if child.hasChildren():
            extract_processing_parameter_model_from_tree(child, value_dictionary)
        else:
            value_dictionary[child.name()] = child.value()
    return value_dictionary
