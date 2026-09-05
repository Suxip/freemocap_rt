# Real-Time Causal Motion Capture with FreeMoCap

An experimental extension of [FreeMoCap](https://github.com/freemocap/freemocap) for causal, real-time human pose processing and quantitative filter comparison.

This project runs MediaPipe pose estimation on live camera recordings or prerecorded videos, visualizes the result in real time, and saves synchronized numerical data for comparing raw pose estimates with One Euro and Kalman filtering.

> This repository is a research prototype built on FreeMoCap. It is not an official FreeMoCap release.

## Project goals

Traditional offline motion-capture pipelines can use future frames when interpolating missing data or smoothing trajectories. That is unsuitable for applications that must respond immediately.

This project investigates a fully causal workflow in which each output depends only on the current and previous frames. The `jitter-comparison` branch applies multiple filters independently to the same raw MediaPipe measurements so their results can be compared fairly.

## Features

- Real-time MediaPipe pose estimation from a camera
- Real-time processing of imported MP4 videos
- Causal One Euro filtering for jitter reduction
- Causal Kalman filtering for short missing-data gaps
- Independent comparison of raw MediaPipe, One Euro, Kalman, and combined Kalman–One Euro output
- Switchable One Euro and Kalman visualization in the GUI
- Three-panel saved preview with annotated video, raw pose, and the selected filtered pose
- Automatic NumPy data export and aggregate jitter reports
- FreeMoCap-compatible timestamped recording folders

## Processing architecture

```text
Live camera or imported video
              |
              v
       MediaPipe tracking
              |
              v
     Raw 33-landmark pose
       /       |        \
      /        |         \
One Euro    Kalman    Kalman -> One Euro
      \        |         /
       \       |        /
        Jitter comparison
```

All filter paths receive copies of the same raw pose sample. The standalone filters and combined pipeline maintain separate internal state, so no filter contaminates another result.

## Filter behavior

### One Euro filter

The One Euro filter adaptively smooths every valid coordinate:

- Applies stronger smoothing during slow or stationary motion
- Becomes more responsive during faster motion
- Reduces visible landmark jitter
- Does not fill coordinates that MediaPipe fails to detect

### Kalman filter

The Kalman implementation in this project is a causal gap filler:

- Preserves valid MediaPipe measurements
- Predicts coordinates during short missing-data gaps
- Uses a constant-velocity motion model
- Does not smooth ordinary jitter when measurements are present

### Combined filter

The combined path applies Kalman first and One Euro second:

```text
Raw pose -> Kalman gap filling -> One Euro smoothing
```

This provides both short-gap handling and smoothing while remaining causal.

## Jitter measurement

Jitter is measured from numerical 3D pose coordinates rather than rendered graph pixels. For each interval, the software calculates the Euclidean displacement of every landmark between consecutive frames and sums across all valid transitions and all 33 landmarks:

```text
total jitter = sum over frames and landmarks of ||pose[t] - pose[t-1]||
```

The current comparison intervals are:

- 5–9 seconds
- 11–15 seconds
- 20–24 seconds
- 54–58 seconds
- 67–71 seconds
- 79–83 seconds

The report includes one aggregate jitter value per interval for raw MediaPipe, One Euro, Kalman, and the combined filter. Valid landmark-transition counts are included because missing measurements can otherwise make a lower total misleading.

Percentage improvement is calculated as:

```text
jitter reduction (%) = ((raw jitter - filtered jitter) / raw jitter) * 100
```

## Saved output

Recordings use FreeMoCap's standard directory structure:

```text
freemocap_data/
└── recording_sessions/
    └── session_<date_time>/
        └── recording_<time>_gmt<offset>/
            ├── synchronized_videos/
            └── realtime_preview/
                ├── <recording>_realtime_preview.mp4
                ├── <recording>_pose_timestamps.npy
                ├── <recording>_raw_mediapipe_pose.npy
                ├── <recording>_one_euro_pose.npy
                ├── <recording>_kalman_pose.npy
                ├── <recording>_kalman_one_euro_pose.npy
                └── <recording>_filter_comparison_jitter_report.csv
```

Each pose array has shape:

```text
(processed_frames, 33, 3)
```

## Using the application

### Live recording

1. Launch FreeMoCap.
2. Start a new motion-capture recording.
3. Select **One Euro Filter** or **Kalman Filter** above the real-time viewer.
4. Record the movement.
5. Stop the recording and wait for the preview writer to finish.
6. Open the recording's `realtime_preview/` folder to view the video, pose arrays, and jitter report.

Both standalone filters and the combined filter continue to run for numerical comparison regardless of which graph is selected for display.

### Imported video

1. Select **Import Videos**.
2. Choose a folder containing one or more `.mp4` files.
3. Choose the recording name and synchronization options.
4. Continue the import.
5. The first imported video is replayed through the same real-time processing worker used for live recordings.

Imported videos are copied; the original source video is not deleted or moved.

## Installation from source

Python 3.10–3.12 is supported by the upstream project; Python 3.12 is recommended for this development environment.

```bash
git clone https://github.com/Suxip/freemocap_rt.git
cd freemocap_rt
python3.12 -m venv fmc_env
source fmc_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Launch the GUI:

```bash
python -m freemocap
```

The environment must be activated again in each new terminal session:

```bash
source fmc_env/bin/activate
```

## Branches

- `prototype` — causal real-time processing foundation
- `one-euro-filter` — One Euro-only experiment and jitter report
- `kalman-filter` — Kalman-only experiment and jitter report
- `jitter-comparison` — parallel raw, One Euro, Kalman, and combined comparison

## Important limitations

- The real-time worker keeps the newest waiting frame to prevent latency from accumulating, so unique processed samples may be fewer than encoded video frames.
- The preview writer may repeat rendered frames to maintain a 30 FPS output video.
- Filter timing currently assumes a 30 Hz sampling rate.
- MediaPipe coordinates are normalized image-relative coordinates, not calibrated metric distances.
- The Kalman implementation only changes missing values; valid measurements pass through unchanged.
- Imported folders may contain multiple MP4 files, but the real-time preview currently processes the first sorted video.
- A lower jitter value indicates less movement, but excessive smoothing can also increase lag. Jitter and responsiveness should therefore be evaluated together.

## Key implementation files

- `freemocap/gui/qt/workers/realtime_mocap_worker.py` — MediaPipe and parallel causal filter processing
- `freemocap/gui/qt/widgets/realtime_data_viewer.py` — live video, graph, and filter-selection controls
- `freemocap/gui/qt/workers/realtime_preview_writer.py` — preview encoding and numerical pose recording
- `freemocap/utilities/realtime_jitter_analysis.py` — aggregate jitter calculations and CSV generation
- `freemocap/core_processes/post_process_skeleton_data/causal_post_processing.py` — filter implementations

## Upstream project

FreeMoCap is a free and open-source, hardware-agnostic motion-capture platform for research, education, and training.

- [FreeMoCap repository](https://github.com/freemocap/freemocap)
- [FreeMoCap documentation](https://freemocap.github.io/documentation)
- [FreeMoCap citation](CITATION.cff)
- [Contribution guidelines](CONTRIBUTING.md)

## License

This project retains FreeMoCap's GNU Affero General Public License. See [LICENSE](LICENSE) for details.
