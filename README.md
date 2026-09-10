# High-Scale Talking Presenter Video Processing Pipeline

An end-to-end, high-throughput data processing and filtering pipeline designed to prepare standardized 4-second (100-frame @ 25 FPS) talking-presenter video clips from in-the-wild YouTube footage for motion synthesis and audio-driven generative modeling.

---

## ⚙️ System Prerequisites & Dependencies

* **System Binaries:** `ffmpeg` and `ffprobe` must be installed and accessible via `$PATH`.
* **Hardware:** A CUDA-compatible GPU is required for Stage 1 (YOLO pose inference & tracking).
* **Python Environment:** Python 3.10+ with `torch`, `torchvision`, `ultralytics`, `duckdb`, `pyarrow`, `pandas`, `opencv-python`, and `pytube`.

---

## 🚀 Pipeline Architecture Overview

```text
[Raw YouTube IDs (~20k IDs) - Alireza Javanmardi / process-talkingpose]
          │
          ▼ (Data Ingestion: pytube_download_v2.py @ 1080p -> 720p -> >=720p)
[Raw In-the-Wild Source Videos]
          │
          ▼ ─── Stage 0: AV Normalization (25 FPS CFR, 16kHz Mono Audio)
[Standardized 25 FPS Videos]
          │
          ▼ ─── Stage 1: Presenter Detection, Keypoints & Integrated Crop Geometry
[stage1_manifest.parquet (Contains temporal bounds & integrated (bx, by, b_side))]
          │
          ▼ ─── Stage 2.1: 100-Frame Window Slicing (CPU, Metadata-Only)
[render_manifest.parquet (Train / Val / Test Hashed Splits)]
          │
          ▼ ─── Stage 2.2: Distributed AV-Muxed Rendering (FFmpeg)
[Final Rendered Clips: 100-frame MP4 @ 512x512 + 16kHz Audio]

```

---

## 📦 Stage-by-Stage Breakdown (Inputs, Outputs & Execution)

### Ingestion: Data Ingestion & Download (`pytube_download_v2.py`)

* **Description:** Downloads videos from a list of YouTube identifiers curated by **Alireza Javanmardi** ([GitHub: ajavanmardii](https://github.com/ajavanmardii) / [DFKI Gitlab: process-talkingpose](https://git.opendfki.de/alireza.javanmardi/process-talkingpose)).
* **Inputs:** CSV / text file with YouTube IDs (one per row).
* **Outputs:** Raw video files in native definitions.
* **Resolution Strategy:** Tries `1080p` $\rightarrow$ `720p` $\rightarrow$ highest available resolution $\ge$ `720p`.
* **Execution:**

```bash
python pytube_download_v2.py ids/missing_ids.csv /path/to/raw_videos

```

---

### Stage 0: Audio-Visual Normalization (`normalize_raw_videos.py`)

* **Description:** Normalizes variable frame rate streams to standard formats to eliminate synchronization drift.
* **Inputs:** Raw downloaded video directory (`/path/to/raw_videos/*.mp4`).
* **Outputs:** 25.0 FPS Constant Frame Rate (CFR) MP4 videos with 16 kHz mono audio.
* **Execution:**

```bash
python normalize_raw_videos.py \
    --input-dir /path/to/raw_videos \
    --output-dir /netscratch/bahrami/dataset/talking_pose_25fps \
    --fps 25.0 \
    --audio-sr 16000

```

---

### Stage 1: Presenter Detection, Keypoints & Integrated Crop Geometry (`stage1_detect.py`)

* **Description:** Executes YOLO-Pose presenter detection, temporal segment boundary extraction, COCO-17 keypoint tracking, and pre-computes static square body-crop geometry `(bx, by, b_side)` directly in a single integrated pass. (Metadata out, NO video out).
* **Inputs:** Standardized 25.0 FPS CFR videos from Stage 0.
* **Outputs:**
1. `stage1_manifest.parquet`: Segment metadata, boundary frame indices, tracking metrics, and pre-calculated `(bx, by, b_side)` body crops.
2. `stage1_keypoints.parquet`: Flattened `(N_det, 17)` keypoint trajectories per segment.
3. `stage1_manifest_preview.csv`: Summary for quick manual inspection.
4. `stage1_run.json`: Parameter hashes, runtime configurations, and summary statistics.


* **Execution:**

```bash
python stage1_detect.py \
    --input-dir /netscratch/bahrami/dataset/talking_pose_25fps \
    --output-dir /netscratch/bahrami/dataset/stage1_meta \
    --num-workers 6 \
    --frame-stride 4 \
    --batch-size 64 \
    --half

```

---

### Stage 2.1: Window Slicing & Speaker-Aware Splitting (`build_windows_table.py`)

* **Description:** Slices valid segments from Stage 1 into non-overlapping 100-frame temporal windows (4.0s @ 25 FPS), propagates spatial crop bounds `(bx, by, b_side)`, and assigns leak-free speaker-hashed train/val/test splits (CPU, metadata-only, no FFmpeg).
* **Inputs:** `stage1_manifest.parquet` (columns: `segment_uid`, `path`, `start_frame`, `end_frame`, `bx`, `by`, `b_side`, `speaker_cluster_id`).
* **Outputs:** `render_manifest.parquet` (columns: `window_uid`, `start_frame_abs`, `split`, `bx`, `by`, `b_side`, `path`, etc.).
* **Execution:**

```bash
python build_windows_table.py \
    --manifest /netscratch/bahrami/dataset/stage1_meta/stage1_manifest.parquet \
    --out /netscratch/bahrami/dataset/stage1_meta/render_manifest.parquet \
    --window-len 100 \
    --stride 100

```

---

### Stage 2.2: Distributed AV-Muxed Rendering (`stage2_render_single.py`)

* **Description:** Executes high-throughput video/audio clipping and spatial cropping from 25 FPS source videos into standalone 100-frame (4-second) MP4 clips with embedded audio in a single container.
* **Inputs:** `render_manifest.parquet` and Stage 0 normalized source videos.
* **Outputs:**
* Rendered clips stored as: `<out-root>/<window_uid[:2]>/<window_uid>/video.mp4` (512×512, 25 FPS, 100 frames, H.264 + 16kHz AAC mono audio).
* Status logs: `<status-dir>/part-<task_id>.parquet`.


* **Execution (Single Node / Test):**

```bash
python stage2_render_single.py \
    --manifest /netscratch/bahrami/dataset/stage1_meta/render_manifest.parquet \
    --out-root /netscratch/bahrami/dataset/stage2_clips \
    --status-dir /netscratch/bahrami/dataset/stage2_meta/status \
    --workers 16

```

* **Execution (Distributed SLURM Array):**

```bash
# Inside SLURM script (#SBATCH --array=0-99):
python stage2_render_single.py \
    --manifest /netscratch/bahrami/dataset/stage1_meta/render_manifest.parquet \
    --out-root /netscratch/bahrami/dataset/stage2_clips \
    --status-dir /netscratch/bahrami/dataset/stage2_meta/status \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --num-tasks 100 \
    --workers 16

```

---

## 💻 Tech Stack

* **Core:** Python 3.10+, PyTorch, Ultralytics YOLO
* **Media Processing:** FFmpeg, FFprobe, OpenCV, PyTube
* **Data & Metadata:** DuckDB, Apache Arrow / PyArrow, Pandas
* **Infrastructure:** SLURM HPC Cluster, SquashFS Containerization
