# High-Scale Talking Presenter Video Processing Pipeline

An end-to-end, high-throughput data processing and filtering pipeline designed to prepare high-quality, standardized 4-second (100-frame @ 25 FPS) talking-presenter video clips from raw in-the-wild YouTube footage for motion synthesis and audio-driven generative modeling.

---

## ⚙️ System Prerequisites & Dependencies

* **System Binaries:** `ffmpeg` and `ffprobe` must be installed on the system and accessible via `$PATH`.
* **Hardware:** A CUDA-compatible GPU is strongly recommended for Stages 1 and 2 (YOLO pose inference & batch video auditing).
* **Python Environment:** Python 3.10+ with `torch`, `torchvision`, `ultralytics`, `duckdb`, `pyarrow`, `pandas`, `opencv-python`, and `pytube`.

---

## 🚀 Pipeline Architecture Overview

The pipeline is organized into modular, deterministic stages designed to scale across HPC clusters with SLURM job scheduling and Parquet-based metadata management:

```
[Raw YouTube IDs List (~20k IDs) - Alireza Javanmardi / process-talkingpose]
          │
          ▼ (1. Ingestion: pytube_download_v2.py @ 1080p -> 720p -> >=720p)
[Raw In-the-Wild Source Videos]
          │
          ▼ ─── Stage 0: AV Standardization (25 FPS CFR, 16kHz Mono Audio)
[Standardized Source Videos (25 FPS, 16kHz Audio)]
          │
          ▼ ─── Stage 1.1: YOLO Pose Tracking & Temporal Segmentation
[Stage 1 Manifest (stage1_manifest.parquet)]
          │
          ▼ ─── Stage 1.2: Dynamic Crop Geometry & Arm-Reach Optimization
[Consolidated Stage 1 Metadata (crop_geometry.parquet & consolidated manifest)]
          │
          ▼ ─── Stage 2.1: 100-Frame Window Slicing & Speaker-Aware Split
[Render Manifest (render_manifest.parquet - Train / Val / Test Assignment)]
          │
          ▼ ─── Stage 2.2: Distributed Rendering & 512x512 Normalization
[Final Rendered Dataset: 100-frame Clips @ 512x512 + Synchronized 16kHz Audio]

```

---

## 📦 Stage-by-Stage Breakdown (Inputs, Outputs & Operations)

### 1. Data Ingestion & Download (`pytube_download_v2.py`)

* **Inputs:**
* CSV/Text file of raw YouTube video IDs (curated list by **Alireza Javanmardi** — [GitHub: ajavanmardii](https://github.com/ajavanmardii) / [DFKI Gitlab: process-talkingpose](https://git.opendfki.de/alireza.javanmardi/process-talkingpose)).
* Output directory path for raw videos.


* **Execution Command:**

```bash
python pytube_download_v2.py /path/to/ids.csv /path/to/raw_videos

```

* **Operations:**
* Prioritized stream downloading: **1080p** $\rightarrow$ **720p** $\rightarrow$ **highest available $\ge$ 720p**.
* Integrity validation to drop private/unavailable streams and corrupted video containers.


* **Outputs:**
* High-resolution raw MP4 video files in arbitrary native frame rates and audio configurations.



---

### Stage 0: Audio-Visual Normalization

* **Inputs:**
* Directory containing raw downloaded videos (`/path/to/raw_videos/*.mp4`).


* **Operations:**
* Constant Frame Rate conversion (**CFR 25.0 FPS**) to resolve variable frame rate (VFR) drift.
* Audio stream extraction and resampling to **16 kHz mono (single-channel)**.


* **Outputs:**
* Standardized source videos ready for continuous keypoint tracking and frame-accurate seeking.



---

### Stage 1.1: Presenter Tracking & Temporal Segmentation

* **Inputs:**
* Standardized source videos from Stage 0.
* YOLOv8-Pose model checkpoint (`yolov8s-pose.pt`).


* **Operations:**
* Runs presenter pose estimation and tracks body landmarks across continuous video streams.
* Detects uninterrupted temporal segments featuring a single, active, visible presenter.
* Discards shot cuts, multi-person frames, severe occlusions, and sudden presenter dropouts.


* **Outputs:**
* `stage1_manifest.parquet`: Tabular metadata listing valid segment start/end frames, keypoint coordinate summaries, and speaker cluster identifiers (`speaker_cluster_id`).



---

### Stage 1.2: Dynamic Crop Geometry & Arm-Reach Optimization

* **Inputs:**
* `stage1_manifest.parquet` from Stage 1.1.
* Extracted keypoint tracks for upper-body joints (head, shoulders, elbows, wrists).


* **Operations:**
* Computes stable, square bounding boxes `(bx, by, b_side)` around the presenter's upper body.
* Expands and centers the crop box according to maximal wrist extensions and arm movements to prevent hand gestures from clipping out of the frame.
* Consolidates bounding records with the main segment manifest using DuckDB.


* **Outputs:**
* `crop_geometry.parquet` & `stage1_manifest_consolidated.parquet`: Consolidated manifest with spatial crop parameters ready for rendering.



---

### Stage 2.1: 100-Frame Window Slicing & Speaker-Aware Dataset Splitting

* **Inputs:**
* `stage1_manifest_consolidated.parquet`.
* Window configuration parameters (Length: 100 frames / 4.0 seconds, Stride: 100 frames).


* **Operations:**
* Slices continuous presenter segments into exact non-overlapping 100-frame training windows.
* Assigns deterministic data partitions (**Train 85% / Val 10% / Test 5%**) using cryptographic hashing on `speaker_cluster_id` to strictly prevent identity leakage across splits.


* **Outputs:**
* `render_manifest.parquet`: Full render table containing global frame offsets, audio seek timestamps, exact crop coordinates, and `split` assignments for every 100-frame sample.



---

### Stage 2.2: Distributed Rendering & 512×512 Normalization

* **Inputs:**
* `render_manifest.parquet`.
* Standardized Stage 0 source video files.


* **Operations:**
* High-throughput distributed FFmpeg rendering executing spatial cropping `(crop=b_side:b_side:bx:by)` and Lanczos resampling to $512 \times 512$ square resolution.
* Synchronous extraction of the corresponding 4.0-second 16 kHz AAC audio stream.
* Sharded file placement across 2-character prefix hash directories to maintain fast filesystem I/O on HPC clusters.


* **Outputs:**
* Final dataset of standardized MP4 video clips (`rendered/<split>/<prefix>/<window_uid>/video.mp4`): 512×512 @ 25 FPS with embedded 16 kHz mono audio.



---

## 🛠️ Quality Assurance & Auditing

* **Inputs:** Rendered video clips directory (`/netscratch/bahrami/dataset/stage2_clips`).
* **Tool:** `audit_exact_10k_stride1.py`
* **Operations:** Evaluates all 100 frames per clip on a random sample of 10,000 clips using batch GPU inference with YOLO to identify presenter presence, empty frames, and dropout ratios.
* **Outputs:** `exact_audit_10k_results.csv` and detailed dataset distribution metrics.

---

## 💻 Tech Stack

* **Core:** Python 3.10+, PyTorch, Torchvision, Ultralytics YOLO
* **Media Processing:** FFmpeg, FFprobe, OpenCV, PyTube
* **Data & Metadata:** DuckDB, Apache Arrow / PyArrow, Pandas
* **Infrastructure:** SLURM HPC Cluster, SquashFS Containerization
