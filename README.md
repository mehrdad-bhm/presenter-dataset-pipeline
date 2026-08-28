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
[Raw YouTube IDs List (~18k IDs) - Alireza Javanmardi / process-talkingpose]
          │
          ▼ (1. Ingestion: pytube_download_v2.py @ 1080p -> 720p -> >=720p)
[Raw In-the-Wild Source Videos]
          │
          ▼ ─── Stage 0: AV Standardization (25 FPS CFR, 16kHz Mono Audio)
[Standardized Source Videos]
          │
          ▼ ─── Stage 1.1: YOLO Pose Tracking & Temporal Segmentation
[Stage 1 Manifest (Continuous Presenter Segments)]
          │
          ▼ ─── Stage 1.2: Dynamic Crop Geometry & Arm-Reach Optimization
[Consolidated Stage 1 Metadata (crop_geometry.parquet)]
          │
          ▼ ─── Stage 2.1: 100-Frame Window Slicing & Speaker-Aware Split
[Render Manifest (Train / Val / Test Assignment)]
          │
          ▼ ─── Stage 2.2: Distributed Rendering & 512x512 Normalization
[Final Rendered Dataset: 100-frame Clips @ 512x512 + Synchronized 16kHz Audio]

```

---

## 📦 Stage-by-Stage Breakdown

### 1. Data Ingestion & Download (`pytube_download_v2.py`)

* **Source Identifiers:** Ingested YouTube video identifiers (~20k unique IDs) curated by **Alireza Javanmardi** ([GitHub: ajavanmardii](https://github.com/ajavanmardii) / [DFKI Gitlab: process-talkingpose](https://git.opendfki.de/alireza.javanmardi/process-talkingpose)).
* **Download Command:**

```bash
python pytube_download_v2.py ids/missing_ids.csv /path/to/raw_videos

```

* **Resolution Fallback Strategy:** The ingestion engine attempts downloading streams in the following priority order:
1. **1080p** (FHD stream)
2. **720p** (HD stream)
3. Falls back to **any resolution above 720p** (prioritizing the stream closest to 720p).


* **Integrity Validation:** Discards corrupted streams, missing audio tracks, or unavailable/private entries automatically.

### Stage 0: Ingestion & Audio-Visual Normalization

* **Frame Rate Standardization:** Converted all raw source videos to a constant frame rate (**CFR 25.0 FPS**) to eliminate variable frame rate (VFR) drift and frame drops.
* **Audio Resampling:** Standardized audio streams to **16 kHz single-channel (mono)** for compatibility with speech and audio feature extractors (e.g., Wav2Vec, HuBERT).

### Stage 1.1: Presenter Tracking & Temporal Segmentation

* **Pose Estimation:** Applied YOLOv8-Pose to track presenter keypoints across continuous video streams.
* **Segment Discovery:** Identified continuous, uninterrupted intervals where a single active presenter is consistently visible and speaking.
* **Outlier Filtering:** Filtered out shot cuts, multi-speaker scenes, extreme occlusions, or presenter dropouts.
* **Metadata Export:** Emitted segment-level metadata (`stage1_manifest.parquet`) containing frame ranges, keypoint motion statistics, and speaker cluster identifiers.

### Stage 1.2: Crop Geometry & Motion Bounding Optimization

* **Upper-Body Bounding Box:** Computed smooth, square crop coordinates `(bx, by, b_side)` per segment.
* **Gesture Retention:** Evaluated presenter wrist reach and upper-body motion bounds to guarantee that hand gestures and arm motions remain unclipped inside the square frame.
* **Metadata Consolidation:** Merged crop geometry records with the primary manifest using DuckDB (`stage1_manifest_consolidated.parquet`).

### Stage 2.1: Window Slicing & Speaker-Aware Dataset Splitting

* **Fixed-Length Windows:** Segmented valid continuous video streams into non-overlapping **100-frame windows (4.0 seconds @ 25 FPS)**.
* **Leakage-Free Splitting:** Employed deterministic hashing on unique speaker cluster IDs (`speaker_cluster_id`) to assign clips to **Train (85%)**, **Validation (10%)**, and **Test (5%)** sets, ensuring no speaker identity appears in multiple splits.
* **Render Manifest:** Generated `render_manifest.parquet` containing exact frame offsets, audio timestamps, crop parameters, and partition labels.

### Stage 2.2: Distributed Rendering & Final Dataset Output

* **Parallel Processing:** Executed high-throughput batch rendering with FFmpeg using Lanczos scaling to **512×512 resolution**.
* **Audio-Video Alignment:** Enforced synchronized 16 kHz AAC audio tracks with exact temporal cut points.
* **Storage Organization:** Structured rendered clips using two-level prefix hashing (`rendered/<split>/<prefix>/<window_uid>/video.mp4`) to optimize I/O on distributed cluster file systems.

---

## 🛠️ Quality Assurance & Auditing

* **Frame-by-Frame Presence Audit:** Integrated automated auditing tools (`audit_exact_10k_stride1.py`) using GPU batch inference to verify 100% presenter presence across random dataset samples.
* **Reproducibility:** All processing steps utilize deterministic hashes and parameter validation to ensure identical reproducibility across runs.

---

## 💻 Tech Stack

* **Core:** Python 3.10+, PyTorch, Torchvision, Ultralytics YOLO
* **Media Processing:** FFmpeg, FFprobe, OpenCV, PyTube
* **Data & Metadata:** DuckDB, Apache Arrow / PyArrow, Pandas
* **Infrastructure:** SLURM HPC Cluster, SquashFS Containerization
