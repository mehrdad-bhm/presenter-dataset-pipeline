#!/usr/bin/env python3
"""
================================================================================
Stage 2.2 — Single-Stream Rendering (AV Muxed, CPU, ffmpeg only)
Optimized for deterministic Slurm Array Sharding.
================================================================================

Description:
  Executes high-throughput, deterministic video/audio clipping and spatial cropping
  from standardized 25 FPS source videos into standalone 100-frame (4-second) MP4 clips
  with embedded audio (video + audio in a single container).

Inputs:
  1. --manifest (render_manifest.parquet):
     Parquet table containing pre-calculated window parameters with columns:
       - window_uid       (str)   : Unique window identifier (e.g., 'ab12cd34...')
       - path             (str)   : Absolute path to the source 25 FPS video (.mp4)
       - start_frame_abs  (int)   : Absolute start frame index in the source video
       - bx, by           (float) : Top-left bounding box coordinates for body crop
       - b_side           (float) : Crop square side length in source pixels

Outputs:
  Per window_uid (saved under: <out-root>/<window_uid[:2]>/<window_uid>/):
    1. video.mp4 : 512x512 (or 1024x1024), 25 FPS, 100 frames, H.264 + 16kHz AAC Audio

  Status Tracking:
    - <status-dir>/part-<task_id>.parquet : Execution log per window (status, latency, errors)

Usage Examples:
  # 1. Single-node local test / Pilot run (first 500 samples):
  python stage2.2_render.py \
      --manifest render_manifest.parquet \
      --out-root /path/to/rendered_chunks \
      --status-dir /path/to/status \
      --workers 8 \
      --limit 500

  # 2. SLURM Array Execution (Multi-Node / Multi-Core):
  # In your SLURM bash script (#SBATCH --array=0-99):
    python /netscratch/bahrami/src/data_prep/stage2.2_render.py \
        --manifest   /netscratch/bahrami/dataset/stage1_metadata/render_manifest.parquet \
        --out-root   /netscratch/bahrami/dataset/stage2_chunks_512 \
        --status-dir /netscratch/bahrami/dataset/stage2_chunks_512/_status \
        --num-tasks  30 \
        --task-id    $SLURM_ARRAY_TASK_ID \
        --workers    14
================================================================================
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
import subprocess
import sys
import time

import cv2
import pandas as pd

FPS = 25.0
OUTPUT_SIZE = 512
N_FRAMES = 100
CLIP_SECONDS = N_FRAMES / FPS
AUDIO_SR = 16000

CRF = 16
PRESET = "veryfast"
THREADS_PER_ENCODE = 2
FLUSH_EVERY = 200


def stable_bucket(key: str, num_tasks: int) -> int:
  return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % num_tasks


def load_shard_done_uids(status_file: Path) -> set[str]:
  """Load completed window UIDs for this specific shard to avoid I/O contention."""
  if not status_file.exists():
    return set()
  try:
    df = pd.read_parquet(status_file, columns=["window_uid", "status"])
    return set(df.loc[df["status"] == "ok", "window_uid"])
  except Exception:
    return set()


def _build_cmd(
    parent: str,
    seek_sec: float,
    b_side_i: int,
    bx_i: int,
    by_i: int,
    tmp_video: Path,
    silent_audio: bool,
) -> list[str]:
  filter_complex = (
      f"[0:v:0]crop={b_side_i}:{b_side_i}:{bx_i}:{by_i},"
      f"scale={OUTPUT_SIZE}:{OUTPUT_SIZE}:flags=bicubic[out_video]"
  )

  cmd = [
      "ffmpeg",
      "-y",
      "-nostdin",
      "-hide_banner",
      "-loglevel",
      "error",
      "-ss",
      f"{seek_sec:.6f}",
      "-i",
      str(parent),
  ]

  if silent_audio:
    cmd += [
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={AUDIO_SR}:cl=mono:d={CLIP_SECONDS:.6f}",
    ]
    audio_map = "1:a:0"
  else:
    audio_map = "0:a:0?"

  cmd += [
      "-filter_complex",
      filter_complex,
      "-map",
      "[out_video]",
      "-map",
      audio_map,
      "-frames:v",
      str(N_FRAMES),
      "-fps_mode",
      "cfr",
      "-c:v",
      "libx264",
      "-preset",
      PRESET,
      "-crf",
      str(CRF),
      "-pix_fmt",
      "yuv420p",
      "-threads",
      str(THREADS_PER_ENCODE),
      "-t",
      f"{CLIP_SECONDS:.6f}",
      "-af",
      f"aresample={AUDIO_SR}:async=0",
      "-ac",
      "1",
      "-c:a",
      "aac",
      "-b:a",
      "128k",
      "-movflags",
      "+faststart",
      "-f",
      "mp4",
      str(tmp_video),
  ]
  return cmd


def render_one(row: dict, out_root: Path) -> dict:
  t0 = time.time()
  window_uid = row["window_uid"]
  parent = row["path"]

  out_dir = out_root / window_uid[:2] / window_uid
  out_video = out_dir / "video.mp4"

  # Skip if target file already exists and is complete
  if out_video.exists():
    return {
        "window_uid": window_uid,
        "status": "ok",
        "error": None,
        "body_side_px": row["b_side"],
        "silent_audio_fallback": False,
        "elapsed_s": time.time() - t0,
    }

  bx, by, b_side = row["bx"], row["by"], row["b_side"]
  abs_start = row["start_frame_abs"]
  seek_sec = abs_start / FPS

  b_side_i = int(round(b_side)) & ~1
  bx_i, by_i = int(round(bx)), int(round(by))

  out_dir.mkdir(parents=True, exist_ok=True)
  tmp_video = out_video.with_suffix(".mp4.part")

  silent_audio = False
  cmd = _build_cmd(
      parent,
      seek_sec,
      b_side_i,
      bx_i,
      by_i,
      tmp_video,
      silent_audio=False,
  )
  res = subprocess.run(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
  )

  # Fallback to silent synthetic audio track if original stream has no audio
  if res.returncode != 0:
    stderr_txt = res.stderr.decode(errors="ignore")
    if "does not contain any stream" in stderr_txt:
      silent_audio = True
      tmp_video.unlink(missing_ok=True)
      cmd = _build_cmd(
          parent,
          seek_sec,
          b_side_i,
          bx_i,
          by_i,
          tmp_video,
          silent_audio=True,
      )
      res = subprocess.run(
          cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
      )

  if res.returncode != 0:
    tmp_video.unlink(missing_ok=True)
    return {
        "window_uid": window_uid,
        "status": "ffmpeg_failed",
        "error": res.stderr.decode(errors="ignore")[-500:],
        "silent_audio_fallback": silent_audio,
        "elapsed_s": time.time() - t0,
    }

  # Frame count verification
  cap = cv2.VideoCapture(str(tmp_video))
  video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  cap.release()

  if video_frames != N_FRAMES:
    tmp_video.unlink(missing_ok=True)
    return {
        "window_uid": window_uid,
        "status": "frame_count_mismatch",
        "error": f"video={video_frames} expected={N_FRAMES}",
        "silent_audio_fallback": silent_audio,
        "elapsed_s": time.time() - t0,
    }

  tmp_video.replace(out_video)

  return {
      "window_uid": window_uid,
      "status": "ok",
      "error": None,
      "body_side_px": b_side,
      "silent_audio_fallback": silent_audio,
      "elapsed_s": time.time() - t0,
  }


def main() -> None:
  ap = argparse.ArgumentParser(
      description="Stage 2.2: Deterministic Chunk Video Renderer"
  )
  ap.add_argument("--manifest", required=True)
  ap.add_argument("--out-root", required=True)
  ap.add_argument("--status-dir", required=True)
  ap.add_argument("--task-id", type=int, default=0)
  ap.add_argument("--num-tasks", type=int, default=30)
  ap.add_argument("--workers", type=int, default=14)
  ap.add_argument("--limit", type=int, default=None)
  args = ap.parse_args()

  out_root = Path(args.out_root)
  status_dir = Path(args.status_dir)
  status_dir.mkdir(parents=True, exist_ok=True)

  task_id = args.task_id
  status_path = status_dir / f"part_{task_id:04d}.parquet"

  # 1. Load manifest
  manifest = pd.read_parquet(args.manifest)

  # 2. Stable hash-based sharding
  manifest["_bucket"] = manifest["window_uid"].apply(
      stable_bucket, args=(args.num_tasks,)
  )
  manifest = (
      manifest[manifest["_bucket"] == task_id]
      .drop(columns="_bucket")
      .sort_values("window_uid")
  )

  # 3. Resume logic per task shard
  done = load_shard_done_uids(status_path)
  if done:
    before = len(manifest)
    manifest = manifest[~manifest["window_uid"].isin(done)]
    print(
        f"[Task {task_id}] Resume: skipping {before - len(manifest)} already-ok"
        " windows"
    )

  if args.limit:
    manifest = manifest.head(args.limit)

  print(
      f"Task ({task_id}/{args.num_tasks}): {len(manifest)} pending windows to"
      " render"
  )
  if len(manifest) == 0:
    return

  rows = manifest.to_dict(orient="records")
  results = []
  t0 = time.time()

  # 4. Multi-processing rendering loop
  with ProcessPoolExecutor(max_workers=args.workers) as ex:
    futures = {
        ex.submit(render_one, row, out_root): row["window_uid"] for row in rows
    }
    for i, fut in enumerate(as_completed(futures), 1):
      res = fut.result()
      results.append(res)

      if i % FLUSH_EVERY == 0 or i == len(rows):
        n_ok = sum(r["status"] == "ok" for r in results)
        rate = i / max(1e-6, time.time() - t0)
        print(
            f"[Task {task_id}] [{i}/{len(rows)}] ok={n_ok} failed={i - n_ok} |"
            f" {rate:.1f} win/s"
        )

  # 5. Persist status updates to parquet
  new_df = pd.DataFrame(results)
  if status_path.exists():
    try:
      old_df = pd.read_parquet(status_path)
      status_df = pd.concat([old_df, new_df], ignore_index=True)
    except Exception:
      status_df = new_df
  else:
    status_df = new_df

  status_df.to_parquet(status_path, index=False, compression="zstd")
  print(f"\n[Task {task_id}] Finished. Total logged: {len(status_df)}")


if __name__ == "__main__":
  main()
