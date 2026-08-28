#!/usr/bin/env python3
"""
================================================================================
Stage 2.1 — Window Slicing (CPU, metadata-only, no ffmpeg)
================================================================================

Description:
  Slices valid segments from Stage 1 into non-overlapping 100-frame temporal windows
  (4.0s @ 25 FPS), propagates spatial crop bounds (bx, by, b_side), and assigns
  leak-free speaker-hashed train/val/test splits.

Inputs:
  - --manifest (stage1_manifest.parquet):
      Columns: segment_uid, video_id, path, start_frame, end_frame, n_frames, kept,
               bx, by, b_side, [speaker_cluster_id]

Outputs:
  - --out (render_manifest.parquet):
      Columns: window_uid, segment_uid, video_id, speaker_cluster_id, window_index,
               start_frame_abs, end_frame_abs, n_frames, split, prev_window_key,
               next_window_key, path, bx, by, b_side

Usage:
  python build_windows_table.py \
      --manifest stage1_manifest.parquet \
      --out render_manifest.parquet \
      --window-len 100 \
      --stride 100
================================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import sys

import pandas as pd
import pyarrow.parquet as pq

WINDOW_LEN = 100
STRIDE = 100
SPLIT_TRAIN, SPLIT_VAL = 85, 95


def stable_hash_bucket(key: str) -> int:
    return int(hashlib.md5(str(key).encode("utf-8")).hexdigest(), 16) % 100


def assign_split(cluster_key: str) -> str:
    b = stable_hash_bucket(cluster_key)
    if b < SPLIT_TRAIN:
        return "train"
    elif b < SPLIT_VAL:
        return "val"
    return "test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="stage1_manifest.parquet")
    ap.add_argument("--out", default="render_manifest.parquet")
    ap.add_argument("--window-len", type=int, default=WINDOW_LEN)
    ap.add_argument("--stride", type=int, default=STRIDE)
    args = ap.parse_args()

    cols = [
        "segment_uid", "video_id", "path",
        "start_frame", "end_frame", "n_frames",
        "kept", "bx", "by", "b_side", "speaker_cluster_id",
    ]
    schema_names = set(pq.ParquetFile(args.manifest).schema.names)
    missing_speaker = "speaker_cluster_id" not in schema_names
    if missing_speaker:
        cols.remove("speaker_cluster_id")

    df = pd.read_parquet(args.manifest, columns=cols)
    df = df[df["kept"] == True].copy()
    print(f"Loaded {len(df):,} kept segments from {args.manifest}")

    if missing_speaker:
        df["speaker_cluster_id"] = df["video_id"]

    records = []
    n_skipped_short = 0

    for row in df.itertuples(index=False):
        n_frames = int(row.n_frames)
        n_win = n_frames // args.window_len
        if n_win == 0:
            n_skipped_short += 1
            continue

        seg_uid = str(row.segment_uid)
        base_start = int(row.start_frame)
        split = assign_split(str(row.speaker_cluster_id))

        for w_idx in range(n_win):
            win_start_rel = w_idx * args.stride
            abs_start = base_start + win_start_rel
            abs_end = abs_start + args.window_len

            window_uid = f"{seg_uid}_w{w_idx:04d}"
            prev_key = f"{seg_uid}_w{w_idx - 1:04d}" if w_idx > 0 else None
            next_key = f"{seg_uid}_w{w_idx + 1:04d}" if w_idx < n_win - 1 else None

            records.append({
                "window_uid": window_uid,
                "segment_uid": seg_uid,
                "video_id": str(row.video_id),
                "speaker_cluster_id": str(row.speaker_cluster_id),
                "window_index": int(w_idx),
                "n_windows_in_segment": int(n_win),
                "start_frame_abs": int(abs_start),
                "end_frame_abs": int(abs_end),
                "n_frames": int(args.window_len),
                "split": split,
                "prev_window_key": prev_key,
                "next_window_key": next_key,
                "path": str(row.path),
                # مختصات کادر برش بدن برای ورکر رندر
                "bx": float(row.bx),
                "by": float(row.by),
                "b_side": float(row.b_side),
            })

    out = pd.DataFrame.from_records(records)
    print(f"Generated {len(out):,} non-overlapping {args.window_len}-frame windows "
          f"({n_skipped_short:,} segments skipped because len < {args.window_len})")

    if len(out) == 0:
        print("ERROR: zero windows produced — check manifest filtering.", file=sys.stderr)
        sys.exit(1)

    split_pct = out["split"].value_counts(normalize=True).mul(100).round(2)
    print(f"Split distribution (%):\n{split_pct}")

    out.to_parquet(args.out, index=False, compression="zstd")
    print(f"Wrote render manifest to {args.out}")


if __name__ == "__main__":
    main()