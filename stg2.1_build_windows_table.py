#!/usr/bin/env python3
"""
================================================================================
Stage 2.1 — Advanced Geometry Refinement, Gating & Window Slicing (OOM-Safe)
================================================================================

Description:
  Evaluates Stage 1 candidate segments using full-body pose keypoints and tracking
  metadata. Recalculates safe spatial crops (bx, by, b_side) tailored for co-speech
  gestures, applies strict frontal alignment and lip-sync face-resolution gating,
  and slices surviving segments into deterministic, non-overlapping 100-frame
  temporal windows (4.0s @ 25 FPS) for Stage 2.2 rendering.

Inputs:
  - --stage1-dir (Directory containing Stage 1 outputs):
      1. stage1_manifest.parquet:
           Columns: segment_uid, video_id, path, seg_index, start_frame, end_frame,
                    width, height, fps, track_cx, track_cy, track_h, status
      2. stage1_keypoints.parquet:
           Columns: segment_uid, kpt_x, kpt_y, kpt_conf

Outputs:
  - --out-dir / render_manifest.parquet:
      Columns: window_uid, segment_uid, video_id, path, start_frame_abs,
               bx, by, b_side, face_height_render, width, height, fps

Usage:
  python /netscratch/bahrami/src/data_prep/stg2.1_build_windows_table.py \
      --stage1-dir          /netscratch/bahrami/dataset/stage1_metadata \
      --out-dir             /netscratch/bahrami/dataset/stage1_metadata \
      --crop-scale          2.0 \
      --hand-margin-frac    0.20 \
      --min-face-size-px    100.0 \
      --max-shoulder-tilt   0.25 \
      --max-wrist-cutoff-frac 0.03 \
      --window-frames       100
================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KP_NOSE = 0
KP_LEYE, KP_REYE = 1, 2
KP_LEAR, KP_REAR = 3, 4
KP_LSHOULDER, KP_RSHOULDER = 5, 6
KP_LELBOW, KP_RELBOW = 7, 8
KP_LWRIST, KP_RWRIST = 9, 10
N_KPT = 17


def moving_average(x: np.ndarray, span: int) -> np.ndarray:
    if span <= 1:
        return x.astype(np.float64)
    if span % 2 == 0:
        span += 1
    pad = span // 2
    padded = np.pad(x.astype(np.float64), pad, mode="edge")
    return np.convolve(padded, np.ones(span) / span, mode="valid")


def clamp_box(cx: float, cy: float, side: float, width: int, height: int, min_crop_px: int):
    reason = None
    max_side = min(width, height)
    if side > max_side:
        side = float(max_side)
        reason = "shrunk_to_frame"
    x = cx - side / 2.0
    y = cy - side / 2.0
    x = min(max(x, 0.0), width - side)
    y = min(max(y, 0.0), height - side)

    if side < min_crop_px:
        reason = "shrunk_below_min_crop_px" if reason == "shrunk_to_frame" else "below_min_crop_px"
    return x, y, side, reason


def evaluate_window_geometry(
    w_start: int,
    w_len: int,
    track_cx: np.ndarray,
    track_cy: np.ndarray,
    track_h: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    kc: np.ndarray,
    width: int,
    height: int,
    args: argparse.Namespace
) -> dict:
    w_slice = slice(w_start, w_start + w_len)

    # 1. Frontal Alignment
    l_sh_c = kc[w_slice, KP_LSHOULDER]
    r_sh_c = kc[w_slice, KP_RSHOULDER]
    valid_sh = (l_sh_c > args.kpt_conf) & (r_sh_c > args.kpt_conf)
    if valid_sh.sum() < (w_len * 0.5):
        return {"kept": False, "drop_reason": "shoulders_not_visible"}

    sh_dy = np.abs(ky[w_slice, KP_LSHOULDER][valid_sh] - ky[w_slice, KP_RSHOULDER][valid_sh])
    sh_dx = np.abs(kx[w_slice, KP_LSHOULDER][valid_sh] - kx[w_slice, KP_RSHOULDER][valid_sh])
    tilt_ratios = sh_dy / np.maximum(sh_dx, 1.0)
    if float(np.median(tilt_ratios)) > args.max_shoulder_tilt:
        return {"kept": False, "drop_reason": "tilted_presenter"}

    # 2. Hand Presence
    l_wr_c = kc[w_slice, KP_LWRIST]
    r_wr_c = kc[w_slice, KP_RWRIST]
    wrist_seen = (l_wr_c > args.wrist_conf_min) | (r_wr_c > args.wrist_conf_min)
    if float(np.mean(wrist_seen)) < args.min_wrist_presence_frac:
        return {"kept": False, "drop_reason": "hands_not_visible_enough"}

    # 3. Static Center & Crop Box
    span = min(args.smooth_span_frames, max(3, int(w_len * args.smooth_span_frac)))
    cx_smooth = moving_average(track_cx[w_slice], span)
    cy_smooth = moving_average(track_cy[w_slice], span)
    sz_smooth = moving_average(track_h[w_slice], span)

    bcx = float(np.median(cx_smooth))
    bcy = float(np.median(cy_smooth))
    base_side = float(np.median(sz_smooth)) * args.crop_scale

    dists = []
    if (l_wr_c > args.wrist_conf_min).any():
        m = l_wr_c > args.wrist_conf_min
        dists.append(np.hypot(kx[w_slice, KP_LWRIST][m] - track_cx[w_slice][m],
                              ky[w_slice, KP_LWRIST][m] - track_cy[w_slice][m]))
    if (r_wr_c > args.wrist_conf_min).any():
        m = r_wr_c > args.wrist_conf_min
        dists.append(np.hypot(kx[w_slice, KP_RWRIST][m] - track_cx[w_slice][m],
                              ky[w_slice, KP_RWRIST][m] - track_cy[w_slice][m]))

    wrist_reach = float(np.percentile(np.concatenate(dists), args.wrist_percentile)) if dists else 0.0
    required_half = wrist_reach * (1.0 + args.hand_margin_frac)
    b_side = max(base_side, 2.0 * required_half)

    bx, by, b_side, reason = clamp_box(bcx, bcy, b_side, width, height, args.min_crop_px)
    if reason in ("below_min_crop_px", "shrunk_below_min_crop_px"):
        return {"kept": False, "drop_reason": reason}

    # 4. Strict Wrist Out-of-Bound Check
    valid_wrists_x, valid_wrists_y = [], []
    if (l_wr_c > args.wrist_conf_min).any():
        m = l_wr_c > args.wrist_conf_min
        valid_wrists_x.append(kx[w_slice, KP_LWRIST][m])
        valid_wrists_y.append(ky[w_slice, KP_LWRIST][m])
    if (r_wr_c > args.wrist_conf_min).any():
        m = r_wr_c > args.wrist_conf_min
        valid_wrists_x.append(kx[w_slice, KP_RWRIST][m])
        valid_wrists_y.append(ky[w_slice, KP_RWRIST][m])

    if valid_wrists_x:
        vwx = np.concatenate(valid_wrists_x)
        vwy = np.concatenate(valid_wrists_y)
        out_of_box = (vwx < bx) | (vwx > (bx + b_side)) | (vwy < by) | (vwy > (by + b_side))
        if float(np.mean(out_of_box)) > args.max_wrist_cutoff_frac:
            return {"kept": False, "drop_reason": "wrist_clipped_boundary"}

    # 5. Face Size & Headroom Check
    face_height_source = float(np.median(
        np.maximum(ky[w_slice, KP_LSHOULDER], ky[w_slice, KP_RSHOULDER]) - ky[w_slice, KP_NOSE]
    ))
    face_height_render = (face_height_source / b_side) * 512.0
    if face_height_render < args.min_face_size_px:
        return {"kept": False, "drop_reason": "face_too_small_for_lipsync"}

    nose_y_render = ((float(np.median(ky[w_slice, KP_NOSE])) - by) / b_side) * 512.0
    if nose_y_render < 60.0 or nose_y_render > 220.0:
        return {"kept": False, "drop_reason": "inconsistent_headroom"}

    return {
        "kept": True,
        "bx": float(bx),
        "by": float(by),
        "b_side": float(b_side),
        "face_height_render": face_height_render,
        "drop_reason": "ok"
    }


def main():
    ap = argparse.ArgumentParser(description="Stage 2.1: Streaming Geometry Refinement")
    ap.add_argument("--stage1-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--crop-scale", type=float, default=2.0)
    ap.add_argument("--hand-margin-frac", type=float, default=0.20)
    ap.add_argument("--min-face-size-px", type=float, default=100.0)
    ap.add_argument("--max-shoulder-tilt", type=float, default=0.25)
    ap.add_argument("--max-wrist-cutoff-frac", type=float, default=0.03)
    ap.add_argument("--min-wrist-presence-frac", type=float, default=0.65)

    ap.add_argument("--kpt-conf", type=float, default=0.40)
    ap.add_argument("--wrist-conf-min", type=float, default=0.30)
    ap.add_argument("--wrist-percentile", type=float, default=95.0)
    ap.add_argument("--smooth-span-frames", type=int, default=25)
    ap.add_argument("--smooth-span-frac", type=float, default=1.0)
    ap.add_argument("--min-crop-px", type=int, default=128)
    ap.add_argument("--window-frames", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=5000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    manifest_path = args.stage1_dir / "stage1_manifest.parquet"
    keypoints_path = args.stage1_dir / "stage1_keypoints.parquet"

    print("Loading lightweight Stage 1 manifest...")
    man_df = pd.read_parquet(manifest_path)
    man_df = man_df[man_df["status"] == "ok"].set_index("segment_uid")
    print(f"Candidate Stage 1 segments: {len(man_df)}")

    print(f"Streaming keypoints from {keypoints_path.name} in batches of {args.batch_size}...")
    kpt_file = pq.ParquetFile(keypoints_path)

    window_records = []
    drop_stats = {}
    total_processed = 0

    for batch in kpt_file.iter_batches(batch_size=args.batch_size, columns=["segment_uid", "kpt_x", "kpt_y", "kpt_conf"]):
        batch_df = batch.to_pandas().set_index("segment_uid")
        common_uids = batch_df.index.intersection(man_df.index)

        for uid in common_uids:
            row = man_df.loc[uid]
            kpt_row = batch_df.loc[uid]

            track_cx = np.asarray(row["track_cx"], dtype=np.float64)
            track_cy = np.asarray(row["track_cy"], dtype=np.float64)
            track_h  = np.asarray(row["track_h"], dtype=np.float64)

            kx = np.asarray(kpt_row["kpt_x"], dtype=np.float64).reshape(-1, N_KPT)
            ky = np.asarray(kpt_row["kpt_y"], dtype=np.float64).reshape(-1, N_KPT)
            kc = np.asarray(kpt_row["kpt_conf"], dtype=np.float64).reshape(-1, N_KPT)

            s_frame = int(row["start_frame"])
            e_frame = int(row["end_frame"])
            n_frames = e_frame - s_frame + 1
            n_windows = n_frames // args.window_frames

            for w_idx in range(n_windows):
                offset = w_idx * args.window_frames
                geom = evaluate_window_geometry(
                    offset, args.window_frames, track_cx, track_cy, track_h,
                    kx, ky, kc, int(row["width"]), int(row["height"]), args
                )

                if geom["kept"]:
                    w_start = s_frame + offset
                    window_uid = f"{row['video_id']}_seg{int(row['seg_index']):04d}_w{w_idx:04d}"
                    window_records.append({
                        "window_uid": window_uid,
                        "segment_uid": uid,
                        "video_id": row["video_id"],
                        "path": row["path"],
                        "start_frame_abs": w_start,
                        "bx": geom["bx"],
                        "by": geom["by"],
                        "b_side": geom["b_side"],
                        "face_height_render": geom["face_height_render"],
                        "width": row["width"],
                        "height": row["height"],
                        "fps": row["fps"]
                    })
                else:
                    reason = geom["drop_reason"]
                    drop_stats[reason] = drop_stats.get(reason, 0) + 1

        total_processed += len(common_uids)
        print(f"Processed {total_processed}/{len(man_df)} segments | Windows so far: {len(window_records)}", end="\r", flush=True)

    print("\nWriting output manifest...")
    windows_df = pd.DataFrame(window_records)
    out_manifest = args.out_dir / "render_manifest.parquet"
    windows_df.to_parquet(out_manifest, index=False, compression="zstd")

    print(f"\nFinished in {time.time() - t0:.1f}s")
    print(f"Saved {len(windows_df)} windows -> {out_manifest}")
    print("Rejection reasons summary:")
    for reason, count in sorted(drop_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
