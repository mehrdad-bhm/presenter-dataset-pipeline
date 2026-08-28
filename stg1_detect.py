#!/usr/bin/env python3
"""
================================================================================
Stage 1 — Presenter Detection, Keypoints & Integrated Crop Geometry
================================================================================

Description:
  Executes YOLO-Pose presenter detection, temporal segment boundary extraction,
  COCO-17 keypoint tracking, and pre-computes static square body-crop geometry
  (bx, by, b_side) directly in a single pass. Metadata out, NO video out.

Inputs:
  - 25.0 FPS CFR normalized videos (from normalize_raw_videos.py)

Outputs:
  1. stage1_manifest.parquet   : One row per segment containing temporal boundaries,
                                 tracking trajectories, and computed crop geometry.
  2. stage1_keypoints.parquet  : One row per segment with flattened (N_det, 17) keypoints.
  3. stage1_manifest_preview.csv: Per-video summary for quick inspection.
  4. stage1_run.json           : Run configurations, parameter hash, and summary metrics.

Usage:
  python stage1_detect.py \
      --input-dir  /netscratch/bahrami/dataset/talking_pose_25fps \
      --output-dir /netscratch/bahrami/dataset/stage1_meta \
      --num-workers 6 --frame-stride 4 --batch-size 64 --half
================================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import queue
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("stage1")

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}
TARGET_FPS = 25.0
FPS_TOL = 0.01
HEAD_TAIL_BYTES = 16384
STAGE1_VERSION = "2.0.0"
FLUSH_EVERY = 50          # videos buffered before a shard flush

# COCO-17 keypoint indices (YOLOv8-Pose)
KP_NOSE, KP_LEYE, KP_REYE, KP_LEAR, KP_REAR = 0, 1, 2, 3, 4
KP_LSHOULDER, KP_RSHOULDER = 5, 6
KP_LELBOW, KP_RELBOW = 7, 8
KP_LWRIST, KP_RWRIST = 9, 10
KP_LKNEE, KP_RKNEE, KP_LANKLE, KP_RANKLE = 13, 14, 15, 16
WRIST_IDX = (KP_LWRIST, KP_RWRIST)
N_KPT = 17


# --------------------------------------------------------------------------- #
# Config & Geometry Parameters
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    input_dir: Path
    output_dir: Path

    # Inference settings
    model_name: str = "yolov8s-pose.pt"
    downscale_height: int = 0
    imgsz: int = 512
    batch_size: int = 64
    half: bool = True
    frame_stride: int = 4
    reader_queue: int = 16
    hwaccel: str = "cuda"

    # Presenter decision gates
    person_conf: float = 0.30
    min_person_height_frac: float = 0.45
    exclude_full_body: bool = True
    lower_body_conf: float = 0.50
    kpt_conf: float = 0.50
    prominence_ratio: float = 1.6

    # Hand visibility gating
    min_visible_hands: int = 1
    hand_extent_ratio: float = 0.75
    hand_fallback_frac: float = 0.11
    hand_frame_margin: float = 0.01

    # Temporal segmentation
    min_segment_frames: int = 104     # 100 + frame_stride slop
    gap_tolerance: int = 15

    # Motion scoring gates
    motion_min: float = 0.0010
    hand_motion_min: float = 0.0050
    hand_motion_pctl: float = 75.0

    # Body-Crop Geometry parameters
    crop_scale: float = 1.8
    smooth_span_frac: float = 1.0
    smooth_span_frames: int = 25
    hand_margin_frac: float = 0.10
    wrist_conf_min: float = 0.30
    wrist_percentile: float = 95.0
    min_crop_px: int = 128

    num_workers: int = 6
    overwrite: bool = False
    normalize_report: Optional[Path] = None

    def param_hash(self) -> str:
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()
             if k not in ("input_dir", "output_dir", "num_workers", "overwrite",
                          "reader_queue", "batch_size", "hwaccel", "normalize_report")}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Video Probing & Fingerprinting
# --------------------------------------------------------------------------- #
def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def probe_video(path: Path) -> Optional[dict]:
    res = _run(["ffprobe", "-v", "error", "-print_format", "json",
                "-show_streams", "-show_format", str(path)])
    if res.returncode != 0:
        return None
    try:
        info = json.loads(res.stdout.decode(errors="ignore"))
    except json.JSONDecodeError:
        return None
    out = {"width": None, "height": None, "fps": None, "nb_frames": None, "duration": None}
    for st in info.get("streams", []):
        if st.get("codec_type") == "video":
            out["width"], out["height"] = st.get("width"), st.get("height")
            try:
                num, den = (st.get("avg_frame_rate") or "0/0").split("/")
                out["fps"] = float(num) / float(den) if float(den) else None
            except (ValueError, ZeroDivisionError):
                pass
            try:
                out["nb_frames"] = int(st.get("nb_frames"))
            except (TypeError, ValueError):
                pass
            break
    try:
        out["duration"] = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        pass
    if not out["width"] or not out["fps"]:
        return None
    return out


def parent_fingerprint(path: Path) -> Tuple[int, str]:
    size = path.stat().st_size
    h = hashlib.md5()
    with path.open("rb") as f:
        h.update(f.read(HEAD_TAIL_BYTES))
        if size > HEAD_TAIL_BYTES * 2:
            f.seek(-HEAD_TAIL_BYTES, 2)
            h.update(f.read(HEAD_TAIL_BYTES))
    h.update(str(size).encode())
    return size, h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Threaded FFmpeg Frame Reader
# --------------------------------------------------------------------------- #
def _reader_thread(path: Path, out_w: int, out_h: int, stride: int, batch: int,
                   hwaccel: str, q: "queue.Queue"):
    vf = f"scale={out_w}:{out_h}"
    if stride > 1:
        vf = f"select=not(mod(n\\,{stride})),{vf}"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", str(path), "-vf", vf, "-fps_mode", "passthrough",
            "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=10 ** 8)
    frame_bytes = out_w * out_h * 3
    decoded = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes * batch)
            if not buf:
                break
            n = len(buf) // frame_bytes
            if n == 0:
                break
            arr = np.frombuffer(buf[: n * frame_bytes], np.uint8).reshape(n, out_h, out_w, 3)
            q.put((decoded, arr))
            decoded += n
    except Exception as exc:
        LOGGER.error("reader error %s: %s", path.name, exc)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()
        q.put(None)


def iter_batches(path: Path, out_w: int, out_h: int, cfg: Config) -> Iterator[Tuple[int, np.ndarray]]:
    q: "queue.Queue" = queue.Queue(maxsize=cfg.reader_queue)
    t = threading.Thread(target=_reader_thread,
                         args=(path, out_w, out_h, cfg.frame_stride, cfg.batch_size,
                               cfg.hwaccel, q), daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item
    t.join()


# --------------------------------------------------------------------------- #
# Frame Decision Logic
# --------------------------------------------------------------------------- #
@dataclass
class Det:
    valid: bool
    reason: str = "ok"
    n_persons: int = 0
    hfrac: float = 0.0
    hands: int = 0
    conf: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0
    kxy: Optional[np.ndarray] = None    # (17, 2)
    kconf: Optional[np.ndarray] = None  # (17,)


def _count_usable_hands(kxy, kc, box_h, frame_w, frame_h, cfg: Config) -> int:
    margin = cfg.hand_frame_margin * min(frame_w, frame_h)
    usable = 0
    for wrist_i, elbow_i in ((KP_LWRIST, KP_LELBOW), (KP_RWRIST, KP_RELBOW)):
        if kc[wrist_i] < cfg.kpt_conf:
            continue
        wx, wy = float(kxy[wrist_i][0]), float(kxy[wrist_i][1])
        if kc[elbow_i] >= cfg.kpt_conf:
            ex, ey = float(kxy[elbow_i][0]), float(kxy[elbow_i][1])
            r = cfg.hand_extent_ratio * float(np.hypot(wx - ex, wy - ey))
        else:
            r = cfg.hand_fallback_frac * box_h
        r = max(r, 4.0)
        if (wx - r >= margin and wy - r >= margin and
                wx + r <= frame_w - margin and wy + r <= frame_h - margin):
            usable += 1
    return usable


def decide_frame(boxes_xyxy, boxes_conf, kpts_xy, kpts_conf,
                 frame_h: int, frame_w: int, cfg: Config) -> Det:
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return Det(False, "no_detection")
    keep = boxes_conf >= cfg.person_conf
    if not np.any(keep):
        return Det(False, "low_conf")
    boxes, confs = boxes_xyxy[keep], boxes_conf[keep]
    kxy, kc = kpts_xy[keep], kpts_conf[keep]

    heights = boxes[:, 3] - boxes[:, 1]
    widths = boxes[:, 2] - boxes[:, 0]
    areas = np.clip(heights, 1, None) * np.clip(widths, 1, None)
    hfrac = heights / float(frame_h)
    n_persons = int(len(boxes))
    max_hfrac = float(hfrac.max())

    prominent = hfrac >= cfg.min_person_height_frac
    n_prom = int(prominent.sum())
    if n_prom == 0:
        return Det(False, "no_prominent", n_persons, max_hfrac)
    if n_prom > 1:
        return Det(False, "multi_prominent", n_persons, max_hfrac)
    p = int(np.argmax(prominent))

    if len(areas) > 1:
        order = np.argsort(areas)[::-1]
        if order[0] != p or areas[order[0]] < cfg.prominence_ratio * max(areas[order[1]], 1.0):
            return Det(False, "not_dominant", n_persons, max_hfrac)

    pc, box = kc[p], boxes[p]
    head = (pc[KP_NOSE] >= cfg.kpt_conf or
            (pc[KP_LEYE] >= cfg.kpt_conf and pc[KP_REYE] >= cfg.kpt_conf) or
            pc[KP_LEAR] >= cfg.kpt_conf or pc[KP_REAR] >= cfg.kpt_conf)
    shoulder = pc[KP_LSHOULDER] >= cfg.kpt_conf or pc[KP_RSHOULDER] >= cfg.kpt_conf
    if not (head and shoulder):
        return Det(False, "no_face_or_shoulder", n_persons, max_hfrac)

    box_h = max(float(box[3] - box[1]), 1.0)
    hands = 0
    if cfg.min_visible_hands > 0:
        if not any(pc[k] >= cfg.kpt_conf for k in WRIST_IDX):
            return Det(False, "no_wrist", n_persons, max_hfrac)
        hands = _count_usable_hands(kxy[p], pc, box_h, frame_w, frame_h, cfg)
        if hands < cfg.min_visible_hands:
            return Det(False, "hand_cropped", n_persons, max_hfrac)

    if cfg.exclude_full_body:
        lower = sum(int(pc[k] >= cfg.lower_body_conf)
                    for k in (KP_LKNEE, KP_RKNEE, KP_LANKLE, KP_RANKLE))
        if lower >= 2:
            return Det(False, "full_body", n_persons, max_hfrac)

    return Det(True, "ok", n_persons, max_hfrac, hands, float(confs[p]),
               cx=float((box[0] + box[2]) / 2), cy=float((box[1] + box[3]) / 2),
               w=float(box[2] - box[0]), h=box_h,
               kxy=kxy[p].astype(np.float32), kconf=pc.astype(np.float32))


# --------------------------------------------------------------------------- #
# Crop Geometry Math Functions (Integrated from compute_crop_geometry.py)
# --------------------------------------------------------------------------- #
def moving_average(x: np.ndarray, span: int) -> np.ndarray:
    if span <= 1:
        return x.astype(np.float64)
    if span % 2 == 0:
        span += 1
    pad = span // 2
    padded = np.pad(x.astype(np.float64), pad, mode="edge")
    return np.convolve(padded, np.ones(span) / span, mode="valid")


def densify_masked(track_frames: np.ndarray, values: np.ndarray, mask: np.ndarray,
                    start_frame: int, n_frames: int) -> np.ndarray | None:
    tf_v = track_frames[mask]
    val_v = values[mask]
    if len(tf_v) < 2:
        return None
    order = np.argsort(tf_v)
    tf_v, val_v = tf_v[order], val_v[order]
    rel = tf_v.astype(np.float64) - start_frame
    grid = np.arange(n_frames, dtype=np.float64)
    return np.interp(grid, rel, val_v.astype(np.float64))


def clamp_box(cx: float, cy: float, side: float, width: int, height: int,
              min_crop_px: int) -> tuple[float, float, float, str | None]:
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


def compute_segment_geometry(s: int, n_frames: int, width: int, height: int,
                             frames: list, D: list, cfg: Config) -> dict:
    """Calculates optimal static square crop bounds covering presenter and gestures."""
    out = {
        "bx": None, "by": None, "b_side": None, "body_side_px": None,
        "wrist_reach_px": 0.0, "body_coverage_frac": 0.0, "body_max_drift_px": 0.0,
        "geom_valid": False, "geom_drop_reason": None,
    }

    if len(frames) < 2:
        out["geom_drop_reason"] = "insufficient_box_track"
        return out

    track_frames = np.asarray(frames, dtype=np.int64)
    track_cx = np.asarray([d.cx for d in D], dtype=np.float64)
    track_cy = np.asarray([d.cy for d in D], dtype=np.float64)
    track_h = np.asarray([d.h for d in D], dtype=np.float64)

    valid_box = np.ones(len(track_frames), dtype=bool)
    cx_dense = densify_masked(track_frames, track_cx, valid_box, s, n_frames)
    cy_dense = densify_masked(track_frames, track_cy, valid_box, s, n_frames)
    sz_dense = densify_masked(track_frames, track_h, valid_box, s, n_frames)
    if cx_dense is None or cy_dense is None or sz_dense is None:
        out["geom_drop_reason"] = "insufficient_box_track"
        return out

    span = min(cfg.smooth_span_frames, max(3, int(n_frames * cfg.smooth_span_frac)))
    cx_smooth = moving_average(cx_dense, span)
    cy_smooth = moving_average(cy_dense, span)
    sz_smooth = moving_average(sz_dense, span)

    bcx = float(np.median(cx_smooth))
    bcy = float(np.median(cy_smooth))
    base_side = float(np.median(sz_smooth)) * cfg.crop_scale

    # Wrist excursion calculation
    wrist_reach = 0.0
    d_count = len(D)
    if d_count > 0:
        kx = np.asarray([d.kxy[:, 0] for d in D], dtype=np.float64)
        ky = np.asarray([d.kxy[:, 1] for d in D], dtype=np.float64)
        kc = np.asarray([d.kconf for d in D], dtype=np.float64)

        wrist_mask = (
            (kc[:, KP_LWRIST] > cfg.wrist_conf_min) |
            (kc[:, KP_RWRIST] > cfg.wrist_conf_min)
        )
        if wrist_mask.any():
            lwx, lwy, lwc = kx[:, KP_LWRIST], ky[:, KP_LWRIST], kc[:, KP_LWRIST]
            rwx, rwy, rwc = kx[:, KP_RWRIST], ky[:, KP_RWRIST], kc[:, KP_RWRIST]

            center_x_at = np.interp(track_frames.astype(np.float64), track_frames.astype(np.float64), track_cx)
            center_y_at = np.interp(track_frames.astype(np.float64), track_frames.astype(np.float64), track_cy)

            dists = []
            if (lwc > cfg.wrist_conf_min).any():
                m = lwc > cfg.wrist_conf_min
                dists.append(np.hypot(lwx[m] - center_x_at[m], lwy[m] - center_y_at[m]))
            if (rwc > cfg.wrist_conf_min).any():
                m = rwc > cfg.wrist_conf_min
                dists.append(np.hypot(rwx[m] - center_x_at[m], rwy[m] - center_y_at[m]))
            if dists:
                all_dists = np.concatenate(dists)
                wrist_reach = float(np.percentile(all_dists, cfg.wrist_percentile))

    required_half = wrist_reach * (1.0 + cfg.hand_margin_frac)
    b_side = max(base_side, 2.0 * required_half)

    bx, by, b_side, body_reason = clamp_box(bcx, bcy, b_side, width, height, cfg.min_crop_px)
    if body_reason in ("below_min_crop_px", "shrunk_below_min_crop_px"):
        out["geom_drop_reason"] = "body_below_min_crop_px"
        return out

    # Diagnostics
    half_sz = sz_smooth / 2.0
    body_left = cx_smooth - half_sz
    body_right = cx_smooth + half_sz
    body_top = cy_smooth - half_sz
    body_bottom = cy_smooth + half_sz

    in_box_mask = (
        (body_left >= bx) & (body_right <= (bx + b_side)) &
        (body_top >= by) & (body_bottom <= (by + b_side))
    )
    body_coverage_frac = float(np.mean(in_box_mask))
    max_drift_x = float(np.max(np.abs(cx_smooth - bcx)))
    max_drift_y = float(np.max(np.abs(cy_smooth - bcy)))

    out.update({
        "geom_valid": True,
        "geom_drop_reason": body_reason,
        "bx": float(bx), "by": float(by), "b_side": float(b_side),
        "body_side_px": float(b_side),
        "wrist_reach_px": wrist_reach,
        "body_coverage_frac": body_coverage_frac,
        "body_max_drift_px": max(max_drift_x, max_drift_y),
    })
    return out


# --------------------------------------------------------------------------- #
# Worker Initialization & Analysis
# --------------------------------------------------------------------------- #
_MODEL = None
_DEVICE = None
_CFG: Optional[Config] = None
_ORIGFPS: dict = {}


def init_worker(cfg: Config, worker_index: int, orig_fps_map: dict):
    global _MODEL, _DEVICE, _CFG, _ORIGFPS
    _CFG, _ORIGFPS = cfg, orig_fps_map
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(processName)s] %(levelname)s %(message)s")
    import torch
    from ultralytics import YOLO
    if torch.cuda.is_available():
        _DEVICE = f"cuda:{worker_index % max(1, torch.cuda.device_count())}"
        torch.backends.cudnn.benchmark = True
    else:
        _DEVICE = "cpu"
    _MODEL = YOLO(cfg.model_name)
    _MODEL.to(_DEVICE)
    if cfg.half and _DEVICE != "cpu":
        _MODEL.model.half()
    LOGGER.info("worker %d on %s (gpus visible=%d)", worker_index, _DEVICE,
                torch.cuda.device_count() if torch.cuda.is_available() else 0)


def analyze(path: Path, meta: dict, cfg: Config):
    orig_w, orig_h = meta["width"], meta["height"]
    if cfg.downscale_height and orig_h > cfg.downscale_height:
        out_h = cfg.downscale_height
        out_w = int(round(orig_w * out_h / orig_h))
    else:
        out_w, out_h = orig_w, orig_h
    out_w -= out_w % 2
    out_h -= out_h % 2
    sx, sy = orig_w / out_w, orig_h / out_h

    stride = max(1, cfg.frame_stride)
    dets: dict[int, Det] = {}
    last_decoded = -1

    for start_idx, batch in iter_batches(path, out_w, out_h, cfg):
        results = _MODEL.predict(list(batch), conf=cfg.person_conf, classes=[0],
                                 device=_DEVICE, imgsz=cfg.imgsz, verbose=False)
        for k, res in enumerate(results):
            b, kp = res.boxes, res.keypoints
            if b is None or kp is None or b.xyxy is None or len(b) == 0:
                d = Det(False, "no_detection")
            else:
                kpc = kp.conf.cpu().numpy() if kp.conf is not None else \
                    np.zeros(kp.xy.shape[:2])
                d = decide_frame(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                                 kp.xy.cpu().numpy(), kpc, out_h, out_w, cfg)
            if d.valid:
                d.cx *= sx; d.w *= sx; d.cy *= sy; d.h *= sy
                d.kxy = d.kxy * np.array([sx, sy], dtype=np.float32)
            di = start_idx + k
            dets[di * stride] = d
            last_decoded = di

    if last_decoded < 0:
        return np.zeros(0, dtype=bool), {}
    n_orig = (last_decoded + 1) * stride
    valid = np.zeros(n_orig, dtype=bool)
    for orig_idx, d in dets.items():
        if d.valid:
            valid[orig_idx: min(orig_idx + stride, n_orig)] = True
    return valid, dets


def raw_segments(valid: np.ndarray, tol: int) -> List[Tuple[int, int]]:
    segs, seg_start, last_valid = [], None, None
    for i in range(len(valid)):
        if valid[i]:
            if seg_start is None:
                seg_start = i
            last_valid = i
        elif seg_start is not None and (i - last_valid) > tol:
            segs.append((seg_start, last_valid))
            seg_start = last_valid = None
    if seg_start is not None:
        segs.append((seg_start, last_valid))
    return segs


def gap_stats(valid: np.ndarray, s: int, e: int) -> Tuple[int, int]:
    inv = ~valid[s:e + 1]
    n_gaps = max_gap = run = 0
    for v in inv:
        if v:
            run += 1
        elif run:
            n_gaps += 1
            max_gap = max(max_gap, run)
            run = 0
    if run:
        n_gaps += 1
        max_gap = max(max_gap, run)
    return n_gaps, max_gap


def segment_record(video: dict, seg_i: int, s: int, e: int, valid: np.ndarray,
                   dets: dict, cfg: Config):
    frames = sorted(f for f in range(s, e + 1) if f in dets and dets[f].valid)
    D = [dets[f] for f in frames]
    n_frames = e - s + 1
    uid = f"{video['video_id']}_seg{seg_i:04d}"

    diag = np.array([np.hypot(d.w, d.h) or 1.0 for d in D])
    body_motions, wrist_motions = [], []
    for i in range(1, len(D)):
        a, b = D[i - 1], D[i]
        dd = float(max(diag[i - 1], 1.0))
        m = np.abs(b.kxy - a.kxy)
        mask = (a.kconf >= cfg.kpt_conf) & (b.kconf >= cfg.kpt_conf)
        if mask.any():
            body_motions.append(float(m[mask].mean()) / dd)
        wmask = np.array([mask[j] for j in WRIST_IDX])
        wm = np.array([m[j].mean() for j in WRIST_IDX])
        if wmask.any():
            wrist_motions.append(float(wm[wmask].mean()) / dd)

    body_med = float(np.median(body_motions)) if body_motions else 0.0
    wrist_p = float(np.percentile(wrist_motions, cfg.hand_motion_pctl)) \
        if wrist_motions else 0.0

    # Compute crop geometry directly
    geom = compute_segment_geometry(s, n_frames, video["width"], video["height"], frames, D, cfg)

    kept, reason = True, ""
    if n_frames < cfg.min_segment_frames:
        kept, reason = False, "too_short"
    elif not geom["geom_valid"]:
        kept, reason = False, geom["geom_drop_reason"] or "invalid_crop_geometry"
    elif body_med < cfg.motion_min:
        kept, reason = False, "low_motion"
    elif cfg.hand_motion_min > 0 and wrist_p < cfg.hand_motion_min:
        kept, reason = False, "low_hand_motion"

    n_gaps, max_gap = gap_stats(valid, s, e)
    man = {
        **video,
        "segment_uid": uid, "seg_index": seg_i,
        "start_frame": s, "end_frame": e,
        "n_frames": n_frames, "n_windows": n_frames // 100,
        "track_stride": cfg.frame_stride,
        "track_frames": np.array(frames, dtype=np.int32),
        "track_cx": np.array([d.cx for d in D], dtype=np.float32),
        "track_cy": np.array([d.cy for d in D], dtype=np.float32),
        "track_w": np.array([d.w for d in D], dtype=np.float32),
        "track_h": np.array([d.h for d in D], dtype=np.float32),
        "track_conf": np.array([d.conf for d in D], dtype=np.float32),
        "track_hands": np.array([d.hands for d in D], dtype=np.int8),
        "n_bridged_gaps": n_gaps, "max_gap_frames": max_gap,
        "hands_usable_mean": float(np.mean([d.hands for d in D])) if D else 0.0,
        "wrist_motion_p75": wrist_p, "body_motion_median": body_med,
        "height_frac_median": float(np.median([d.hfrac for d in D])) if D else 0.0,
        "n_persons_max": int(max((d.n_persons for d in D), default=0)),
        "kept": kept, "drop_reason": reason,
        # Integrated geometry columns
        **geom,
    }
    kpt = {
        "segment_uid": uid, "video_id": video["video_id"],
        "track_frames": np.array(frames, dtype=np.int32),
        "n_detections": len(D), "kpt_layout": "coco17", "n_keypoints": 17,
        "kpt_x": np.concatenate([d.kxy[:, 0] for d in D]) if D else np.zeros(0, np.float32),
        "kpt_y": np.concatenate([d.kxy[:, 1] for d in D]) if D else np.zeros(0, np.float32),
        "kpt_conf": np.concatenate([d.kconf for d in D]) if D else np.zeros(0, np.float32),
    }
    return man, kpt


_SEG_NULLS = {
    "segment_uid": None, "seg_index": None, "start_frame": None, "end_frame": None,
    "n_frames": None, "n_windows": 0, "track_stride": None,
    "track_frames": None, "track_cx": None, "track_cy": None, "track_w": None,
    "track_h": None, "track_conf": None, "track_hands": None,
    "n_bridged_gaps": None, "max_gap_frames": None, "hands_usable_mean": None,
    "wrist_motion_p75": None, "body_motion_median": None,
    "height_frac_median": None, "n_persons_max": None,
    "bx": None, "by": None, "b_side": None, "body_side_px": None,
    "wrist_reach_px": None, "body_coverage_frac": None, "body_max_drift_px": None,
    "geom_valid": False, "geom_drop_reason": None,
    "kept": False, "drop_reason": "",
}


def process_one_video(path_str: str):
    cfg = _CFG
    path = Path(path_str)
    vid = path.stem
    size, md5 = parent_fingerprint(path)
    base = {
        "video_id": vid, "path": str(path),
        "parent_size_bytes": size, "parent_md5_16k": md5,
        "stage1_version": STAGE1_VERSION, "stage1_param_hash": cfg.param_hash(),
        "model_name": cfg.model_name,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "orig_fps": _ORIGFPS.get(vid),
    }

    meta = probe_video(path)
    if meta is None:
        return [{**base, **_SEG_NULLS, "width": None, "height": None, "fps": None,
                 "parent_n_frames": None, "status": "unreadable", "n_segments": 0,
                 "drop_reason": "unreadable"}], []
    base.update(width=meta["width"], height=meta["height"], fps=meta["fps"],
                parent_n_frames=meta["nb_frames"])

    if abs(meta["fps"] - TARGET_FPS) > FPS_TOL:
        return [{**base, **_SEG_NULLS, "status": "bad_fps", "n_segments": 0,
                 "drop_reason": f"fps={meta['fps']:.3f}"}], []

    try:
        valid, dets = analyze(path, meta, cfg)
    except Exception:
        LOGGER.error("analyze failed %s\n%s", path.name, traceback.format_exc())
        return [{**base, **_SEG_NULLS, "status": "analyze_failed", "n_segments": 0,
                 "drop_reason": "analyze_failed"}], []

    segs = raw_segments(valid, cfg.gap_tolerance)
    if not segs:
        return [{**base, **_SEG_NULLS, "status": "no_segments",
                 "n_segments": 0, "drop_reason": "no_valid_frames"}], []

    base.update(status="ok", n_segments=len(segs))
    man_rows, kpt_rows = [], []
    for i, (s, e) in enumerate(segs):
        m, k = segment_record(base, i, s, e, valid, dets, cfg)
        man_rows.append(m)
        kpt_rows.append(k)
    return man_rows, kpt_rows


# --------------------------------------------------------------------------- #
# Shard Writing & Finalization
# --------------------------------------------------------------------------- #
def write_shard(rows: list, out: Path, schema):
    import pyarrow as pa
    import pyarrow.parquet as pq
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, np.ndarray)).any():
            df[col] = df[col].map(lambda v: v.tolist() if isinstance(v, np.ndarray) else v)
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, out, compression="zstd")


def build_schemas():
    import pyarrow as pa
    man = pa.schema([
        ("segment_uid", pa.string()), ("video_id", pa.string()), ("seg_index", pa.int32()),
        ("path", pa.string()), ("parent_size_bytes", pa.int64()),
        ("parent_md5_16k", pa.string()), ("parent_n_frames", pa.int64()),
        ("start_frame", pa.int64()), ("end_frame", pa.int64()),
        ("n_frames", pa.int32()), ("n_windows", pa.int32()),
        ("track_stride", pa.int8()),
        ("track_frames", pa.list_(pa.int32())), ("track_cx", pa.list_(pa.float32())),
        ("track_cy", pa.list_(pa.float32())), ("track_w", pa.list_(pa.float32())),
        ("track_h", pa.list_(pa.float32())), ("track_conf", pa.list_(pa.float32())),
        ("track_hands", pa.list_(pa.int8())),
        ("n_bridged_gaps", pa.int32()), ("max_gap_frames", pa.int32()),
        ("hands_usable_mean", pa.float32()), ("wrist_motion_p75", pa.float32()),
        ("body_motion_median", pa.float32()), ("height_frac_median", pa.float32()),
        ("n_persons_max", pa.int32()),
        ("kept", pa.bool_()), ("drop_reason", pa.string()),
        ("width", pa.int32()), ("height", pa.int32()),
        ("fps", pa.float32()), ("orig_fps", pa.float32()),
        ("status", pa.string()), ("n_segments", pa.int32()),
        ("stage1_version", pa.string()), ("stage1_param_hash", pa.string()),
        ("model_name", pa.string()), ("created_utc", pa.string()),
        # Body Crop Geometry fields
        ("bx", pa.float32()), ("by", pa.float32()), ("b_side", pa.float32()),
        ("body_side_px", pa.float32()), ("wrist_reach_px", pa.float32()),
        ("body_coverage_frac", pa.float32()), ("body_max_drift_px", pa.float32()),
        ("geom_valid", pa.bool_()), ("geom_drop_reason", pa.string()),
    ])
    kpt = pa.schema([
        ("segment_uid", pa.string()), ("video_id", pa.string()),
        ("track_frames", pa.list_(pa.int32())), ("n_detections", pa.int32()),
        ("kpt_layout", pa.string()), ("n_keypoints", pa.int8()),
        ("kpt_x", pa.list_(pa.float32())), ("kpt_y", pa.list_(pa.float32())),
        ("kpt_conf", pa.list_(pa.float32())),
    ])
    return man, kpt


def finalize(cfg: Config):
    import pyarrow.parquet as pq
    import pyarrow as pa
    out = cfg.output_dir
    man_parts = sorted((out / "manifest_parts").glob("part_*.parquet"))
    kpt_parts = sorted((out / "keypoint_parts").glob("part_*.parquet"))
    if not man_parts:
        LOGGER.warning("no shards to finalize")
        return

    man = pa.concat_tables([pq.read_table(p) for p in man_parts])
    pq.write_table(man, out / "stage1_manifest.parquet", compression="zstd")
    if kpt_parts:
        kpt = pa.concat_tables([pq.read_table(p) for p in kpt_parts])
        pq.write_table(kpt, out / "stage1_keypoints.parquet", compression="zstd")

    df = man.to_pandas()

    def seg_str(g):
        parts = []
        for r in g.sort_values("seg_index").itertuples():
            if pd.isna(r.start_frame):
                continue
            mark = "" if r.kept else "*"
            parts.append(f"({int(r.start_frame)},{int(r.end_frame)}){mark}")
        return "[" + ",".join(parts) + "]"

    prev = (df.groupby("video_id", sort=True)
              .apply(lambda g: pd.Series({
                  "resolution": f"{int(g['width'].iloc[0])}x{int(g['height'].iloc[0])}"
                                if pd.notna(g["width"].iloc[0]) else "",
                  "fps": g["fps"].iloc[0],
                  "num_segments": int(g["n_segments"].iloc[0]),
                  "num_segments_kept": int(g["kept"].sum()),
                  "total_kept_frames": int(g.loc[g["kept"], "n_frames"].sum()),
                  "n_windows_total": int(g.loc[g["kept"], "n_windows"].sum()),
                  "segments": seg_str(g),
                  "status": g["status"].iloc[0],
              }), include_groups=False)
              .reset_index())
    prev.to_csv(out / "stage1_manifest_preview.csv", index=False)

    totals = {
        "videos_scanned": int(df["video_id"].nunique()),
        "videos_ok": int(df.loc[df["status"] == "ok", "video_id"].nunique()),
        "segments_total": int(df["segment_uid"].notna().sum()),
        "segments_kept": int(df["kept"].sum()),
        "windows_total": int(df.loc[df["kept"], "n_windows"].sum()),
        "hours_kept": round(float(df.loc[df["kept"], "n_frames"].sum()) / TARGET_FPS / 3600, 1),
    }
    (out / "stage1_run.json").write_text(json.dumps({
        "stage1_version": STAGE1_VERSION, "stage1_param_hash": cfg.param_hash(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_name": cfg.model_name,
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "coordinate_space": "source pixels, full parent resolution, top-left origin",
        "frame_index_space": "absolute 0-based frame index in 25.0fps parent",
        "totals": totals,
    }, indent=2))
    LOGGER.info("finalized: %s", json.dumps(totals))


# --------------------------------------------------------------------------- #
# Main Driver
# --------------------------------------------------------------------------- #
def load_orig_fps(report: Optional[Path]) -> dict:
    if not report or not report.exists():
        return {}
    out = {}
    for line in report.open():
        try:
            r = json.loads(line)
            if r.get("src_fps"):
                out[r["video_id"]] = float(r["src_fps"])
        except (json.JSONDecodeError, ValueError):
            continue
    return out


def _pool_init(cfg: Config, orig_fps: dict):
    import multiprocessing
    name = multiprocessing.current_process().name
    try:
        idx = int(name.rsplit("-", 1)[-1]) - 1
    except ValueError:
        idx = 0
    init_worker(cfg, idx, orig_fps)


def run(cfg: Config):
    out = cfg.output_dir
    (out / "manifest_parts").mkdir(parents=True, exist_ok=True)
    (out / "keypoint_parts").mkdir(parents=True, exist_ok=True)
    done_file = out / "_done_videos.txt"
    if cfg.overwrite:
        for p in (out / "manifest_parts").glob("part_*.parquet"):
            p.unlink()
        for p in (out / "keypoint_parts").glob("part_*.parquet"):
            p.unlink()
        done_file.unlink(missing_ok=True)

    done = set(done_file.read_text().split()) if done_file.exists() else set()
    videos = sorted(p for p in cfg.input_dir.rglob("*")
                    if p.suffix.lower() in VIDEO_EXTS and p.stem not in done)
    LOGGER.info("found %d videos to process (%d already done)", len(videos), len(done))

    orig_fps = load_orig_fps(cfg.normalize_report)
    man_schema, kpt_schema = build_schemas()
    shard_idx = len(list((out / "manifest_parts").glob("part_*.parquet")))
    buf_man, buf_kpt, buf_vids = [], [], []

    def flush():
        nonlocal shard_idx, buf_man, buf_kpt, buf_vids
        if not buf_man:
            return
        write_shard(buf_man, out / "manifest_parts" / f"part_{shard_idx:05d}.parquet",
                    man_schema)
        if buf_kpt:
            write_shard(buf_kpt, out / "keypoint_parts" / f"part_{shard_idx:05d}.parquet",
                        kpt_schema)
        with done_file.open("a") as f:
            f.write("".join(v + "\n" for v in buf_vids))
        shard_idx += 1
        buf_man, buf_kpt, buf_vids = [], [], []

    t0 = time.time()
    n_done = 0

    def consume(vid_stem: str, result):
        nonlocal n_done
        man_rows, kpt_rows = result
        buf_man.extend(man_rows)
        buf_kpt.extend(kpt_rows)
        buf_vids.append(vid_stem)
        n_done += 1
        if len(buf_vids) >= FLUSH_EVERY:
            flush()
        log_every = 5 if len(videos) <= 100 else 25
        if n_done % log_every == 0 or n_done == len(videos):
            rate = n_done / max(1e-6, time.time() - t0) * 60
            LOGGER.info("%d/%d videos (%.1f/min)", n_done, len(videos), rate)

    if videos:
        if cfg.num_workers <= 1:
            init_worker(cfg, 0, orig_fps)
            for v in videos:
                consume(v.stem, process_one_video(str(v)))
        else:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(cfg.num_workers, initializer=_pool_init,
                          initargs=(cfg, orig_fps)) as pool:
                for v, result in zip(videos, pool.imap(process_one_video,
                                                       [str(v) for v in videos],
                                                       chunksize=1)):
                    consume(v.stem, result)
        flush()

    finalize(cfg)
    LOGGER.info("done in %.1f min", (time.time() - t0) / 60)


# --------------------------------------------------------------------------- #
# CLI Parser
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> Config:
    ap = argparse.ArgumentParser(description="Stage 1: presenter detection, keypoints & integrated crop geometry",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model-name", default="yolov8s-pose.pt")
    ap.add_argument("--downscale-height", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--half", dest="half", action="store_true", default=True)
    ap.add_argument("--no-half", dest="half", action="store_false")
    ap.add_argument("--frame-stride", type=int, default=4)
    ap.add_argument("--reader-queue", type=int, default=16)
    ap.add_argument("--hwaccel", default="cuda")
    ap.add_argument("--person-conf", type=float, default=0.30)
    ap.add_argument("--min-person-height-frac", type=float, default=0.45)
    ap.add_argument("--keep-full-body", dest="exclude_full_body", action="store_false")
    ap.add_argument("--lower-body-conf", type=float, default=0.50)
    ap.add_argument("--kpt-conf", type=float, default=0.50)
    ap.add_argument("--prominence-ratio", type=float, default=1.6)
    ap.add_argument("--min-visible-hands", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--hand-extent-ratio", type=float, default=0.75)
    ap.add_argument("--hand-fallback-frac", type=float, default=0.11)
    ap.add_argument("--hand-frame-margin", type=float, default=0.01)
    ap.add_argument("--min-segment-frames", type=int, default=104)
    ap.add_argument("--gap-tolerance", type=int, default=15)
    ap.add_argument("--motion-min", type=float, default=0.0010)
    ap.add_argument("--hand-motion-min", type=float, default=0.0050)
    ap.add_argument("--hand-motion-pctl", type=float, default=75.0)

    # Crop geometry hyper-parameters
    ap.add_argument("--crop-scale", type=float, default=1.8)
    ap.add_argument("--smooth-span-frac", type=float, default=1.0)
    ap.add_argument("--smooth-span-frames", type=int, default=25)
    ap.add_argument("--hand-margin-frac", type=float, default=0.10)
    ap.add_argument("--wrist-conf-min", type=float, default=0.30)
    ap.add_argument("--wrist-percentile", type=float, default=95.0)
    ap.add_argument("--min-crop-px", type=int, default=128)

    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--normalize-report", type=Path, default=None)
    a = ap.parse_args(argv)
    d = vars(a)
    return Config(**d)


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(processName)s] %(levelname)s %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()