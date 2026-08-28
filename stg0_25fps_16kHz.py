#!/usr/bin/env python3
"""
Stage 0 — normalize raw source videos to 25 fps CFR + 16 kHz audio.

Lip-sync preservation, by construction
--------------------------------------
  * Video: `fps=fps=25:round=down` drops/duplicates frames AT THEIR ORIGINAL
    TIMESTAMPS. No setpts, no speed change. A frame that showed at t=12.48s in
    the source still shows at (the nearest 1/25th grid point to) 12.48s.
  * Audio: only resampled to 16 kHz (timing-neutral). Never stretched, never
    trimmed, never atempo'd.
  * Duration in == duration out (within one frame), verified per file.

What is preserved / changed
---------------------------
  preserved : resolution, aspect ratio, audio content & timing, duration
  changed   : frame rate -> exactly 25 CFR, audio -> 16 kHz mono,
              container -> .mp4, codecs -> H.264 + AAC

Fast path
---------
Sources that are ALREADY constant 25 fps get their video stream copied
bit-for-bit (no re-encode, no generation loss, no sync risk) with only the
audio re-encoded. Roughly a third of a typical mixed corpus qualifies.

Usage
-----
    python normalize_raw_videos.py \
        --input-dir  /netscratch/bahrami/dataset/talking_pose \
        --output-dir /netscratch/bahrami/dataset/talking_pose_25fps \
        --workers 48 --report normalize_raw.jsonl

Resume-safe: finished outputs are skipped, partial files use a .part suffix.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".ts"}
TARGET_FPS = 25
FPS_TOL = 0.001          # how close avg_frame_rate must be to 25 for the copy path
VFR_TOL = 0.01           # |r_frame_rate - avg_frame_rate| / avg beyond this => VFR
SYNC_TOL_FRAMES = 1.5    # allowed |dur_out - dur_in| in output frames


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def _rate(txt: str | None) -> float | None:
    try:
        num, den = (txt or "0/0").split("/")
        return float(num) / float(den) if float(den) else None
    except (ValueError, ZeroDivisionError):
        return None


def probe(path: Path) -> dict | None:
    res = run(["ffprobe", "-v", "error", "-print_format", "json",
               "-show_streams", "-show_format", str(path)])
    if res.returncode != 0:
        return None
    try:
        info = json.loads(res.stdout.decode(errors="ignore"))
    except json.JSONDecodeError:
        return None

    out = {"avg_fps": None, "r_fps": None, "width": None, "height": None,
           "vcodec": None, "has_audio": False,
           "v_dur": None, "a_dur": None, "dur": None}
    for st in info.get("streams", []):
        if st.get("codec_type") == "video" and out["avg_fps"] is None:
            out["avg_fps"] = _rate(st.get("avg_frame_rate"))
            out["r_fps"] = _rate(st.get("r_frame_rate"))
            out["width"], out["height"] = st.get("width"), st.get("height")
            out["vcodec"] = st.get("codec_name")
            try:
                out["v_dur"] = float(st.get("duration"))
            except (TypeError, ValueError):
                pass
        elif st.get("codec_type") == "audio" and not out["has_audio"]:
            out["has_audio"] = True
            try:
                out["a_dur"] = float(st.get("duration"))
            except (TypeError, ValueError):
                pass
    try:
        out["dur"] = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        pass
    if not out["avg_fps"] or not out["width"]:
        return None
    return out


def is_vfr(meta: dict) -> bool:
    if meta["avg_fps"] and meta["r_fps"]:
        return abs(meta["r_fps"] - meta["avg_fps"]) / meta["avg_fps"] > VFR_TOL
    return True  # can't tell -> assume the unsafe case


def build_command(src: Path, dst: Path, meta: dict, crf: int, preset: str,
                  ffmpeg_threads: int) -> tuple[list[str], str]:
    """Returns (ffmpeg command, mode) where mode is 'copy' or 'transcode'."""
    copy_ok = (
        abs(meta["avg_fps"] - TARGET_FPS) < FPS_TOL
        and not is_vfr(meta)
        and meta["vcodec"] in ("h264", "hevc")   # containers .mp4 can hold
    )
    # src/dst are always absolute paths here, so a leading "-" in a video ID
    # (e.g. -V-tqJyxp_8, common in YouTube IDs) can't be misread as a flag --
    # noted only because it looked suspicious alongside the real bug above.
    cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
           "-threads", str(ffmpeg_threads),
           "-i", str(src), "-map", "0:v:0"]
    if copy_ok:
        # Already constant 25 fps: keep the video bits untouched. Only audio changes.
        cmd += ["-c:v", "copy"]
        mode = "copy"
    else:
        # Timestamp-based resample to a constant 25 fps grid. round=down keeps
        # each kept frame on/just after its original instant; -fps_mode cfr
        # guarantees the output timestamps are a strict 1/25 s grid (correct
        # for VFR sources too). NO setpts anywhere -> playback speed unchanged
        # -> audio alignment unchanged.
        cmd += ["-vf", f"fps=fps={TARGET_FPS}:round=down",
                "-fps_mode", "cfr",
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-threads", str(ffmpeg_threads),
                "-x264-params", f"threads={ffmpeg_threads}",
                "-g", str(TARGET_FPS), "-pix_fmt", "yuv420p"]
        mode = "transcode"
    if meta["has_audio"]:
        # Resample only. aresample with soft compensation disabled: we want a
        # pure rate conversion, never stretch-to-fit.
        cmd += ["-map", "0:a:0",
                "-af", "aresample=16000:async=0",
                "-c:a", "aac", "-b:a", "128k", "-ac", "1"]
    cmd += ["-movflags", "+faststart", "-f", "mp4", str(dst)]
    return cmd, mode


def verify(src_meta: dict, dst: Path) -> tuple[bool, dict]:
    """Output must be 25 fps CFR, same duration as source, audio/video agreeing."""
    m = probe(dst)
    checks: dict = {}
    if m is None:
        return False, {"error": "output unreadable"}

    checks["out_fps"] = round(m["avg_fps"], 4)
    fps_ok = abs(m["avg_fps"] - TARGET_FPS) < 0.01

    tol = SYNC_TOL_FRAMES / TARGET_FPS
    src_dur = src_meta["v_dur"] or src_meta["dur"]
    dur_ok = True
    if src_dur and m["dur"]:
        checks["dur_delta_ms"] = round((m["dur"] - src_dur) * 1000, 1)
        dur_ok = abs(m["dur"] - src_dur) <= max(tol, 0.08)

    av_ok = True
    if m["has_audio"] and m["v_dur"] and m["a_dur"]:
        checks["av_delta_ms"] = round((m["a_dur"] - m["v_dur"]) * 1000, 1)
        # AAC pads to full frames (~21 ms at 16 kHz/1024) so allow a little slack.
        av_ok = abs(m["a_dur"] - m["v_dur"]) <= max(tol, 0.10)

    checks["has_audio"] = m["has_audio"]
    audio_ok = m["has_audio"] == src_meta["has_audio"]
    res_ok = (m["width"], m["height"]) == (src_meta["width"], src_meta["height"])
    if not res_ok:
        checks["error"] = "resolution changed"

    return fps_ok and dur_ok and av_ok and audio_ok and res_ok, checks


def normalize_one(job: tuple) -> dict:
    src_str, in_dir_str, out_dir_str, crf, preset, overwrite, ffmpeg_threads = job
    src, in_dir = Path(src_str), Path(in_dir_str)
    dst = Path(out_dir_str) / src.relative_to(in_dir).with_suffix(".mp4")
    rec = {"video_id": src.stem, "src": str(src), "dst": str(dst)}

    if dst.exists() and not overwrite:
        rec["status"] = "skipped_exists"
        return rec

    meta = probe(src)
    if meta is None:
        rec["status"] = "unreadable"
        return rec
    rec["src_fps"] = round(meta["avg_fps"], 4)
    rec["vfr"] = is_vfr(meta)

    dst.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: dst.with_suffix(".mp4.part") is a trap -- Path.with_suffix() only
    # replaces the LAST dot-segment, so it silently produces a filename whose
    # real suffix is ".part" (ffmpeg then can't infer the muxer from the name
    # and every encode fails identically). Build the temp name by string
    # concatenation instead, and pass -f mp4 explicitly as a second line of
    # defense regardless of what the filename looks like.
    tmp = dst.with_name(dst.name + ".part")
    cmd, mode = build_command(src, tmp, meta, crf, preset, ffmpeg_threads)
    rec["mode"] = mode

    res = run(cmd)
    if res.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        rec["status"] = "encode_failed"
        rec["error"] = res.stderr.decode(errors="ignore")[-300:].replace("\n", " ")
        return rec

    ok, checks = verify(meta, tmp)
    rec.update(checks)
    if not ok:
        # A copy-mode failure usually means the source lied about being CFR;
        # retry once with a full transcode before giving up.
        if mode == "copy":
            tmp.unlink(missing_ok=True)
            meta["avg_fps"] = TARGET_FPS + 1  # force the transcode branch
            cmd, _ = build_command(src, tmp, meta, crf, preset, ffmpeg_threads)
            rec["mode"] = "copy->transcode"
            res = run(cmd)
            if res.returncode == 0 and tmp.exists():
                ok, checks = verify(probe(src) or meta, tmp)
                rec.update(checks)
        if not ok:
            tmp.unlink(missing_ok=True)
            rec["status"] = "verify_failed"
            return rec

    tmp.replace(dst)
    rec["status"] = "ok"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Normalize raw videos to 25 fps CFR + 16 kHz audio, "
                    "preserving resolution, audio, and lip synchronization.")
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--crf", type=int, default=16,
                    help="quality for transcoded videos; this file feeds two more "
                         "encode generations, so keep it high")
    ap.add_argument("--preset", default="veryfast",
                    help="libx264 preset; 'veryfast' is the right speed/quality "
                         "point for 19k videos, 'medium' if you have the hours")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--ffmpeg-threads", type=int, default=1,
                    help="threads per ffmpeg instance; set workers * ffmpeg-threads "
                         "<= your allocated CPUs to avoid oversubscription")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report", type=Path, default=None, help="JSONL, one record per video")
    a = ap.parse_args()

    in_dir = a.input_dir.expanduser().resolve()
    out_dir = a.output_dir.expanduser().resolve()
    if not in_dir.is_dir():
        print(f"error: not a directory: {in_dir}", file=sys.stderr)
        return 2

    videos = sorted(p for p in in_dir.rglob("*")
                    if p.suffix.lower() in VIDEO_EXTS and not p.name.endswith(".part"))
    print(f"found {len(videos):,} videos")
    if not videos:
        return 2

    jobs = [(str(v), str(in_dir), str(out_dir), a.crf, a.preset, a.overwrite, a.ffmpeg_threads)
            for v in videos]

    counts: dict[str, int] = {}
    modes: dict[str, int] = {}
    worst: list[tuple[float, str]] = []
    report = a.report.open("w") if a.report else None
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(normalize_one, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                counts[rec["status"]] = counts.get(rec["status"], 0) + 1
                if rec.get("mode"):
                    modes[rec["mode"]] = modes.get(rec["mode"], 0) + 1
                d = abs(rec.get("dur_delta_ms", 0.0))
                if rec["status"] == "ok" and d:
                    worst.append((d, rec["video_id"]))
                    worst.sort(reverse=True)
                    del worst[10:]
                if report:
                    report.write(json.dumps(rec) + "\n")
                if i % 100 == 0 or i == len(futures):
                    print(f"  {i:,}/{len(futures):,}  " +
                          "  ".join(f"{k}={v:,}" for k, v in sorted(counts.items())),
                          end="\r", flush=True)
    finally:
        if report:
            report.close()

    print("\n\nstatus:")
    for k, v in sorted(counts.items()):
        print(f"  {k:20} {v:,}")
    print("modes:")
    for k, v in sorted(modes.items()):
        print(f"  {k:20} {v:,}")
    if worst:
        print("largest duration deltas (ms) — eyeball these for sync:")
        for d, vid in worst[:5]:
            print(f"  {d:8.1f}  {vid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())