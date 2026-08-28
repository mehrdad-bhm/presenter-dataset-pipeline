import argparse
import csv
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


YOUTUBE_ID_RE = re.compile(r"(?<![A-Za-z0-9_-])([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])")
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
DEFAULT_CSV_FILE = "path to the csv file"
DEFAULT_TARGET_DIR = "path to the target directory"
## the item below is just for saving those ids where it could inly find low quality streams, so that we don't try to download them again
DEFAULT_LOW_QUALITY_CSV = "path to the low quality files csv"
DEFAULT_MIN_SLEEP_SECONDS = 5
DEFAULT_MAX_SLEEP_SECONDS = 10

# Resolutions to try first, in priority order. When neither is available, any
# resolution above 720p is accepted as a fallback (closest to 720p first).
PREFERRED_RESOLUTIONS = ["1080p", "720p"]
FALLBACK_MIN_HEIGHT = 720


def _get_env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError:
        print(f"Invalid {name}={raw!r}; using default {default}.")
        return default

    if value < minimum:
        print(f"Invalid {name}={raw!r}; must be >= {minimum}. Using default {default}.")
        return default

    return value


def get_existing_video_ids(target_directory: str) -> set[str]:
    """
    Detect completed videos by a YouTube ID anywhere in a media filename.
    Partial yt-dlp downloads (for example, ``.mp4.part``) are intentionally ignored.
    """
    existing_ids: set[str] = set()

    if not os.path.isdir(target_directory):
        return existing_ids

    for entry in os.scandir(target_directory):
        if not entry.is_file() or Path(entry.name).suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        match = YOUTUBE_ID_RE.search(entry.name)
        if match and entry.stat().st_size > 0:
            existing_ids.add(match.group(1))

    return existing_ids


def read_archive(archive_path: Path) -> set[str]:
    if not archive_path.exists():
        return set()

    with archive_path.open("r", encoding="utf-8") as archive_file:
        return {line.strip() for line in archive_file if line.strip()}


def append_to_archive(archive_path: Path, video_id: str) -> None:
    with archive_path.open("a", encoding="utf-8") as archive_file:
        archive_file.write(f"{video_id}\n")


def read_low_quality_ids(low_quality_csv: Path) -> set[str]:
    if not low_quality_csv.exists():
        return set()

    with low_quality_csv.open("r", encoding="utf-8") as file_handle:
        reader = csv.reader(file_handle)
        return {row[0].strip() for row in reader if row and row[0].strip()}


def append_low_quality_id(low_quality_csv: Path, video_id: str, best_resolution: Optional[str]) -> None:
    is_new_file = not low_quality_csv.exists()
    with low_quality_csv.open("a", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        if is_new_file:
            writer.writerow(["video_id", "best_available_resolution"])
        writer.writerow([video_id, best_resolution or "none"])


def read_video_ids_from_csv(csv_file: str) -> list[str]:
    video_ids: list[str] = []

    with open(csv_file, "r", encoding="utf-8") as file_handle:
        reader = csv.reader(file_handle)

        for row in reader:
            if not row:
                continue

            video_id = row[0].strip()
            if video_id and len(video_id) == 11:
                video_ids.append(video_id)

    return video_ids


def iter_unique_video_ids(video_ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_ids: list[str] = []

    for video_id in video_ids:
        if video_id in seen:
            continue
        seen.add(video_id)
        unique_ids.append(video_id)

    return unique_ids


def get_video_info(video_id: str, timeout: int, max_retries: int) -> dict[str, Any]:
    """Fetch metadata once so availability checks do not require an extra download request."""
    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": timeout,
        "retries": max_retries,
    }
    with YoutubeDL(options) as downloader:
        return downloader.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)


def has_video_at_resolution(info: dict[str, Any], height: int) -> bool:
    return any(
        stream.get("vcodec") != "none" and stream.get("height") == height
        for stream in info.get("formats", [])
    )


def get_best_available_resolution(info: dict[str, Any]) -> Optional[str]:
    """Used only for logging: report the best video height that yt-dlp found."""
    heights = [
        stream["height"]
        for stream in info.get("formats", [])
        if stream.get("vcodec") != "none" and isinstance(stream.get("height"), int)
    ]
    return f"{max(heights)}p" if heights else None


def resolutions_to_try(info: dict[str, Any]) -> list[str]:
    """
    Preferred resolutions first (1080p, then 720p). If neither is available the
    remaining heights above 720p are appended, closest to 720p first, so the
    smallest acceptable file is downloaded.
    """
    preferred_heights = [int(item.removesuffix("p")) for item in PREFERRED_RESOLUTIONS]
    available_heights = sorted(
        {
            stream["height"]
            for stream in info.get("formats", [])
            if stream.get("vcodec") != "none" and isinstance(stream.get("height"), int)
        }
    )
    fallback_heights = [
        height
        for height in available_heights
        if height > FALLBACK_MIN_HEIGHT and height not in preferred_heights
    ]
    return PREFERRED_RESOLUTIONS + [f"{height}p" for height in fallback_heights]


def try_download_at_resolution(
    info: dict[str, Any], video_id: str, resolution: str, target_directory: str, timeout: int, max_retries: int
) -> bool:
    """Download the best video at exactly ``resolution`` plus audio, merged as MP4 when needed."""
    height = int(resolution.removesuffix("p"))
    if not has_video_at_resolution(info, height):
        return False

    print(f"[{video_id}] Found a {resolution} stream. Downloading with yt-dlp.")
    options = {
        "format": f"bestvideo[height={height}]+bestaudio/best[height={height}]",
        "outtmpl": str(Path(target_directory) / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "socket_timeout": timeout,
        "retries": max_retries,
        "continuedl": True,
        "overwrites": False,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(options) as downloader:
        downloader.process_ie_result(info, download=True)
    return True


def download_video(
    video_id: str, target_directory: str, archive_path: Path, low_quality_csv: Path
) -> str:
    """
    Returns one of: "downloaded", "low_quality", "failed".
    """
    timeout = _get_env_int("PYTUBE_TIMEOUT", 30, minimum=1)
    max_retries = _get_env_int("PYTUBE_MAX_RETRIES", 2, minimum=0)
    info = get_video_info(video_id, timeout, max_retries)

    for resolution in resolutions_to_try(info):
        if try_download_at_resolution(info, video_id, resolution, target_directory, timeout, max_retries):
            append_to_archive(archive_path, video_id)
            if resolution in PREFERRED_RESOLUTIONS:
                print(f"[{video_id}] Saved at {resolution}.")
            else:
                print(f"[{video_id}] Saved at fallback resolution {resolution}.")
            return "downloaded"

    # Nothing at 1080p/720p and nothing above 720p either.
    best_resolution = get_best_available_resolution(info)
    append_low_quality_id(low_quality_csv, video_id, best_resolution)
    print(
        f"[{video_id}] No stream at 1080p/720p or above 720p "
        f"(best found: {best_resolution or 'unknown'}). Logged to {low_quality_csv}."
    )
    return "low_quality"


def download_videos_from_csv(
    csv_file: str,
    target_directory: str,
    low_quality_csv: str = DEFAULT_LOW_QUALITY_CSV,
    min_sleep_seconds: float = DEFAULT_MIN_SLEEP_SECONDS,
    max_sleep_seconds: float = DEFAULT_MAX_SLEEP_SECONDS,
) -> None:
    """
    Read video IDs from a CSV and download them with yt-dlp, preferring 1080p then 720p
    and falling back to any resolution above 720p. IDs with no acceptable stream are
    logged to `low_quality_csv` instead of downloaded.
    """
    os.makedirs(target_directory, exist_ok=True)

    try:
        raw_video_ids = read_video_ids_from_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file {csv_file} was not found.")
        return

    video_ids = iter_unique_video_ids(raw_video_ids)
    existing_ids = get_existing_video_ids(target_directory)
    archive_path = Path(target_directory) / ".download_archive.txt"
    archived_ids = read_archive(archive_path)

    low_quality_path = Path(low_quality_csv)
    if not low_quality_path.is_absolute():
        low_quality_path = Path(target_directory) / low_quality_csv
    already_low_quality_ids = read_low_quality_ids(low_quality_path)

    queued_ids: list[str] = []
    skipped_existing = 0
    skipped_archived = 0
    skipped_low_quality = 0

    for video_id in video_ids:
        if video_id in existing_ids:
            skipped_existing += 1
            print(f"Skipping already downloaded video: {video_id}")
            continue

        if video_id in archived_ids:
            skipped_archived += 1
            print(f"Skipping archived video: {video_id}")
            continue

        if video_id in already_low_quality_ids:
            skipped_low_quality += 1
            print(f"Skipping previously logged low-quality video: {video_id}")
            continue

        queued_ids.append(video_id)

    print(f"Found {len(raw_video_ids)} rows in CSV.")
    print(f"Queued {len(queued_ids)} unique videos for download.")
    print(f"Skipped {skipped_existing} videos already present in {target_directory}.")
    print(f"Skipped {skipped_archived} videos already recorded in {archive_path}.")
    print(f"Skipped {skipped_low_quality} videos already recorded in {low_quality_path}.")

    if not queued_ids:
        print("Nothing left to process.")
        return

    successes = 0
    failures = 0
    low_quality_count = 0

    for index, video_id in enumerate(queued_ids, start=1):
        if index > 1 and max_sleep_seconds > 0:
            sleep_seconds = random.uniform(min_sleep_seconds, max_sleep_seconds)
            print(
                f"Sleeping {sleep_seconds:.1f}s before the next video request "
                f"(random range: {min_sleep_seconds:g}-{max_sleep_seconds:g}s)."
            )
            time.sleep(sleep_seconds)

        print(f"\n[{index}/{len(queued_ids)}] Processing {video_id}")
        try:
            result = download_video(video_id, target_directory, archive_path, low_quality_path)
            if result == "downloaded":
                successes += 1
            elif result == "low_quality":
                low_quality_count += 1
            else:
                failures += 1
        except (DownloadError, OSError, KeyError, ValueError) as exc:
            failures += 1
            print(f"[{video_id}] Failed: {exc}")

    print(
        f"\nFinished. Downloaded (1080p/720p preferred, >720p fallback): {successes}. "
        f"Logged as low quality: {low_quality_count}. Failed: {failures}."
    )
    if low_quality_count:
        print(f"Low-quality video IDs written to: {low_quality_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download YouTube videos from a CSV of video IDs using yt-dlp, preferring 1080p "
            "then 720p and falling back to any resolution above 720p. Videos with no "
            "acceptable stream are logged to a separate CSV."
        )
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=DEFAULT_CSV_FILE,
        help=f"Path to the CSV file containing YouTube IDs. Default: {DEFAULT_CSV_FILE}",
    )
    parser.add_argument(
        "target_directory",
        nargs="?",
        default=DEFAULT_TARGET_DIR,
        help=f"Directory where downloaded videos will be saved. Default: {DEFAULT_TARGET_DIR}",
    )
    parser.add_argument(
        "--low-quality-csv",
        default=DEFAULT_LOW_QUALITY_CSV,
        help=(
            "Path (or filename, relative to target_directory) for the CSV listing video IDs "
            f"that had no stream at 720p or above. Default: {DEFAULT_LOW_QUALITY_CSV}"
        ),
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=None,
        help=(
            "Use a fixed delay between video requests. Overrides the random delay range. "
            "The PYTUBE_SLEEP_SECONDS environment variable is also supported."
        ),
    )
    parser.add_argument(
        "--min-sleep-seconds",
        type=float,
        default=DEFAULT_MIN_SLEEP_SECONDS,
        help=f"Minimum random delay between video requests. Default: {DEFAULT_MIN_SLEEP_SECONDS}",
    )
    parser.add_argument(
        "--max-sleep-seconds",
        type=float,
        default=DEFAULT_MAX_SLEEP_SECONDS,
        help=f"Maximum random delay between video requests. Default: {DEFAULT_MAX_SLEEP_SECONDS}",
    )
    args = parser.parse_args()

    if args.sleep_seconds is None and os.getenv("PYTUBE_SLEEP_SECONDS") is not None:
        args.sleep_seconds = _get_env_int("PYTUBE_SLEEP_SECONDS", 0, minimum=0)

    if args.sleep_seconds is not None:
        if args.sleep_seconds < 0:
            parser.error("--sleep-seconds must be >= 0")
        args.min_sleep_seconds = args.sleep_seconds
        args.max_sleep_seconds = args.sleep_seconds
    elif args.min_sleep_seconds < 0 or args.max_sleep_seconds < 0:
        parser.error("sleep delays must be >= 0")
    elif args.min_sleep_seconds > args.max_sleep_seconds:
        parser.error("--min-sleep-seconds cannot exceed --max-sleep-seconds")

    return args


def main() -> None:
    args = parse_args()
    download_videos_from_csv(
        args.csv_file,
        args.target_directory,
        low_quality_csv=args.low_quality_csv,
        min_sleep_seconds=args.min_sleep_seconds,
        max_sleep_seconds=args.max_sleep_seconds,
    )


if __name__ == "__main__":
    main()
