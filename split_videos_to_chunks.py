## Create 60 seconds chunk videos from input video files.
## 60 sec chunks can be used to generate GT

import os
import re
import subprocess
from pathlib import Path

# === CONFIGURATION ===
# INPUT_ROOT = "/mnt/ssd_2T/users/tester/Videol/Data/May-2025/ATM-Normal" 
#INPUT_ROOT = "/mnt/ssd_2T/users/tester/Videol/Data/May-2025/ATM-Vandalism" 
INPUT_ROOT = "/mnt/ssd_2T/users/tester/Videol/Data/Aug-2025/Normal" 
#INPUT_ROOT = "/mnt/ssd_2T/users/tester/Videol/Data/Aug-2025/Vandalism" 
OUTPUT_ROOT = "/mnt/ssd_2T/users/tester/Videol/Data/ATMGT"
CHUNK_DURATION = 60  # seconds per chunk
OUTPUT_PREFIX = "video_"  # prefix for output chunk names


def get_rel_from_month_folder(path: str) -> str:
    """
    Extract relative path starting from Month-Year folder (e.g., May-2025/ATM-Normal/...).
    """
    parts = Path(path).parts
    pattern = re.compile(r"^[A-Za-z]{3,9}-\d{4}$")
    for i, part in enumerate(parts):
        if pattern.match(part):
            return os.path.join(*parts[i:])
    return os.path.basename(path)


def split_video_incremental(video_path: str, output_dir: str, start_counter: int, duration: int = 60):
    """
    Split the video into chunks with incremental numbering and prefix.
    Returns the next counter after processing this video.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ffmpeg segment output (temporary numeric names)
    temp_pattern = os.path.join(output_dir, "temp_%04d.mp4")
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-c", "copy",
        "-map", "0",
        "-f", "segment",
        "-segment_time", str(duration),
        "-reset_timestamps", "1",
        temp_pattern
    ]

    try:
        subprocess.run(cmd, check=True)

        # Rename temp chunks to video_XXXX.mp4 using incremental counter
        temp_chunks = sorted([f for f in os.listdir(output_dir) if f.startswith("temp_") and f.endswith(".mp4")])
        counter = start_counter
        for temp_chunk in temp_chunks:
            new_name = f"{OUTPUT_PREFIX}{counter:04d}.mp4"
            os.rename(os.path.join(output_dir, temp_chunk), os.path.join(output_dir, new_name))
            counter += 1

        print(f"✅ Processed {video_path} into {len(temp_chunks)} chunks, next counter: {counter}")
        return counter

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to split {video_path} — {e}")
        return start_counter


def process_all_videos(input_root: str, output_root: str, chunk_duration: int = 60):
    """
    Walk all videos and split into chunks with prefix and incremental numbering.
    """
    counter_map = {}  # next counter per output directory

    for root, _, files in os.walk(input_root):
        for file in files:
            if not file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            video_path = os.path.join(root, file)
            rel_path = get_rel_from_month_folder(video_path)
            rel_dir = os.path.dirname(rel_path)
            output_dir = os.path.join(output_root, rel_dir)
            os.makedirs(output_dir, exist_ok=True)

            # Get starting counter for this folder
            start_counter = counter_map.get(output_dir, 1)
            next_counter = split_video_incremental(video_path, output_dir, start_counter, duration=chunk_duration)
            counter_map[output_dir] = next_counter


if __name__ == "__main__":
    print(f"🔍 Scanning for videos in: {INPUT_ROOT}")
    process_all_videos(INPUT_ROOT, OUTPUT_ROOT, CHUNK_DURATION)
    print(f"\n✅ All videos processed. Output saved under: {OUTPUT_ROOT}")
