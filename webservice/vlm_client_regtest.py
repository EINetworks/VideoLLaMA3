import os
import time
import requests
import logging
import argparse
from glob import glob
import subprocess

PROC_FPS = 0.5
PROC_MAX_FRAMES = 30

# JETSON_IP = "192.168.1.197"
JETSON_IP = "127.0.0.1"
BASE_URL = f"http://{JETSON_IP}:5010/inference"

# Setup logging
logging.basicConfig(
    filename="results_client.txt",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def call_inference(video_path, fps=1, max_frames=30, turn="first", prev_response=None, single_api=False):
    if single_api:
        url = f"{BASE_URL}/both_turns"
        print(f"[INFO] Sending single-API inference request for {video_path}...")
        with open(video_path, "rb") as f:
            files = {"video": f}
            data = {"fps": fps, "max_frames": max_frames}
            start_time = time.time()
            response = requests.post(url, files=files, data=data)
            elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            print(f"[INFO] Received single-API result in {elapsed:.3f} sec: {result}")
            return result
        else:
            print(f"[ERROR] Single-API inference failed in {elapsed:.3f} sec: {response.text}")
            return None
    else:
        url = f"{BASE_URL}/{turn}"
        print(f"[INFO] Sending {turn}-turn inference request for {video_path}...")

        with open(video_path, "rb") as f:
            files = {"video": f}
            data = {"fps": fps, "max_frames": max_frames}
            if turn == "second" and prev_response:
                data["prev_response"] = prev_response
            start_time = time.time()
            response = requests.post(url, files=files, data=data)
            elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            print(f"[INFO] Received {turn}-turn result in {elapsed:.3f} sec: {result}")
            return result
        else:
            print(f"[ERROR] {turn}-turn inference failed in {elapsed:.3f} sec: {response.text}")
            return None


def split_video_into_chunks(video_path, chunk_seconds, output_dir="chunks"):
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = int(float(result.stdout.strip()))
    duration = min(duration, 240)
    print(f"duration: {duration}, chunk_seconds {chunk_seconds}")

    parts = []
    chunk_idx = 0

    for start in range(0, duration, chunk_seconds):
        end = min(start + chunk_seconds, duration)
        chunk_filename = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(video_path))[0]}_chunk{chunk_idx}.mp4"
        )

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-ss", str(start),
            "-t", str(end - start),
            "-c:v", "libx264",
            "-an",
            chunk_filename
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        parts.append(chunk_filename)
        chunk_idx += 1

    return parts


def classify_response(response_text):
    if not response_text:
        return "Unknown"
    if response_text.lower().startswith("normal"):
        return "Normal"
    elif response_text.lower().startswith("suspicious"):
        return "Suspicious"
    return "Unknown"


def safe_get_response(result: dict, turn: str = "") -> str | None:
    if not result:
        logging.error(f"[{turn}] Empty result from server")
        return None
    if "response" in result:
        return result["response"]
    if "error" in result:
        logging.error(f"[{turn}] Server returned error: {result['error']}")
        return None
    logging.error(f"[{turn}] Unexpected result format: {result}")
    return None


def get_video_resolution(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    res = result.stdout.strip()
    return res if res else "Unknown"


def process_video(video_path, multi_turn=False, chunk_seconds=None, single_api=True):
    start_time = time.time()
    file_decision = "Normal"
    chunk_count = 0
    responses = []

    resolution = get_video_resolution(video_path)
    print(f"\n--- START OF VIDEO: {video_path} | Resolution: {resolution} ---")
    logging.info(f"--- START OF VIDEO: {video_path} | Resolution: {resolution} ---")

    parts = split_video_into_chunks(video_path, chunk_seconds) if chunk_seconds else [video_path]

    for idx, part in enumerate(parts):
        chunk_count += 1
        print(f"[INFO] Processing chunk {idx+1}/{len(parts)}: {part}")
        logging.info(f"Processing chunk {idx+1}/{len(parts)}: {part}")

        if single_api:
            res_both = call_inference(part, fps=PROC_FPS, max_frames=PROC_MAX_FRAMES, single_api=True)
            if res_both:
                first_resp = res_both.get("first_turn")
                second_resp = res_both.get("second_turn")
                if first_resp:
                    responses.append(("First", first_resp))
                    logging.info(f"First Response: {first_resp}")
                if second_resp:
                    responses.append(("Second", second_resp))
                    logging.info(f"Second Response: {second_resp}")
                final_resp = second_resp or first_resp
            else:
                final_resp = "Unknown"

        else:
            res_first = call_inference(part, fps=PROC_FPS, max_frames=PROC_MAX_FRAMES, turn="first")
            first_resp = safe_get_response(res_first, "First") if res_first else None
            if first_resp:
                responses.append(("First", first_resp))
                logging.info(f"First Response: {first_resp}")

            if multi_turn and first_resp:
                res_second = call_inference(
                    part,
                    fps=PROC_FPS,
                    max_frames=PROC_MAX_FRAMES,
                    turn="second",
                    prev_response=first_resp
                )
                second_resp = safe_get_response(res_second, "Second") if res_second else None
                if second_resp:
                    responses.append(("Second", second_resp))
                    logging.info(f"Second Response: {second_resp}")
                final_resp = second_resp or first_resp
            else:
                final_resp = first_resp or "Unknown"

        label = classify_response(final_resp)
        print(f"[RESULT] Chunk {idx+1} Decision: {label}")
        logging.info(f"Chunk {idx+1} Decision: {label}")

        if label == "Suspicious":
            file_decision = "Suspicious"

    proc_time = time.time() - start_time
    logging.info(f"Processing Time: {proc_time:.3f} sec")
    logging.info(f"--- END OF VIDEO: {video_path} ---\n")

    print(f"[INFO] Video completed in {proc_time:.3f} sec")
    print(f"[SUMMARY] Video Decision: {file_decision} | Chunks processed: {chunk_count}")
    print(f"--- END OF VIDEO: {video_path} ---\n")

    return file_decision, chunk_count


def process_folder(video_folder, multi_turn=False, chunk_seconds=None, single_api=True):
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob(os.path.join(video_folder, "**", ext), recursive=True))

    if not video_files:
        print("No videos found.")
        return

    print(f"Found {len(video_files)} videos in {video_folder}")
    logging.info(f"Found {len(video_files)} videos in {video_folder}")

    total_normal_clips = 0
    total_abnormal_clips = 0
    total_chunks = 0

    for video_path in video_files:
        file_decision, chunks = process_video(
            video_path,
            multi_turn=multi_turn,
            chunk_seconds=chunk_seconds,
            single_api=single_api
        )
        total_chunks += chunks

        if file_decision == "Normal":
            total_normal_clips += 1
        elif file_decision == "Suspicious":
            total_abnormal_clips += 1
        print(f"Normal: {total_normal_clips}, Suspicious {total_abnormal_clips}")

    total_clips = total_normal_clips + total_abnormal_clips

    print("\n" + "="*50)
    print("=== FINAL SUMMARY ===")
    print(f"Total Clips Processed: {total_clips}")
    print(f"Total Normal Clips:    {total_normal_clips}")
    print(f"Total Suspicious Clips:{total_abnormal_clips}")
    print(f"Total Chunks Processed:{total_chunks}")
    print("="*50)

    logging.info("=== FINAL SUMMARY ===")
    logging.info(f"Total Clips Processed: {total_clips}")
    logging.info(f"Total Normal Clips:    {total_normal_clips}")
    logging.info(f"Total Suspicious Clips:{total_abnormal_clips}")
    logging.info(f"Total Chunks Processed:{total_chunks}")
    logging.info("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos via Jetson Nano Webservice")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to folder containing videos")
    parser.add_argument("--multi_turn", action="store_true", help="Enable old multi-turn (first + second inference)")
    parser.add_argument("--chunk_seconds", type=int, default=None, help="Split video into N-second chunks")
    parser.add_argument("--single_api", action="store_true", help="Use new single API (default)")
    args = parser.parse_args()

    # Default = single API unless explicitly disabled
    single_api_mode = True if args.single_api or not args.multi_turn else False

    process_folder(
        args.video_folder,
        multi_turn=args.multi_turn,
        chunk_seconds=args.chunk_seconds,
        single_api=single_api_mode
    )

