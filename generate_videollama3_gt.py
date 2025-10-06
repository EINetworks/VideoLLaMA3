import os
import re
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


# ===============================
# CONFIG
# ===============================
MODEL_PATH = "DAMO-NLP-SG/VideoLLaMA3-2B"
DEVICE = "cuda:2"

PROMPT_ROLE = "You are a security officer"

PROMPT_FIRST = (
    "Analyze ATM video for: vandalism, fire, chains, explosives, destruction, theft, tampering, "
    "robbery, fraud, coercion. Normal use includes standard transactions, official maintenance, "
    "and instructional demonstrations. Distinguish normal use vs suspicious acts."
)

PROMPT_SEC = (
    "Based on the detailed analysis you just provided, is the primary activity in the video a "
    "'Suspicious Act' or 'Normal Use'? A suspicious act is any action involving vandalism, theft, "
    "tampering, destruction, or coercion. Normal use includes standard transactions, official "
    "maintenance, and instructional demonstrations. Answer with only one of these two phrases: "
    "'Suspicious' or 'Normal'."
)


# ===============================
# LOAD MODEL
# ===============================
print("Loading VideoLLaMA3 model ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map={"": DEVICE},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("✅ Model loaded successfully\n")


# ===============================
# PATH UTIL
# ===============================
def get_rel_video_path(full_path: str) -> str:
    """
    Extracts relative path starting from first folder that looks like 'Month-YYYY'
    e.g.:
      /mnt/ssd_4T/users/tester/Videol/Data/May-2025/ATM-Normal/clip1.mp4
      -> May-2025/ATM-Normal/clip1.mp4
    Works for any month-year folder (e.g. June-2025, Jul-2024, etc.)
    """
    full_path = full_path.replace("\\", "/")
    parts = full_path.split("/")
    month_year_pattern = re.compile(r"^[A-Za-z]{3,9}-\d{4}$")  # e.g. May-2025, September-2024, Jul-2025

    for i, part in enumerate(parts):
        if month_year_pattern.match(part):
            return "/".join(parts[i:])

    # fallback: if no month-year folder found
    return os.path.basename(full_path)


# ===============================
# INFERENCE
# ===============================
def run_inference(video_path, fps=0.5, max_frames=60, turn="single", prev_response=None):
    """Run single-turn or two-turn inference."""
    try:
        if turn == "single":
            # Single-turn → uses PROMPT_FIRST only
            conversation = [
                {"role": "system", "content": PROMPT_ROLE},
                {"role": "user", "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": fps, "max_frames": max_frames}},
                    {"type": "text", "text": PROMPT_FIRST}
                ]}
            ]

        elif turn == "first":
            # First turn of two-turn inference
            conversation = [
                {"role": "system", "content": PROMPT_ROLE},
                {"role": "user", "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": fps, "max_frames": max_frames}},
                    {"type": "text", "text": PROMPT_FIRST}
                ]}
            ]

        elif turn == "second" and prev_response:
            # Second turn → uses PROMPT_SEC with previous assistant output
            conversation = [
                {"role": "system", "content": PROMPT_ROLE},
                {"role": "user", "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": fps, "max_frames": max_frames}},
                    {"type": "text", "text": PROMPT_FIRST}
                ]},
                {"role": "assistant", "content": prev_response},
                {"role": "user", "content": [{"type": "text", "text": PROMPT_SEC}]}
            ]
        else:
            return {"error": "Invalid turn or missing prev_response"}

        # Prepare input
        inputs = processor(conversation=conversation, add_system_prompt=True, add_generation_prompt=True, return_tensors="pt")
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # Token limit: short for classification
        max_tokens = 10 if turn == "second" else 256
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
        final_response = decoded[0].strip() if decoded else ""
        return {"response": final_response}

    except Exception as e:
        torch.cuda.empty_cache()
        return {"error": str(e)}
    finally:
        torch.cuda.empty_cache()


# ===============================
# MAIN LOGIC
# ===============================
def generate_ground_truths(video_folder, output_single="gt_single.jsonl", output_two_turn="gt_two_turn.jsonl"):
    """Run inference on all .mp4 videos in a folder and generate GT JSONL files."""
    videos = [
        os.path.join(root, file)
        for root, _, files in os.walk(video_folder)
        for file in files if file.lower().endswith(".mp4")
    ]

    print(f"Found {len(videos)} videos in {video_folder}\n")

    with open(output_single, "w", encoding="utf-8") as f_single, \
         open(output_two_turn, "w", encoding="utf-8") as f_two:

        for video_path in tqdm(videos, desc="Processing videos"):
            rel_path = get_rel_video_path(video_path)

            # ---- SINGLE TURN ----
            single_out = run_inference(video_path, turn="single")
            single_resp = single_out.get("response", "")
            gt_single = {
                "video": [rel_path],
                "conversations": [
                    {"from": "human", "value": "<video>\nIs there any abnormal incidence happening? explain in detail"},
                    {"from": "gpt", "value": single_resp}
                ]
            }
            print('gt_single', gt_single)
            f_single.write(json.dumps(gt_single, ensure_ascii=False) + "\n")

            # ---- TWO TURN ----
            first_out = run_inference(video_path, turn="first")
            first_resp = first_out.get("response", "")
            second_out = run_inference(video_path, turn="second", prev_response=first_resp)
            second_resp = second_out.get("response", "")

            gt_two = {
                "video": [rel_path],
                "conversations": [
                    {"from": "human", "value": "<video>\n" + PROMPT_FIRST},
                    {"from": "gpt", "value": first_resp},
                    {"from": "human", "value": PROMPT_SEC},
                    {"from": "gpt", "value": second_resp}
                ]
            }
            print('gt_two', gt_two)
            f_two.write(json.dumps(gt_two, ensure_ascii=False) + "\n")

    print(f"\n✅ GT files created:\n  • {output_single}\n  • {output_two_turn}")


# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run VideoLLaMA3 inference and create GT JSON files.")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to folder containing video files")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save GT files")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample")
    parser.add_argument("--max_frames", type=int, default=60, help="Max frames per video")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    out_single = os.path.join(args.output_dir, "gt_single.jsonl")
    out_two = os.path.join(args.output_dir, "gt_two_turn.jsonl")

    generate_ground_truths(args.video_folder, out_single, out_two)

'''
nohup python generate_videollama3_gt.py \
    --video_folder /mnt/ssd_2T/users/tester/Videol/Data/ATMGT/May-2025/ATM-Normal \
    --output_dir /mnt/ssd_2T/users/tester/Videol/Data/ATMGT/May-2025/ATM-Normal \
    > gt_logs/videollama3_may2025_normal_gt.log 2>&1 &

nohup python generate_videollama3_gt.py \
    --video_folder /mnt/ssd_2T/users/tester/Videol/Data/ATMGT/May-2025/ATM-Vandalism/ \
    --output_dir /mnt/ssd_2T/users/tester/Videol/Data/ATMGT/May-2025/ATM-Vandalism/ \
    > gt_logs/videollama3_may2025_vandalism_gt.log 2>&1 &


nohup python generate_videollama3_gt.py \
    --video_folder /mnt/ssd_2T/users/tester/Videol/Data/ATMGT/Aug-2025/Normal/ \
    --output_dir /mnt/ssd_2T/users/tester/Videol/Data/ATMGT/Aug-2025/Normal/ \
    > gt_logs/videollama3_aug2025_normal_gt.log 2>&1 &

nohup python generate_videollama3_gt.py \
    --video_folder /mnt/ssd_2T/users/tester/Videol/Data/ATMGT/Aug-2025/Vandalism/ \
    --output_dir /mnt/ssd_2T/users/tester/Videol/Data/ATMGT/Aug-2025/Vandalism/ \
    > gt_logs/videollama3_aug2025_vandalism_gt.log 2>&1 &    
'''