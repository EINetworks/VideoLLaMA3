import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import time

PROMPT_FIRST = (
    "Analyze ATM video for: vandalism, fire, chains, explosives, destruction, theft, tampering, "
    "robbery, fraud, coercion. Normal use includes standard transactions, official maintenance, "
    "and instructional demonstrations. Distinguish normal use vs suspicious acts."
)

video_file = "/mnt/ssd_2T/users/tester/Videol/Data/ATMGT/May-2025/ATM-Vandalism/video_0001.mp4"

device = "cuda:0"
model_path = "DAMO-NLP-SG/VideoLLaMA3-2B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

conversation = [
    {"role": "system", "content": "You are a security officer"},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": {"video_path": video_file, "fps": 1, "max_frames": 100}},
            {"type": "text", "text": PROMPT_FIRST},
        ]
    },
]

inputs = processor(
    conversation=conversation,
    add_system_prompt=True,
    add_generation_prompt=True,
    return_tensors="pt"
)
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

num_iterations = 5
print(f"\nStarting inference loop for {num_iterations} iterations...")

for i in range(num_iterations):
    start_time = time.time()
    output_ids = model.generate(**inputs, max_new_tokens=256)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    proc_time = time.time() - start_time
    print(response)
    print(f'proc_time: {proc_time:.3f}')