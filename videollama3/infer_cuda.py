import sys
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append('./')
from videollama3 import disable_torch_init, model_init, mm_infer
from videollama3.mm_utils import load_video, load_images

device = "cuda:0"

PROMPT_FIRST = (
    "Analyze ATM video for: vandalism, fire, chains, explosives, destruction, theft, tampering, "
    "robbery, fraud, coercion. Normal use includes standard transactions, official maintenance, "
    "and instructional demonstrations. Distinguish normal use vs suspicious acts."
)

def main():
    disable_torch_init()

    modal = "text"
    conversation = [
        {
            "role": "user",
            "content": "What is the color of bananas?",
        }
    ]

    modal = "image"
    frames = load_images("assets/sora.png")[0]
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is the woman wearing?"},
            ]
        }
    ]

    modal = "video"
    frames, timestamps = load_video("video_0001.mp4", fps=1, max_frames=180)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "timestamps": timestamps, "num_frames": len(frames)},
                {"type": "text", "text": PROMPT_FIRST},
            ]
        }
    ]

    #model_path = "/path/to/your/model"
    model_path = "ModelBackups/work_dirs_chunksv0/trained_hf_checkpoint/"
    #model_path = "weights/videollama3_2b_local/"
    model, processor = model_init(model_path)

    model.to(device)

    inputs = processor(
        images=[frames] if modal != "text" else None,
        text=conversation,
        merge_size=2 if modal == "video" else 1,
        return_tensors="pt",
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    # --- 2. Run Inference in a Loop ---
    num_iterations = 5
    print(f"\nStarting inference loop for {num_iterations} iterations...")

    for i in range(num_iterations):
        print(f"\n--- Iteration {i + 1}/{num_iterations} ---")
        
        # Record the start time
        start_time = time.time()

        # Run the inference function
        output = mm_infer(
            inputs,
            model=model,
            tokenizer=processor.tokenizer,
            do_sample=False,
            modal=modal
        )

        # Record the end time
        end_time = time.time()
        
        # Calculate and print the execution time
        execution_time = end_time - start_time

        print("Model Output:", output)
        print(f"Execution Time: {execution_time:.4f} seconds")



if __name__ == "__main__":
    main()
