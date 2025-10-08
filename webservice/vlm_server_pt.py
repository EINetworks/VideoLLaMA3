from flask import Flask, request, jsonify, render_template_string
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only GPU 0 will be visible
import torch
import tempfile

# Import new inference libraries
import sys
sys.path.append('../') # Assuming videollama3 is in the current directory or a sub-directory
from videollama3 import disable_torch_init, model_init, mm_infer
from videollama3.mm_utils import load_video

# === CONFIG ===
# Updated model path as per the new inference logic
MODEL_PATH = "../weights/videollama3_2b_local/"
DEVICE = "cuda:0" # The new inference logic will handle device placement

PROMPT_ROLE = "You are a security officer"

PROMPT_FIRST = "Analyze ATM video for: vandalism, fire, chains, explosives, destruction, theft, tampering, robbery, fraud, coercion. Normal use includes standard transactions, official maintenance, and instructional demonstrations. Distinguish normal use vs suspicious acts."
PROMPT_SEC = "Based on the detailed analysis you just provided, is the primary activity in the video a 'Suspicious Act' or 'Normal Use'? A suspicious act is any action involving vandalism, theft, tampering, destruction, or coercion. Normal use includes standard transactions, official maintenance, and instructional demonstrations. Answer with only one of these two phrases: 'Suspicious' or 'Normal'."

PROMPT_SINGLE = "Analyze ATM video for: vandalism, fire, chains, explosives, destruction, theft, tampering, robbery, fraud, coercion. Distinguish normal use vs suspicious acts."

# === LOAD MODEL ONCE (New Method) ===
print("Disabling torch init...")
disable_torch_init()
print("Loading model...")
model, processor = model_init(MODEL_PATH)
print("Model ready!")


app = Flask(__name__)

# --- Updated Inference Function with New Parameters ---
def run_inference(video_path, fps, max_frames, turn="single", prev_response=None):
    try:
        # Load video using the new utility
        frames, timestamps = load_video(video_path, fps=fps, max_frames=max_frames)

        if turn == "single":
            conversation = [
                {"role": "user", "content": [
                    {"type": "video", "timestamps": timestamps, "num_frames": len(frames)},
                    {"type": "text", "text": PROMPT_SINGLE}
                ]}
            ]
        elif turn == "first":
            conversation = [
                {"role": "user", "content": [
                    {"type": "video", "timestamps": timestamps, "num_frames": len(frames)},
                    {"type": "text", "text": PROMPT_FIRST}
                ]}
            ]
        elif turn == "second" and prev_response:
            conversation = [
                {"role": "user", "content": [
                    {"type": "video", "timestamps": timestamps, "num_frames": len(frames)},
                    {"type": "text", "text": PROMPT_FIRST}
                ]},
                {"role": "assistant", "content": prev_response},
                {"role": "user", "content": [{"type": "text", "text": PROMPT_SEC}]}
            ]
        else:
            return {"error": "Invalid turn or missing prev_response"}

        # Prepare inputs using the new processor
        inputs = processor(
            images=[frames],
            text=conversation,
            merge_size=2, # For video
            return_tensors="pt",
        )

        # Run inference using the new mm_infer function with specified parameters
        final_response = mm_infer(
            inputs,
            model=model,
            tokenizer=processor.tokenizer,
            modal="video",
            # --- USER REQUESTED PARAMETERS ---
            do_sample=True,
            max_new_tokens=256,
            temperature=0.7
        )

        return {"response": final_response.strip()}

    except Exception as e:
        # It's good practice to log the full error for debugging
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}


# --- Web Interface with formatted results (No Changes Here) ---
HTML_PAGE = """
<!doctype html>
<html>
<head>
  <title>ATM Video Analyzer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 30px; }
    video { max-width: 400px; margin-top: 10px; display: none; }
    .result-container { margin-top: 20px; }
    .result-box { margin-bottom: 15px; padding: 10px; border: 1px solid #ccc; background: #f9f9f9; white-space: pre-wrap; }
    .label { font-weight: bold; margin-bottom: 5px; display: block; }
  </style>
</head>
<body>
  <h2>ATM Video Analyzer</h2>
  <form id="uploadForm" enctype="multipart/form-data">
    <input type="file" id="video" name="video" accept="video/*" required><br>
    <video id="preview" controls></video><br><br>
    <button type="button" onclick="analyze('single')">Analyze (Single Turn)</button>
    <button type="button" onclick="analyze('both')">Analyze (Both Turns)</button>
  </form>

  <div id="results" class="result-container"></div>

<script>
const videoInput = document.getElementById('video');
const preview = document.getElementById('preview');
const results = document.getElementById('results');

videoInput.addEventListener('change', () => {
    const file = videoInput.files[0];
    if (file) {
        const url = URL.createObjectURL(file);
        preview.src = url;
        preview.style.display = "block";
    } else {
        preview.style.display = "none";
    }
});

function analyze(mode) {
    const file = videoInput.files[0];
    if (!file) {
        alert("Please select a video first.");
        return;
    }

    const formData = new FormData();
    formData.append("video", file);
    formData.append("fps", 0.5);         // default fps
    formData.append("max_frames", 60);   // default max_frames

    const endpoint = mode === "single" ? "/inference/single" : "/inference/both_turns";

    results.innerHTML = "<div class='result-box'>Processing... please wait ⏳</div>";

    fetch(endpoint, { method: "POST", body: formData })
      .then(r => r.json())
      .then(data => {
        results.innerHTML = "";
        for (const key in data) {
            const box = document.createElement("div");
            box.className = "result-box";
            box.innerHTML = `<span class="label">${key.replace('_', ' ').toUpperCase()}:</span>${data[key] || "No response"}`;
            results.appendChild(box);
        }
      })
      .catch(err => {
        results.innerHTML = "<div class='result-box'>Error: " + err + "</div>";
      });
}
</script>

</body>
</html>
"""

# --- Flask Routes (No Changes Here) ---
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/inference/<turn>", methods=["POST"])
def inference(turn):
    file = request.files.get("video")
    fps = float(request.form.get("fps", 0.5))
    max_frames = int(request.form.get("max_frames", 30))
    prev_response = request.form.get("prev_response")

    print(f"Request: FPS {fps}, Frames {max_frames}, prev_response {prev_response}")

    if not file:
        return jsonify({"error": "No video uploaded"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        file.save(tmp.name)
        video_path = tmp.name

    result = run_inference(video_path, fps, max_frames, turn, prev_response)
    print(f"result: {result}")
    os.remove(video_path)
    return jsonify(result)

@app.route("/inference/both_turns", methods=["POST"])
def inference_both_turns():
    file = request.files.get("video")
    fps = float(request.form.get("fps", 0.5))
    max_frames = int(request.form.get("max_frames", 30))

    if not file:
        return jsonify({"error": "No video uploaded"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        file.save(tmp.name)
        video_path = tmp.name

    try:
        # --- First Turn ---
        first_result = run_inference(video_path, fps, max_frames, turn="first")
        first_resp = first_result.get("response") if "response" in first_result else None

        # --- Second Turn (using first response) ---
        second_result = None
        second_resp = None
        if first_resp:
            second_result = run_inference(video_path, fps, max_frames, turn="second", prev_response=first_resp)
            second_resp = second_result.get("response") if "response" in second_result else None

        result = {
            "first_turn": first_resp,
            "second_turn": second_resp
        }
    except Exception as e:
        result = {"error": str(e)}

    os.remove(video_path)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5010)