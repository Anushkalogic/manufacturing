from flask import Flask, render_template, request, jsonify, url_for
import os, cv2, time
from roboflow import Roboflow
from PIL import Image
from werkzeug.utils import secure_filename
from collections import Counter
import threading

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "static/uploads"
TEMP_OUTPUT = "static/outputs/temp_output.mp4"
FINAL_OUTPUT = "static/outputs/output_final.mp4"
FRAME_OUTPUT_DIR = "static/detected_frames"
INPUT_VIDEO = os.path.join(UPLOAD_FOLDER, "input.mp4")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static/outputs", exist_ok=True)
os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

# Global state for live detection
detection_counts_global = Counter()
processing_flag = False

# ========== Utils ========== #
def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False

def wait_and_remove(filepath, retries=10, delay=0.5):
    for _ in range(retries):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return
        except PermissionError:
            time.sleep(delay)

# ========== Detection Pipeline ========== #
def run_roboflow_pipeline(input_video_path):
    global detection_counts_global, processing_flag
    detection_counts_global.clear()
    processing_flag = True

    wait_and_remove(TEMP_OUTPUT)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 120:
        fps = 25

    target_w, target_h = 640, 360
    out = cv2.VideoWriter(TEMP_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_w, target_h))

    rf = Roboflow(api_key="DiBsOHUZVRTHIOZjUoWJ")
    project = rf.workspace().project("manufacturing_base")
    model = project.version(2).model

    frame_index = 1
    frame_skip_interval = 10
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_pos = 0

    for file in os.listdir(FRAME_OUTPUT_DIR):
        os.remove(os.path.join(FRAME_OUTPUT_DIR, file))

    while current_pos < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        success, frame = cap.read()
        if not success or frame is None:
            current_pos += frame_skip_interval
            continue

        frame = cv2.resize(frame, (target_w, target_h))
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)

        if not is_valid_image(temp_path):
            current_pos += frame_skip_interval
            continue

        try:
            result = model.predict(temp_path, confidence=40, overlap=30).json()
            predictions = result.get("predictions", [])

            for pred in predictions:
                x1 = int(pred["x"] - pred["width"] / 2)
                y1 = int(pred["y"] - pred["height"] / 2)
                x2 = int(pred["x"] + pred["width"] / 2)
                y2 = int(pred["y"] + pred["height"] / 2)
                label = pred["class"]
                detection_counts_global[label] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            out_frame_path = os.path.join(FRAME_OUTPUT_DIR, f"frame_{frame_index}.jpg")
            cv2.imwrite(out_frame_path, frame)

        except Exception as e:
            print("Prediction error:", e)

        out.write(frame)
        frame_index += 1
        current_pos += frame_skip_interval

    cap.release()
    out.release()
    processing_flag = False

# ========== Routes ========== #
@app.route('/')
def index():
    global processing_flag

    # Run pipeline in a background thread
    if not processing_flag:
        threading.Thread(target=run_roboflow_pipeline, args=(INPUT_VIDEO,)).start()

    video_path = url_for('static', filename='outputs/output_final.mp4')
    return render_template("video_result.html", video_path=video_path, counts=dict(detection_counts_global), phone_alert=False)

@app.route('/api/process', methods=['POST'])
def process_video_api():
    global processing_flag
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files['video']
    filename = secure_filename(video.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(input_path)

    INPUT_VIDEO = input_path  # Update input path for reprocessing

    threading.Thread(target=run_roboflow_pipeline, args=(input_path,)).start()
    return jsonify({"video_url": "/static/outputs/output_final.mp4", "message": "Processing started"})

@app.route('/api/live_counts')
def live_counts():
    return jsonify({
        "counts": dict(detection_counts_global),
        "done": not processing_flag
    })

# ========== Run ========== #
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
