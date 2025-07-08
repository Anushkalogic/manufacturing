from flask import Flask, render_template, url_for
import os, cv2, subprocess, time
from roboflow import Roboflow
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)

# Config
INPUT_VIDEO = "static/uploads/input.mp4"
TEMP_OUTPUT = "static/outputs/temp_output.mp4"
FINAL_OUTPUT = "static/outputs/output_final.mp4"
FRAME_OUTPUT_DIR = "static/detected_frames"

os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/outputs", exist_ok=True)
os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

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
def run_roboflow_pipeline():
    wait_and_remove(TEMP_OUTPUT)

    # Clear previous frames
    for file in os.listdir(FRAME_OUTPUT_DIR):
        os.remove(os.path.join(FRAME_OUTPUT_DIR, file))

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 120:
        fps = 25

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(TEMP_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("VideoWriter not opened.")

    rf = Roboflow(api_key="DiBsOHUZVRTHIOZjUoWJ")
    project = rf.workspace().project("manufacturing_base")
    model = project.version(2).model  # ✅ .model not .mod


    frame_index = 1
    current_pos = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        success, frame = cap.read()

        if not success or frame is None or frame.size == 0:
            retry = 0
            while retry < 3:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                success, frame = cap.read()
                if success and frame is not None and frame.size > 0:
                    break
                retry += 1
            else:
                print(f"❌ Skipping unreadable frame at index {current_pos}")
                current_pos += 1
                continue

        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)

        if is_valid_image(temp_path):
            try:
                result = model.predict(temp_path, confidence=40, overlap=30).json()
                predictions = result.get("predictions", [])
                if predictions:
                    for pred in predictions:
                        x1 = int(pred["x"] - pred["width"] / 2)
                        y1 = int(pred["y"] - pred["height"] / 2)
                        x2 = int(pred["x"] + pred["width"] / 2)
                        y2 = int(pred["y"] + pred["height"] / 2)
                        label = pred["class"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Save annotated frame
                    out_frame_path = os.path.join(FRAME_OUTPUT_DIR, f"frame_{frame_index}.jpg")
                    cv2.imwrite(out_frame_path, frame)

            except Exception as e:
                print("Prediction error:", e)

        out.write(frame)
        frame_index += 1
        current_pos += 1

        if current_pos >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break

    cap.release()
    out.release()

    subprocess.run([
        "ffmpeg", "-y", "-fflags", "+genpts", "-i", TEMP_OUTPUT,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        FINAL_OUTPUT
    ], capture_output=True)

@app.route('/')
def index():
    run_roboflow_pipeline()
    video_path = url_for('static', filename='outputs/output_final.mp4')
    return render_template("video_result.html", video_path=video_path)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
