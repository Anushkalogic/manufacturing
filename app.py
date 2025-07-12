from flask import Flask, render_template, request, jsonify, url_for, Response
import os, cv2, time
import mysql.connector
from datetime import datetime
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

# DB config
DB_CONFIG = {
    'host': 'localhost',
    'user': 'flaskuser',
    'password': 'flaskpassword',
    'database': 'cattle_detection'
}

# Global state for live detection
detection_counts_global = Counter()
processing_flag = False

# Initialize Roboflow model only once
rf = Roboflow(api_key="DiBsOHUZVRTHIOZjUoWJ")
project = rf.workspace().project("manufacturing_base")
model = project.version(2).model

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

    frame_index = 1
    frame_skip_interval = 10
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_pos = 0

    # Clean up old frames
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
            if not predictions:
                out_frame_path = os.path.join(FRAME_OUTPUT_DIR, f"frame_{frame_index}_unauth.jpg")
                cv2.imwrite(out_frame_path, frame)
                insert_detection_mysql(out_frame_path, label=None)

            for pred in predictions:
                x1 = int(pred["x"] - pred["width"] / 2)
                y1 = int(pred["y"] - pred["height"] / 2)
                x2 = int(pred["x"] + pred["width"] / 2)
                y2 = int(pred["y"] + pred["height"] / 2)
                label = pred["class"]
                detection_counts_global[label] += 1

                # Annotate
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Save and insert
                out_frame_path = os.path.join(FRAME_OUTPUT_DIR, f"frame_{frame_index}.jpg")
                cv2.imwrite(out_frame_path, frame)
                insert_detection_mysql(out_frame_path, label)

        except Exception as e:
            print("Prediction error:", e)

        out.write(frame)
        frame_index += 1
        current_pos += frame_skip_interval

    cap.release()
    out.release()
    processing_flag = False

def insert_detection_mysql(image_path, label, camera_name="web"):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        now = datetime.now()
        hour = now.hour
        is_late = hour >= 10
        status = "Late Arrival" if is_late else "On Time"
        remarks = "Checked In" if label else "Unauthorized"

        cursor.execute("""
            INSERT INTO images (image_path, volume_liters, label, camera_name, timestamp, status, remarks)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (image_path, None, label or "none", camera_name, now, status, remarks))

        conn.commit()
        print(f"✅ MySQL Inserted: {image_path} | {label} | {status}")

    except mysql.connector.Error as e:
        print(f"❌ MySQL Error: {e}")

    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

# ========== Video Feed Route ========== #
def generate_live_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        result = model.predict(frame, confidence=40, overlap=30).json()
        predictions = result.get("predictions", [])
        for pred in predictions:
            x1 = int(pred["x"] - pred["width"] / 2)
            y1 = int(pred["y"] - pred["height"] / 2)
            x2 = int(pred["x"] + pred["width"] / 2)
            y2 = int(pred["y"] + pred["height"] / 2)
            label = pred["class"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/live_video')
def live_video():
    return Response(generate_live_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    global processing_flag
    if not processing_flag:
        threading.Thread(target=run_roboflow_pipeline, args=(INPUT_VIDEO,)).start()
    video_path = url_for('static', filename='outputs/output_final.mp4')
    return render_template("video_result.html", video_path=video_path, counts=dict(detection_counts_global), phone_alert=False)

@app.route('/api/process', methods=['POST'])
def process_video_api():
    global processing_flag, INPUT_VIDEO
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files['video']
    filename = secure_filename(video.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(input_path)
    INPUT_VIDEO = input_path

    threading.Thread(target=run_roboflow_pipeline, args=(input_path,)).start()
    return jsonify({"video_url": "/static/outputs/output_final.mp4", "message": "Processing started"})

@app.route('/api/live_counts')
def live_counts():
    return jsonify({
        "counts": dict(detection_counts_global),
        "done": not processing_flag
    })


@app.route('/api/compliance_data', methods=['GET'])
def get_compliance_data():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_detections = 0
    late_arrivals = 0
    unauthorized_access = 0
    logs = []

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        # Check if DB has any rows
        cursor.execute("SELECT COUNT(*) as count FROM images")
        row_count = cursor.fetchone()["count"]

        if row_count == 0:
            # 🔁 First time / no data yet
            return jsonify({
                "compliancePercentage": 100,
                "lateArrivals": 0,
                "unauthorizedAccess": 0,
                "totalDetections": 0,
                "logs": [],
                "videoFeed": "/live_video"
            })

        # ✅ Fetch last 50 rows
        cursor.execute("SELECT * FROM images ORDER BY timestamp DESC LIMIT 50")
        rows = cursor.fetchall()

        for row in rows:
            if row["label"] != "none":
                total_detections += 1
            if row["status"] == "Late Arrival":
                late_arrivals += 1
            if row["label"] == "none":
                unauthorized_access += 1

            logs.append({
                "id": row["id"],
                "datetime": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "department": "N/A",
                "employee": "Leo",
                "device": row["camera_name"],
                "location": "Main Gate",
                "duration": "8h",
                "remarks": row["remarks"],
                "status": row["status"]
            })

        # Compute compliance %
        compliance_percentage = max(0, 100 - ((late_arrivals + unauthorized_access) * 100 // (total_detections or 1)))

        return jsonify({
            "compliancePercentage": compliance_percentage,
            "lateArrivals": late_arrivals,
            "unauthorizedAccess": unauthorized_access,
            "totalDetections": total_detections,
            "logs": logs,
            "videoFeed": "/live_video"
        })

    except mysql.connector.Error as e:
        print(f"❌ MySQL Error: {e}")
        return jsonify({
            "error": "Database connection failed",
            "videoFeed": "/live_video"
        }), 500

    finally:
        try:
            if conn.is_connected():
                cursor.close()
                conn.close()
        except:
            pass

# @app.route('/api/compliance_data', methods=['GET'])
# def get_compliance_data():
#     now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
#     total_detections = sum(detection_counts_global.values())  # use your actual detection counter here
#     late_arrivals = 6  # You can customize this logic
#     unauthorized_access = 0  # Add logic if needed

#     logs = [
#         {
#             "id": 1,
#             "datetime": now,
#             "department": "N/A",
#             "employee": "Leo",
#             "device": "CCTV",
#             "location": "Main Gate",
#             "duration": "8h",
#             "remarks": "On Time",
#             "status": "Checked In"
#         }
#     ]

#     return jsonify({
#         "compliancePercentage": 0,  # Logic for this can also be added
#         "lateArrivals": late_arrivals,
#         "unauthorizedAccess": unauthorized_access,
#         "logs": logs,
#         "totalDetections": total_detections,
#         "videoFeed": "/live_video"
#     })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
