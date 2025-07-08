
import cv2
import supervision as sv
from inference import get_roboflow_model
from config import ROBOFLOW_API_KEY, MODEL_ID

model = get_roboflow_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)
annotator = sv.BoxAnnotator()

def generate_frames():
    cap = cv2.VideoCapture(0)  # Change to RTSP or file if needed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        labels = [f"{model_class} {confidence:.2f}" for model_class, confidence in zip(detections.class_id, detections.confidence)]
        annotated_frame = annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
