
# Manufacturing Detection App (Flask + Roboflow)

## Features:
- Detects machine status (ON/OFF)
- Detects PPE kit status (worn/not worn)
- Counts people entry/exit
- Live video stream via Flask
- Roboflow integration (YOLOv8/11 compatible)

## Setup:
1. Replace API Key and Model ID in `config.py`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open browser at `http://localhost:5000`

## Notes:
- Default input is webcam. Modify `cv2.VideoCapture(0)` in `detection.py` for RTSP or file input.
