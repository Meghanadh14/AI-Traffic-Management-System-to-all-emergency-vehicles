from flask import Flask, Response
import cv2
import torch
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model for real-time performance

# Define emergency vehicle classes (COCO dataset IDs)
EMERGENCY_CLASSES = {"ambulance": 47, "fire_truck": 10, "police_car": 7, "paramedic": 47}  # Adjust if needed

# Initialize Video Capture (0 for webcam, or provide video file path)
video_path = "/Users/meghanadhkottana/Documents/pythonProjects/vid.mp4"  # Change this if using a video file
cap = cv2.VideoCapture(video_path)

def detect_vehicles():
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Perform YOLO detection
        results = model(frame)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)

                # Filter only emergency vehicles
                if class_id in EMERGENCY_CLASSES.values() and confidence > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    label = list(EMERGENCY_CLASSES.keys())[list(EMERGENCY_CLASSES.values()).index(class_id)]

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(detect_vehicles(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Vehicle Detection API Running! Go to /video_feed to view."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
