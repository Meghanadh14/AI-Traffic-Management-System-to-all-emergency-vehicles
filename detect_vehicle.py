import cv2
from ultralytics import YOLO

# Load YOLO model (use a more accurate version)
model = YOLO("yolov8m.pt")  # 'yolov8l.pt' is even better but slower

# Define emergency vehicles for India
EMERGENCY_VEHICLES = {
    41: "ambulance",   # Indian Ambulance
    7: "fire_truck",   # Indian Fire Truck
    2: "police_car"    # Indian Police Vehicle
}

# Load video (Replace with a live webcam feed if needed)
video_path = "/Users/meghanadhkottana/Documents/pythonProjects/megha.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video settings
output_path = "output.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Run YOLO detection
    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            conf = box.conf[0].item()  
            cls = int(box.cls[0].item())  

            # Detect only Indian emergency vehicles with a high confidence threshold
            if cls in EMERGENCY_VEHICLES and conf > 0.7:  # Increased threshold to reduce false positives
                label = f"{EMERGENCY_VEHICLES[cls]}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out.write(frame)

    # Show real-time detection
    cv2.imshow("Indian Emergency Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


