import cv2
import torch
import time
import threading
import requests
from ultralytics import YOLO
from flask import Flask, request, jsonify


model = YOLO("yolov8n.pt")  


traffic_light = {"red": True, "green": False}


LOCATION = {"lat": 12.9716, "long": 77.5946}  


def send_alert(vehicle_type):
    alert_data = {"vehicle": vehicle_type, "location": LOCATION, "priority": "high"}
    api_url = "http://127.0.0.1:5000/alert"  # Local API
    try:
        response = requests.post(api_url, json=alert_data)
        print("🚨 Alert Sent:", response.json())
    except Exception as e:
        print("⚠️ Alert Failed:", e)


def switch_light(state):
    if state == "green":
        print("🔵 GREEN LIGHT: Emergency vehicle detected. Allow passage!")
        traffic_light["red"] = False
        traffic_light["green"] = True
    else:
        print("🔴 RED LIGHT: No emergency vehicle detected.")
        traffic_light["red"] = True
        traffic_light["green"] = False


def detect_vehicles():
    cap = cv2.VideoCapture("/Users/meghanadhkottana/Documents/pythonProjects/vid.mp4")  # 0 for webcam, or use a video file

    if not cap.isOpened():
        print("❌ ERROR: Could not open video source!")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Could not read frame!")
            break

     
        results = model(frame)
        emergency_detected = False

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0]) 
                confidence = float(box.conf[0])  
                
              
                if class_id in [5, 7] and confidence > 0.3:  
                    emergency_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  
                    cv2.putText(frame, "Emergency Vehicle", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    break
        
    
        if emergency_detected:
            switch_light("green") 
            send_alert("Emergency Vehicle")  
        else:
            switch_light("red") 

       
        cv2.imshow("Traffic Feed", frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


app = Flask(__name__)

@app.route('/alert', methods=['POST'])
def emergency_alert():
    data = request.json
    vehicle_type = data.get("vehicle")
    location = data.get("location")

    if not vehicle_type or not location:
        return jsonify({"error": "Missing vehicle type or location"}), 400

    print(f"🚑 ALERT RECEIVED: {vehicle_type} at {location}")
    return jsonify({"message": "Alert processed successfully!"})


def run_api():
    app.run(debug=True, port=5000, use_reloader=False)


if __name__ == "__main__":
    threading.Thread(target=run_api).start()  
    detect_vehicles()  



