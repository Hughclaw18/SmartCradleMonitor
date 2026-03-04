import cv2
import base64
import json
import time
import os
import sys
import numpy as np
import websocket
from PIL import Image

# Add the parent directory to sys.path to import from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_loader import load_object_detection_model, load_posture_detection_model, HAZARDOUS_CLASSES
import supervision as sv

# Configuration
WS_URL = os.getenv("WS_URL", "ws://localhost:5000/socket")
SIMULATOR_TOKEN = os.getenv("SIMULATOR_TOKEN", "default-simulator-token")

WIDTH, HEIGHT = 320, 240 # Lower resolution for WS streaming to save bandwidth
FPS = 10 # Lower FPS for WS streaming
FRAME_SKIP = 2

def get_base64_frame(frame):
    """Converts a BGR frame to a base64 encoded JPEG string."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode('utf-8')

def annotate_frame(frame, posture_model, object_model):
    """Runs detection and returns an annotated frame."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 1. Posture Detection
    posture_results = posture_model.predict(source=img_rgb, save=False, verbose=False)
    posture_text = "Unknown"
    if posture_results:
        result = posture_results[0]
        detections = sv.Detections.from_ultralytics(result)
        detected_parts = [result.names[cid] for cid in detections.class_id] if detections.class_id is not None else []
        
        if "Face" in detected_parts or "nose" in detected_parts:
            posture_text = "Facing up"
        elif "back" in detected_parts:
            posture_text = "Sleeping on stomach"
        elif "Lear" in detected_parts or "Rear" in detected_parts:
            posture_text = "Side sleeping"
            
        box_annotator = sv.BoxAnnotator()
        frame = box_annotator.annotate(scene=frame, detections=detections)

    # 2. Object Detection
    object_results = object_model.predict(source=frame, save=False, verbose=False)
    hazardous_found = []
    if object_results:
        result = object_results[0]
        detections = sv.Detections.from_ultralytics(result)
        detected_objects = [result.names[cid] for cid in detections.class_id] if detections.class_id is not None else []
        hazardous_found = [obj for obj in detected_objects if obj in HAZARDOUS_CLASSES]
        
        box_annotator = sv.BoxAnnotator()
        frame = box_annotator.annotate(scene=frame, detections=detections)

    # Add overlay text (smaller for lower res)
    cv2.putText(frame, f"Posture: {posture_text}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    if hazardous_found:
        cv2.putText(frame, "DANGER!", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
    return frame

def main():
    print(f"--- Smart Cradle Monitor Simulator (Remote) ---")
    print(f"Target URL: {WS_URL}")
    print("Loading models...")
    posture_model = load_posture_detection_model()
    object_model = load_object_detection_model()
    
    print(f"Connecting to WebSocket...")
    try:
        # Use custom headers for authentication
        headers = {"x-simulator-token": SIMULATOR_TOKEN}
        ws = websocket.create_connection(WS_URL, header=headers)
        print("🚀 Connected successfully to the remote server!")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("Make sure your WS_URL and SIMULATOR_TOKEN are correct.")
        return

    # Try to open camera with different indices and backends
    cap = None
    for index in [0, 1, 2]:
        # On Windows, CAP_DSHOW is often more stable and fixes "can't grab frame" errors
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"✅ Camera opened successfully on index {index}")
            break
        cap.release()
    
    if not cap or not cap.isOpened():
        print("❌ Warning: Could not open any camera. Using dummy source.")
        cap = None
    
    frame_count = 0
    try:
        while True:
            ret, frame = False, None
            if cap:
                ret, frame = cap.read()
            
            if not ret or frame is None:
                frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera Busy or Disconnected", (10, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
                
            if frame_count % FRAME_SKIP == 0:
                frame = annotate_frame(frame, posture_model, object_model)
            
            # Send via WS
            b64_data = get_base64_frame(frame)
            message = {
                "type": "video_frame",
                "data": b64_data
            }
            
            try:
                ws.send(json.dumps(message))
            except Exception as e:
                headers = {"x-simulator-token": SIMULATOR_TOKEN}
                try:
                    ws = websocket.create_connection(WS_URL, header=headers)
                except:
                    pass
                
            frame_count += 1
            time.sleep(1.0 / FPS)
            
            if frame_count % 100 == 0:
                print(f"Relayed {frame_count} frames via WebSocket")
                
    except KeyboardInterrupt:
        print("Stopping streamer...")
    finally:
        cap.release()
        ws.close()

if __name__ == "__main__":
    main()
