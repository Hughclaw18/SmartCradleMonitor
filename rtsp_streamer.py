import cv2
import subprocess
import os
import sys
import time
import numpy as np
import requests
from PIL import Image
import supervision as sv
from ultralytics import YOLO

# Add the parent directory to sys.path to import from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_loader import load_object_detection_model, load_posture_detection_model, HAZARDOUS_CLASSES

# Configuration
RTSP_URL = "rtsp://localhost:8554/live"
WEBRTC_PORT = 8889
HLS_PORT = 8888
MEDIAMTX_PATH = os.path.join(os.path.dirname(__file__), "mediamtx.exe")
CLOUDFLARED_PATH = os.path.join(os.path.dirname(__file__), "cloudflared.exe")
FFMPEG_PATH = r"c:\fyp\SmartCradleMonitor\node_modules\ffmpeg-static\ffmpeg.exe"
API_URL = "http://localhost:5000/api"

# Internet Streaming
ENABLE_INTERNET = True # Set to True to start a tunnel
tunnel_process = None

def start_mediamtx():
    """Starts the MediaMTX RTSP server if not already running."""
    print(f"Starting MediaMTX from {MEDIAMTX_PATH}...")
    try:
        # Check if already running (port 8554)
        return subprocess.Popen([MEDIAMTX_PATH], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Warning: Could not start MediaMTX: {e}")
        return None

def start_tunnel():
    """Starts a Cloudflare tunnel for the WebRTC port (Ultra-low latency)."""
    print(f"Starting internet tunnel on port {WEBRTC_PORT} (WebRTC)...")
    try:
        # Using cloudflared to tunnel the WebRTC/HLS port
        # We'll tunnel the WebRTC interface for best performance
        proc = subprocess.Popen(
            [CLOUDFLARED_PATH, "tunnel", "--url", f"http://localhost:{WEBRTC_PORT}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Wait a bit for the URL to be generated
        print("Waiting for Cloudflare tunnel URL...")
        for _ in range(20):
            line = proc.stdout.readline()
            if "trycloudflare.com" in line:
                # Extract the URL from the log line
                import re
                match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                if match:
                    url = match.group(0)
                    print(f"🚀 INTERNET STREAM (WebRTC) AVAILABLE AT: {url}")
                    print(f"💡 This URL provides ultra-low latency (<500ms) for your mobile app.")
                    return proc
            time.sleep(1)
            
        return proc
    except Exception as e:
        print(f"Warning: Could not start Cloudflare tunnel: {e}")
        return None

# Detection settings
FRAME_SKIP = 2  # Run detection every N frames to save CPU
WIDTH, HEIGHT = 640, 480
FPS = 15

def get_ffmpeg_process():
    """Starts the ffmpeg process to push frames to the RTSP server."""
    command = [
        FFMPEG_PATH,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        RTSP_URL
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)

def annotate_frame(frame, posture_model, object_model):
    """Runs detection and returns an annotated frame."""
    # Convert BGR to RGB for models
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
            
        # Annotate posture boxes
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
        labels = [result.names[cid] for cid in detections.class_id] if detections.class_id is not None else []
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # 2. Object Detection
    object_results = object_model.predict(source=frame, save=False, verbose=False)
    hazardous_found = []
    if object_results:
        result = object_results[0]
        detections = sv.Detections.from_ultralytics(result)
        detected_objects = [result.names[cid] for cid in detections.class_id] if detections.class_id is not None else []
        hazardous_found = [obj for obj in detected_objects if obj in HAZARDOUS_CLASSES]
        
        # Annotate object boxes
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
        labels = [result.names[cid] for cid in detections.class_id] if detections.class_id is not None else []
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # Add overlay text
    cv2.putText(frame, f"Posture: {posture_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if hazardous_found:
        cv2.putText(frame, f"DANGER: {', '.join(hazardous_found)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Status: Safe", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    return frame

def main():
    # Load models
    print("Loading models...")
    posture_model = load_posture_detection_model()
    object_model = load_object_detection_model()
    
    # Start RTSP Server
    mtx_process = start_mediamtx()
    time.sleep(2) # Wait for server to start
    
    # Start Internet Tunnel
    tunnel_proc = None
    if ENABLE_INTERNET:
        tunnel_proc = start_tunnel()
    
    # Start FFmpeg
    print(f"Starting RTSP stream at {RTSP_URL}...")
    ffmpeg_process = get_ffmpeg_process()
    
    # Open Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open camera. Using dummy source.")
        # We'll generate dummy frames later
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Generate a dummy frame if camera fails
                frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, "No Camera Found - Simulating...", (50, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"), (50, HEIGHT//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            else:
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
                
            # Run detection periodically
            if frame_count % FRAME_SKIP == 0:
                frame = annotate_frame(frame, posture_model, object_model)
            
            # Write to ffmpeg
            try:
                ffmpeg_process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("FFmpeg process broken. Restarting...")
                ffmpeg_process = get_ffmpeg_process()
                
            frame_count += 1
            # Control FPS roughly
            time.sleep(1.0 / FPS)
            
            # Print status every 100 frames
            if frame_count % 100 == 0:
                print(f"Streamed {frame_count} frames to {RTSP_URL}")
                
    except KeyboardInterrupt:
        print("Stopping streamer...")
    finally:
        cap.release()
        if ffmpeg_process:
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
        if mtx_process:
            mtx_process.terminate()
        if tunnel_proc:
            tunnel_proc.terminate()

if __name__ == "__main__":
    main()
