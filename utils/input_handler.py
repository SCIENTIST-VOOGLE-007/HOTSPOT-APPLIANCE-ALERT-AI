import os
import cv2
import torch
import threading
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
add_safe_globals([DetectionModel])  # ✅ Allow YOLO class to load
from ultralytics import YOLO
from src.logic import analyze_detection
import logging
import time

# Set up logging to file to reduce console output
logging.basicConfig(filename='video_processing.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLOv8 model globally with timing
start_time = time.time()
model_path = "yolov8n.pt"
if not os.path.exists(model_path):
    logger.error(f"Model file {model_path} not found")
    raise FileNotFoundError(f"Model file {model_path} not found")
model = YOLO(model_path)
logger.info(f"Model loaded in {time.time() - start_time} seconds")
COCO_CLASSES = model.names

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, {"error": "Failed to load image"}
    results = model(img)
    annotated_img = results[0].plot()
    analysis = analyze_detection(results, COCO_CLASSES)
    return annotated_img, analysis

def process_folder(image_paths):
    analyses = []
    images = []
    for path in image_paths:
        if os.path.isfile(path) and path.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(path)
            if img is not None:
                results = model(img)
                annotated = results[0].plot()
                analysis = analyze_detection(results, COCO_CLASSES)
                analyses.append((os.path.basename(path), analysis))
                images.append((os.path.basename(path), annotated))
    return images, analyses

def process_video(video_path, output_path="static/results/video_output.avi"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {"error": "Failed to open video"}

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    all_analyses = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        out.write(annotated)
        analysis = analyze_detection(results, COCO_CLASSES)
        all_analyses.append(analysis)

    cap.release()
    out.release()
    alert_required = any(analysis.get('alert_required', False) for analysis in all_analyses)
    final_analysis = {
        "frames": all_analyses,
        "alert_required": alert_required
    }
    return output_path, final_analysis

def process_webcam(duration=10):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return [{"error": "Failed to open webcam"}]
    
    start_time = cv2.getTickCount()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(fps * duration)
    analyses = []

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        cv2.imshow("Webcam Feed", annotated)
        analysis = analyze_detection(results, COCO_CLASSES)
        analyses.append(analysis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return analyses