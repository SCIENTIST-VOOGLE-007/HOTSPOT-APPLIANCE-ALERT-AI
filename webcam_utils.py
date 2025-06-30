import cv2
import numpy as np
from threading import Lock
from datetime import datetime

# Global analysis variable
analysis_lock = Lock()
current_analysis = {
    'person_count': 0,
    'appliances': [],
    'alert_required': False,
    'timestamp': ''
}

class CameraStream:
    def __init__(self):
        self.camera = None
        self.detection_active = False
        self.model = self.load_model()
        self.last_detection_time = datetime.now()

    def load_model(self):
        # Try loading the model with error handling
        try:
            net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            print("Model loaded successfully")
            return net
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to Tiny YOLO
            try:
                net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
                print("Using Tiny YOLO as fallback")
                return net
            except Exception as e:
                print(f"Failed to load fallback model: {e}")
                return None

    def start(self):
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("Error: Could not open video device")
                return False
            self.detection_active = True
            return True
        return True

    def stop(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.detection_active = False

    def start_detection(self):
        if self.model is not None:
            self.detection_active = True
            return True
        return False

    def stop_detection(self):
        self.detection_active = False

    def get_frame(self):
        if self.camera is None:
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            print("Error reading frame from camera")
            return None
            
        if self.detection_active and self.model is not None:
            try:
                frame = self.detect_objects(frame)
            except Exception as e:
                print(f"Detection error: {e}")
            
        return frame

    def detect_objects(self, frame):
        height, width = frame.shape[:2]
        
        # Create blob from image with increased size for better detection
        blob = cv2.dnn.blobFromImage(frame, 1/255, (608, 608), swapRB=True, crop=False)
        self.model.setInput(blob)
        output_layers = self.model.getUnconnectedOutLayersNames()
        outputs = self.model.forward(output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Lower confidence threshold and include more classes
                if confidence > 0.3:  # Lowered from 0.5
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-max suppression with relaxed parameters
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)  # Reduced thresholds
        
        person_count = 0
        for i in indexes.flatten():
            if class_ids[i] == 0:  # Only count person class (0 in COCO)
                person_count += 1
        
        with analysis_lock:
            current_analysis['person_count'] = person_count
            current_analysis['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            current_analysis['alert_required'] = person_count > 0  # Simplified for testing
        
        # Draw bounding boxes
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
        
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(self.classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} {confidence}", (x, y-5), font, 1, color, 2)
        
        print(f"Detected {person_count} persons")  # Debug output
        return frame

def get_analysis_snapshot():
    with analysis_lock:
        return current_analysis.copy()