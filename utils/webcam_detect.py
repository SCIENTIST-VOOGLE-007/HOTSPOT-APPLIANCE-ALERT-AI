import cv2
import threading
import logging
from queue import Queue
from datetime import datetime
import time  # Added for time.sleep

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global lock for thread-safe access to analysis results
analysis_lock = threading.Lock()

# Global variable to store the latest detection result
current_analysis = {
    'person_count': 0,
    'appliances': [],
    'alert_required': False,
    'timestamp': ''
}

class CameraStream:
    def __init__(self):
        self.camera = None
        self.frame_queue = Queue(maxsize=10)
        self.running = False
        self.detecting = False
        self.thread = None
        self.lock = threading.Lock()
        
    def start(self):
        with self.lock:
            if not self.running:
                try:
                    logger.info("Attempting to open camera...")
                    # Try different camera indices if 0 fails
                    for i in range(3):  # Try indices 0, 1, 2
                        self.camera = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                        if self.camera.isOpened():
                            logger.info(f"Camera opened successfully on index {i}")
                            break
                    else:  # Executed if no break occurs (all indices failed)
                        raise RuntimeError("No camera device found after trying indices 0-2")
                    self.running = True
                    self.thread = threading.Thread(target=self._update_frame, daemon=True)
                    self.thread.start()
                    logger.info("Camera thread started successfully")
                    return True
                except Exception as e:
                    logger.error(f"Camera start error: {e}")
                    self.running = False
                    return False
    
    def start_detection(self):
        with self.lock:
            if self.running:
                self.detecting = True
                logger.info("Detection started")
                return True
            return False
    
    def stop_detection(self):
        with self.lock:
            self.detecting = False
            logger.info("Detection stopped")
    
    def _update_frame(self):
        global current_analysis
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to grab frame")
                    time.sleep(0.1)
                    continue
                
                if self.detecting:
                    try:
                        detection_result = detect_objects(frame)
                        with analysis_lock:
                            current_analysis.update({
                                'person_count': detection_result.get('person_count', 0),
                                'appliances': detection_result.get('appliances', []),
                                'alert_required': detection_result.get('alert_required', False),
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                    except Exception as e:
                        logger.error(f"Detection error: {e}")
                
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame)
                
            except Exception as e:
                logger.error(f"Frame update error: {e}")
                time.sleep(0.1)  # Avoid tight loop on error
    
    def get_frame(self):
        try:
            return self.frame_queue.get(block=True, timeout=1)
        except Queue.Empty:
            return None  # More specific exception handling
    
    def stop(self):
        with self.lock:
            self.running = False
            self.detecting = False
            if self.thread is not None:
                self.thread.join(timeout=2)
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            while not self.frame_queue.empty():
                self.frame_queue.get()
            logger.info("Camera stream stopped")

# Ensure camera is released on program exit (fixed global reference)
camera_stream = None  # Define globally here
import atexit
atexit.register(lambda: camera_stream.stop() if camera_stream and hasattr(camera_stream, 'stop') else None)