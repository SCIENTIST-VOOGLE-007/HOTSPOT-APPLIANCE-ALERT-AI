from ultralytics import YOLO
import cv2
import numpy as np

def detect_objects(frame):
    # Load the YOLOv8 model
    try:
        model = YOLO("yolov8n.pt")  # Path to your yolov8n.pt file
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        return {
            'person_count': 0,
            'appliances': [],
            'alert_required': False,
            'detections': {}
        }

    # Define custom appliance classes (based on COCO or your custom training)
    appliance_classes = ["oven"]  # Example; adjust based on your model
    appliance_indices = [62]  # COCO index for "oven"; adjust for custom classes

    # Perform detection
    results = model(frame)[0]  # Get the first result (single image)

    person_count = 0
    appliances = []
    detections = {}

    # Process detections
    for detection in results.boxes.data:  # xyxy, confidence, class_id
        x1, y1, x2, y2, conf, class_id = detection
        label = model.names[int(class_id)]
        conf = float(conf)
        print(f"Detected: {label} with confidence {conf}")  # Debug log

        # Draw bounding box (for visualization)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Count people and appliances
        if label == "person":
            person_count += 1
            print(f"Incremented person_count to {person_count}")  # Debug log
        elif int(class_id) in appliance_indices and label in appliance_classes:
            appliances.append(label)
            print(f"Added appliance: {label}")  # Debug log

        detections[label] = conf

    # Determine alert
    alert_required = person_count == 0 and len(appliances) > 0

    return {
        'person_count': person_count,
        'appliances': appliances,
        'alert_required': alert_required,
        'detections': detections
    }

# Example usage (for testing)
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = detect_objects(frame)
        print(f"Detection result: {result}")
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()