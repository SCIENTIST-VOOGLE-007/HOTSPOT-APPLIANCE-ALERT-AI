# src/logic.py

import datetime

# Define appliance and person keywords
DANGEROUS_APPLIANCES = ['oven', 'toaster', 'microwave', 'refrigerator']
PERSON_CLASS = 'person'

def analyze_detection(results, coco_names):
    """
    Analyze YOLOv8 results to determine:
    - Is any dangerous appliance detected?
    - Is any person detected?
    
    Returns a dict containing:
    - alert_required (True/False)
    - detected_appliances
    - person_count
    - timestamp
    """
    detected_appliances = set()
    person_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = coco_names.get(cls_id, "")
            if cls_name == PERSON_CLASS:
                person_count += 1
            elif cls_name in DANGEROUS_APPLIANCES:
                detected_appliances.add(cls_name)

    alert_required = bool(detected_appliances) and person_count == 0
    return {
        "alert_required": alert_required,
        "appliances": list(detected_appliances),
        "person_count": person_count,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
