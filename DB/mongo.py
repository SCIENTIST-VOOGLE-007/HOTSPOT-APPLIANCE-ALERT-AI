

from pymongo import MongoClient
import os
from dotenv import load_dotenv


# Load credentials from .env
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)

# Create/use DB and collections
db = client["hotspot"]
users_collection = db["users"]
logs_collection = db["detections"]

# -------------------- USER HANDLING --------------------

def register_user(username, password, phone_number, trusted_contacts):
    existing = users_collection.find_one({"username": username})
    if existing:
        return False, "User already exists"

    user_data = {
        "username": username,
        "password": password,  # You can hash this in production
        "phone": phone_number,
        "trusted_contacts": trusted_contacts
    }
    users_collection.insert_one(user_data)
    return True, "Registration successful"

def authenticate_user(username, password):
    user = users_collection.find_one({"username": username, "password": password})
    if user:
        return True, user
    return False, None

def get_user_contacts(username):
    user = users_collection.find_one({"username": username})
    if user:
        return user.get("trusted_contacts", []), user.get("phone")
    return [], None

# -------------------- DETECTION LOGGING --------------------

def log_detection(username, analysis_result):
    log = {
        "username": username,
        "timestamp": analysis_result["timestamp"],
        "appliances": analysis_result["appliances"],
        "person_count": analysis_result["person_count"],
        "alert_triggered": analysis_result["alert_required"]
    }
    logs_collection.insert_one(log)

def get_user_by_username(username):
    return users_collection.find_one({"username": username})
