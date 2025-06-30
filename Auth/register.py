# db/mongo.py

from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['hotspot']
users_collection = db['users']


def get_user_by_username(username):
    return users_collection.find_one({"username": username})


def register_user(name, username, password, phone_number, trusted_contacts):
    """
    Registers a new user in MongoDB.
    """
    if get_user_by_username(username):
        return False, "🚫 Username already exists."

    user_data = {
        "name": name,
        "username": username,
        "password": password,  # In production, always hash this!
        "phone_number": phone_number,
        "trusted_contacts": trusted_contacts
    }

    try:
        users_collection.insert_one(user_data)
        return True, "✅ Registration successful."
    except Exception as e:
        return False, f"❌ Registration failed: {str(e)}"
