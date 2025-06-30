from flask import session
from db.mongo import get_user_by_username
import re

# ------------------ Utility Functions ------------------

def sanitize_input(text):
    return text.strip()

def is_valid_username(username):
    return bool(re.match(r"^[A-Za-z0-9_]{3,30}$", username))

def is_valid_password(password):
    return len(password) >= 4

# ------------------ Session Management ------------------

def login_user(username, password):
    user = get_user_by_username(username)
    if user and user.get('password') == password:
        user['_id'] = str(user['_id'])  # Convert ObjectId to string for session safety
        session['user'] = user
        session['username'] = username
        session['logged_in'] = True
        return True, "Login successful"
    return False, "Invalid credentials"

def logout_user():
    session.clear()  # Clears all session data

def is_logged_in():
    return session.get("logged_in", False)

def current_user():
    return session.get("username")

def current_user_data():
    return session.get("user")
