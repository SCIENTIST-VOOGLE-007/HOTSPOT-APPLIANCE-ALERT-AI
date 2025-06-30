# src/auth_utils.py

import bcrypt

def hash_password(plain_password):
    """
    Hash a plain password using bcrypt.
    Returns the hashed password as a UTF-8 string.
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password, hashed_password):
    """
    Verify that the provided plain password matches the stored hashed password.
    Returns True if it matches, False otherwise.
    """
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception as e:
        print("Password verification failed:", e)
        return False
