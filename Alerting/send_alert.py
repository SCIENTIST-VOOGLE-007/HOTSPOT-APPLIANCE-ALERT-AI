# utils/send_alert.py

import requests
import json
import os

# You can load these from .env in production
ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN_HERE")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "YOUR_PHONE_NUMBER_ID")

WHATSAPP_URL = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json"
}

def send_whatsapp_alert(phone_numbers, alert_text):
    """
    Send WhatsApp message to a list of phone numbers.
    phone_numbers: list of strings (E.164 format)
    alert_text: message body
    """
    for number in phone_numbers:
        payload = {
            "messaging_product": "whatsapp",
            "to": number,
            "type": "text",
            "text": {
                "body": alert_text
            }
        }
        try:
            response = requests.post(WHATSAPP_URL, headers=HEADERS, data=json.dumps(payload))
            if response.status_code == 200:
                print(f"✅ WhatsApp alert sent to {number}")
            else:
                print(f"❌ Failed to send alert to {number}: {response.text}")
        except Exception as e:
            print(f"⚠️ Error sending message to {number}: {e}")
