# Vista-Hackathon
# 🔥 HotSpot – Appliance Alert AI

**Because safety should never be accidental.**

HotSpot is an AI-powered intelligent monitoring system that detects potential fire hazards from household appliances like stoves, ovens, or irons left ON in an empty room — and instantly alerts you or your trusted contacts to prevent disasters. 🚨

Built as a response to real-world safety concerns, HotSpot acts as your virtual safety assistant, constantly watching over your appliances through webcam or CCTV feeds. It combines the power of **Computer Vision**, **YOLO Object Detection**, and **Smart Alert Systems** to give you peace of mind, wherever you are.

---

## 🌟 Key Features

- ✅ **Real-time Object Detection** using YOLOv8
- 🎥 **Webcam / Video Feed Monitoring**
- 📸 **Single Image & Video Upload Detection**
- 🧠 **Appliance + Person Contextual Awareness**
- 🚫 **Alert Trigger if Appliance is ON but No Person Nearby**
- 📱 **WhatsApp Notifications** to user & trusted contacts (via Twilio API)
- 🔐 **User Login & Registration**
- 🌐 **Flask Web App Interface**

---

## 🛠️ Tech Stack

| Layer        | Technologies Used                     |
|--------------|----------------------------------------|
| **Frontend** | HTML, CSS, Bootstrap (Jinja templates) |
| **Backend**  | Python, Flask                          |
| **Database** | MongoDB (for user data & trusted contacts) |
| **AI Model** | YOLOv8 – trained to detect appliances + humans |
| **Notification** | Twilio WhatsApp API                 |
| **Dev Tools**| VSCode, GitHub, Render                 |

---

## 💡 How It Works

1. 🔓 User logs in (or registers with trusted contact numbers).
2. 🖼️ Uploads an image or video OR enables webcam.
3. 🤖 YOLOv8 detects appliances (like stoves, ovens) and people.
4. 🧠 If appliance is ON but no person detected → **Potential Hazard**.
5. 📲 Sends alert messages via WhatsApp to the registered user and trusted contacts.

---

## 🧪 Detection Logic

| Detected | Appliance ON | Person Nearby | Action            |
|----------|--------------|----------------|-------------------|
| ❌       | No           | -              | No alert          |
| ✅       | Yes          | ✅ Yes         | No alert          |
| ✅       | Yes          | ❌ No          | 🚨 Send Alert      |

---

## 🎯 Use Cases

- 👵 **Elderly Care** – Automatically notify family if appliances are left on unattended.
- 🏠 **Smart Homes** – Add an extra layer of AI-driven safety.
- 🏢 **Offices / PGs** – Detect fire hazards in shared spaces.
- 🔬 **Hackathons & Research** – Real-world application of AI for social good.

---

## 🚀 Getting Started
![image](https://github.com/user-attachments/assets/f99873a1-ce5c-437b-9d84-cf17fc6e0f86)
![image](https://github.com/user-attachments/assets/6981a82b-8648-45f2-89d4-10c0bf62e3ce)
![image](https://github.com/user-attachments/assets/ef0eb2b5-fce9-4571-9532-261bedf1d118)
![image](https://github.com/user-attachments/assets/c66c0bbf-fc06-4fd7-b1a0-8f73413bc743)
![image](https://github.com/user-attachments/assets/8067af0e-e858-471a-a341-b6076c15e066)
![image](https://github.com/user-attachments/assets/f0dbefe9-0306-4889-8d13-fa703a18b7d8)
![image](https://github.com/user-attachments/assets/d26d50ec-dc1a-4565-a17f-865da651134e)
![image](https://github.com/user-attachments/assets/c0905dba-60ca-4a37-852c-140fe5a464cc)


### 🔧 Installation

```bash
git clone https://github.com/your-username/hotspot-appliance-alert-ai.git
cd hotspot-appliance-alert-ai
pip install -r requirements.txt
```
▶️ Run Locally
```bash
python app.py
```

Open your browser and go to http://127.0.0.1:5000/

🧠 Model Training
YOLOv8 trained on a custom dataset with labels: stove, oven, microwave, toaster, person.

Trained using Roboflow and Ultralytics API.

Accuracy: 94% mAP @ IoU 0.5

📲 Sample WhatsApp Alert
🚨 HotSpot Appliance Alert
An appliance is ON and no person is nearby in the monitored area.
Location: Kitchen – 2nd Floor
Time: 2025-04-13 11:47 AM
Please check immediately.

🙌 Team & Vision
HotSpot was built with a mission to prevent fire hazards through AI-powered automation. We're passionate about leveraging technology to make homes and communities safer, smarter, and more responsive.

“The best tech is invisible — it quietly protects, empowers, and cares.”


YOLOv8 – Ultralytics

Twilio WhatsApp API

Flask, MongoDB, OpenCV

📬 Want to Contribute?
We’re open to feedback, suggestions, and collaborations. Feel free to fork the repo and drop a PR or issue!

🛡️ License
MIT License
