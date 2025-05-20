# 🔐 Violence & Scream Detection Surveillance System

A Flask-based web surveillance system for **real-time violence and scream detection** using:

- **YOLOv8** for object and violence detection from video
- **Custom Scream Detection** model using PyTorch
- **Live Webcam & Video File Processing**
- **Audio Validation for Screams**
- **Automatic Alerts** via **Email**, **WhatsApp**, and **SMS**

---

## 📸 Screenshots

### 1. 🏠 Login Page  
![Login Page](SCREENSHOT/scrn_.PNG)

### 2. 🏠 Signup Page  
![Signup Page](SCREENSHOT/scrn_2.PNG)

### 3. 🏠 Home Page  
![Home Page](screenshots/1_home_page.png)

### 2. 📹 Webcam Detection Page  
![Webcam Detection](screenshots/2_webcam_detection.png)

### 3. 🎞️ Video File Detection Page  
![Video Detection](screenshots/3_video_file_detection.png)

### 4. 🎧 Scream Detection Result  
![Scream Detected](screenshots/4_scream_detected.png)

### 5. 🚨 Violence Detection Result  
![Violence Detected](screenshots/5_violence_detected.png)

### 6. 📤 Alerts Sent (Email / SMS / WhatsApp)  
![Alerts Sent](screenshots/6_alerts_sent.png)

---

## ⚙️ Features

✅ Real-time **Webcam & Video File Detection**  
✅ Integrated **YOLOv8** for Object/Violence Detection  
✅ Custom **Scream Classifier** with PyTorch  
✅ **Audio Validation** using librosa  
✅ Sends Alerts via:
- 📧 Email (SMTP)
- 📞 WhatsApp (Twilio)
- 📲 SMS (Twilio)

---

## 🧠 Tech Stack

| Technology | Purpose |
|------------|---------|
| **Flask**  | Web framework |
| **YOLOv8** | Object detection (Ultralytics) |
| **OpenCV** | Video stream handling |
| **PyTorch** | Scream classification model |
| **librosa** | Audio validation |
| **Twilio API** | SMS and WhatsApp alerting |
| **smtplib** | Email alerts |

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/violence-scream-detection.git
cd violence-scream-detection


HOSTED WEBISTE ONLY FOR THE USER INTERFACE

