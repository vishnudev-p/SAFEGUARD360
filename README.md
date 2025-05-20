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
![Home Page](SCREENSHOT/home.PNG)

### 4. 📹 Violence Detection Page  
![Violence Detection](SCREENSHOT/scrn_3.PNG)

### 5. 🎞️ Scream Detection Page  
![Scream Detection](SCREENSHOT/scrn_5.PNG)

### 6. 🎧 Scream Detection Result  
![Scream Detected](SCREENSHOT/scrn_6.PNG)

### 7. 🚨 Violence Detection Result  
![Violence Detected](SCREENSHOT/scrn_4.PNG)



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

