import os
import json
import cv2
import torch
import torchaudio
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, jsonify
from ultralytics import YOLO, settings
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from email.message import EmailMessage
import smtplib
from twilio.rest import Client
from torchvision import transforms
from config import CONFIG
import base64
from io import BytesIO
import wave
import requests

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Set YOLO settings for Render
settings.update({"runs_dir": os.path.join(os.environ.get("UPLOAD_FOLDER", "Uploads"), "yolo_runs")})

# Ensure upload folder exists
os.makedirs(CONFIG["upload_folder"], exist_ok=True)

def download_model(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading model from {url} to {dest}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Model downloaded to {dest}")

class DetectionSystem:
    def __init__(self):
        violence_model_path = CONFIG["violence_model_path"]
        scream_model_path = CONFIG["scream_model_path"]
        # Download models if they don't exist
        download_model("https://drive.google.com/uc?export=download&id=1MZUAGLDUWXR1-c8lsWU0Z8VSBEOM_ZYQ", violence_model_path)
        download_model("https://drive.google.com/uc?export=download&id=1qgTqA0QiMCwTNzzY86GhpsIFg9M1Y4tK", scream_model_path)
        try:
            self.violence_model = YOLO(violence_model_path)
        except FileNotFoundError:
            print(f"❌ Violence model not found at {violence_model_path}")
            raise
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.scream_model = torch.load(scream_model_path, map_location=self.device, weights_only=False)
        except FileNotFoundError:
            print(f"❌ Scream model not found at {scream_model_path}")
            raise
        self.scream_model.eval()
        self.registered_users = self.load_users()
        self.is_streaming = False  # Flag for webcam streaming
        self.cap = None  # Webcam capture object
        self.is_upload_streaming = False  # Flag for uploaded video streaming
        self.upload_cap = None  # Capture object for uploaded video
        self.current_upload_path = None  # Path to current uploaded video

    def load_users(self):
        if os.path.exists(CONFIG["user_file"]):
            with open(CONFIG["user_file"], "r") as file:
                return json.load(file)
        return {}

    def save_users(self):
        with open(CONFIG["user_file"], "w") as file:
            json.dump(self.registered_users, file)

    def send_email_alert(self, image_path=None, message="Violence has been detected!"):
        msg = EmailMessage()
        msg["Subject"] = "🚨 Alert!"
        msg["From"] = CONFIG["email"]["sender"]
        msg["To"] = CONFIG["email"]["receiver"]
        msg.set_content(message)

        if image_path:
            with open(image_path, "rb") as f:
                msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="alert.jpg")

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(CONFIG["email"]["sender"], CONFIG["email"]["password"])
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"❌ Email Sending Failed: {e}")
            return False

    def send_whatsapp_alert(self, message):
        try:
            client = Client(CONFIG["twilio"]["account_sid"], CONFIG["twilio"]["auth_token"])
            message = client.messages.create(
                body=message,
                from_=CONFIG["twilio"]["whatsapp_number"],
                to=CONFIG["twilio"]["recipient_whatsapp"]
            )
            return True
        except Exception as e:
            print(f"❌ WhatsApp Sending Failed: {e}")
            return False

    def send_sms_alert(self, message):
        try:
            client = Client(CONFIG["twilio"]["account_sid"], CONFIG["twilio"]["auth_token"])
            message = client.messages.create(
                body=message,
                from_=CONFIG["twilio"]["whatsapp_number"].replace("whatsapp:", ""),
                to=CONFIG["twilio"]["recipient_phone"]
            )
            return True
        except Exception as e:
            print(f"❌ SMS Sending Failed: {e}")
            return False

    def detect_violence(self, video_path):
        cap = cv2.VideoCapture(video_path)
        violence_detected = False
        image_base64 = None

        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return False, None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.violence_model(frame)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = self.violence_model.names[cls]

                    if label.lower() == "violence" and conf > 0.50:
                        violence_detected = True
                        color = (0, 0, 255)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(CONFIG["upload_folder"], f"alert_{timestamp}.jpg")
                        cv2.imwrite(output_path, frame)
                        self.send_email_alert(output_path)
                        self.send_whatsapp_alert("🚨 Violence Alert! Check your email.")
                        with open(output_path, "rb") as f:
                            image_base64 = base64.b64encode(f.read()).decode('utf-8')
                        os.remove(output_path)
                    else:
                        color = (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cap.release()
        return violence_detected, image_base64

    def generate_frames(self):
        self.is_streaming = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Failed to open webcam")
            self.is_streaming = False
            return

        while self.is_streaming:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.violence_model(frame)
            violence_detected = False

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = self.violence_model.names[cls]

                    if label.lower() == "violence" and conf > 0.50:
                        violence_detected = True
                        color = (0, 0, 255)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(CONFIG["upload_folder"], f"alert_{timestamp}.jpg")
                        cv2.imwrite(output_path, frame)
                        self.send_email_alert(output_path)
                        self.send_whatsapp_alert("🚨 Violence Alert! Check your email.")
                        os.remove(output_path)
                    else:
                        color = (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        self.cap.release()
        self.cap = None
        self.is_streaming = False

    def stop_streaming(self):
        self.is_streaming = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def generate_upload_frames(self, video_path):
        print(f"Attempting to open video: {video_path}")
        self.is_upload_streaming = True
        self.current_upload_path = video_path
        self.upload_cap = cv2.VideoCapture(video_path)
        if not self.upload_cap.isOpened():
            print(f"❌ Failed to open uploaded video: {video_path}")
            self.is_upload_streaming = False
            return

        while self.is_upload_streaming:
            ret, frame = self.upload_cap.read()
            if not ret:
                print("End of video or error reading frame")
                break

            results = self.violence_model(frame)
            violence_detected = False

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = self.violence_model.names[cls]

                    if label.lower() == "violence" and conf > 0.50:
                        violence_detected = True
                        color = (0, 0, 255)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(CONFIG["upload_folder"], f"alert_{timestamp}.jpg")
                        cv2.imwrite(output_path, frame)
                        self.send_email_alert(output_path)
                        self.send_whatsapp_alert("🚨 Violence Alert! Check your email.")
                        os.remove(output_path)
                    else:
                        color = (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                break
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        print("Stopping upload stream")
        self.upload_cap.release()
        self.upload_cap = None
        self.is_upload_streaming = False
        if os.path.exists(video_path):
            os.remove(video_path)
            self.current_upload_path = None

    def stop_upload_streaming(self):
        self.is_upload_streaming = False
        if self.upload_cap:
            self.upload_cap.release()
            self.upload_cap = None
        if self.current_upload_path and os.path.exists(self.current_upload_path):
            os.remove(self.current_upload_path)
            self.current_upload_path = None

    def pad_waveform(self, waveform, target_length):
        num_channels, current_length = waveform.shape
        if current_length < target_length:
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform

    def transform_audio_to_image(self, audio, sample_rate, temp_path="temp_audio_img.png"):
        audio = self.pad_waveform(audio, 441000)
        spectrogram_tensor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=64, n_fft=1024
        )(audio)[0] + 1e-10
        plt.imsave(temp_path, spectrogram_tensor.log2().numpy(), cmap='viridis')
        return temp_path

    def validate_wav_file(self, file_path):
        try:
            with wave.open(file_path, 'rb') as wav_file:
                # Check basic WAV properties
                if wav_file.getnchannels() not in (1, 2):
                    return False, "Unsupported number of channels"
                if wav_file.getsampwidth() not in (1, 2, 4):
                    return False, "Unsupported sample width"
                if wav_file.getframerate() not in (8000, 16000, 22050, 44100, 48000):
                    return False, "Unsupported sample rate"
            return True, "Valid WAV file"
        except Exception as e:
            return False, f"Invalid WAV file: {str(e)}"

    def detect_scream(self, file_path):
        try:
            # Validate WAV file
            is_valid, validation_message = self.validate_wav_file(file_path)
            if not is_valid:
                print(f"❌ WAV Validation Failed: {validation_message}")
                return False, validation_message

            audio, sample_rate = torchaudio.load(file_path)
            print(f"Loaded audio: channels={audio.shape[0]}, samples={audio.shape[1]}, sample_rate={sample_rate}")
            temp_image_path = self.transform_audio_to_image(audio, sample_rate)
            transform = transforms.Compose([
                transforms.Resize((64, 862)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x[:3, :, :])
            ])
            image = Image.open(temp_image_path)
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.scream_model(image_tensor)
                prediction = outputs.argmax(dim=1).cpu().numpy()[0]
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            scream_detected = bool(prediction == 1)  # Convert numpy.bool_ to Python bool
            if scream_detected:
                self.send_sms_alert("🚨 Scream Detected! High Risk Situation!")
                self.send_email_alert(message="Scream Detected! High Risk Situation!")
                self.send_whatsapp_alert("🚨 Scream Detected! High Risk Situation!")
            return scream_detected, "Scream" if prediction == 1 else "Non-Scream"
        except Exception as e:
            print(f"❌ Scream Detection Error: {e}")
            return False, f"Error processing audio: {str(e)}"

detection_system = DetectionSystem()

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in detection_system.registered_users and detection_system.registered_users[username] == password:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        flash('Invalid username or password!', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if not all([username, email, password, confirm_password]):
            flash('All fields are required!', 'danger')
        elif password != confirm_password:
            flash('Passwords do not match!', 'danger')
        elif username in detection_system.registered_users:
            flash('Username already exists!', 'danger')
        else:
            detection_system.registered_users[username] = password
            detection_system.save_users()
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

@app.route('/violence', methods=['GET', 'POST'])
def violence():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'video' not in request.files:
            return jsonify({"success": False, "message": "No video file provided"}), 400
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"success": False, "message": "No file selected"}), 400
        if not video_file.filename.endswith('.mp4'):
            return jsonify({"success": False, "message": "Only MP4 files are allowed"}), 400
        try:
            video_path = os.path.join(CONFIG["upload_folder"], video_file.filename)
            video_file.save(video_path)
            detection_system.current_upload_path = video_path
            return jsonify({"success": True, "message": "Video uploaded successfully"})
        except Exception as e:
            print(f"❌ Upload Error: {e}")
            return jsonify({"success": False, "message": f"Upload failed: {str(e)}"}), 500
    return render_template('violence.html', image_base64=None)

@app.route('/video_feed')
def video_feed():
    if 'username' not in session:
        return redirect(url_for('login'))
    if os.environ.get('RENDER'):  # Check if running on Render
        return jsonify({"error": "Webcam streaming is not supported on this server"}), 503
    return Response(detection_system.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    if 'username' not in session:
        return redirect(url_for('login'))
    if os.environ.get('RENDER'):  # Check if running on Render
        return jsonify({"message": "Webcam streaming not available on this server"}), 200
    detection_system.stop_streaming()
    return '', 204

@app.route('/video_upload_feed')
def video_upload_feed():
    if 'username' not in session:
        return redirect(url_for('login'))
    filename = request.args.get('filename')
    if not filename or not detection_system.current_upload_path:
        return jsonify({"error": "No video uploaded"}), 400
    video_path = detection_system.current_upload_path
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404
    return Response(detection_system.generate_upload_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_upload_analysis', methods=['POST'])
def stop_upload_analysis():
    if 'username' not in session:
        return redirect(url_for('login'))
    detection_system.stop_upload_streaming()
    return '', 204

@app.route('/scream', methods=['GET', 'POST'])
def scream():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'audio' not in request.files:
            return jsonify({"success": False, "message": "No audio file provided"}), 400
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"success": False, "message": "No file selected"}), 400
        if not audio_file.filename.endswith('.wav'):
            return jsonify({"success": False, "message": "Only WAV files are allowed"}), 400
        try:
            audio_path = os.path.join(CONFIG["upload_folder"], audio_file.filename)
            audio_file.save(audio_path)
            scream_detected, result = detection_system.detect_scream(audio_path)
            os.remove(audio_path)
            if scream_detected:
                flash('Scream detected! Alerts sent.', 'danger')
            else:
                flash('No scream detected.', 'success')
            return jsonify({
                "success": bool(scream_detected or result == "Non-Scream"),  # Ensure Python bool
                "message": "Analysis complete",
                "result": result
            })
        except Exception as e:
            print(f"❌ Scream Detection Error: {e}")
            return jsonify({"success": False, "message": f"Analysis failed: {str(e)}"}), 500
    return render_template('scream.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Render's PORT or default to 5000
    app.run(host='0.0.0.0', port=port, debug=False)  # Disable debug for production
