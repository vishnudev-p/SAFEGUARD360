import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "twilio": {
        "account_sid": os.getenv("TWILIO_ACCOUNT_SID"),
        "auth_token": os.getenv("TWILIO_AUTH_TOKEN"),
        "whatsapp_number": "whatsapp:+14155238888",
        "recipient_whatsapp": "whatsapp:+918714198416",
        "recipient_phone": "+918714198416"
    },
    "email": {
        "sender": os.getenv("EMAIL_SENDER"),
        "password": os.getenv("EMAIL_PASSWORD"),
        "receiver": os.getenv("EMAIL_RECEIVER")
    },
    "violence_model_path": "models/my_model1.pt",
    "scream_model_path": "models/Resnet34_Model_2023-12-05--08-37-20.pt",
    "user_file": "users.json",
    "upload_folder": "uploads"
}