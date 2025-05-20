import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed, using environment variables directly")

CONFIG = {
    "upload_folder": os.environ.get("UPLOAD_FOLDER", "Uploads"),
    "user_file": os.environ.get("USER_FILE", "users.json"),
    "violence_model_path": os.environ.get("VIOLENCE_MODEL_PATH", "/app/models/my_model1.pt"),
    "scream_model_path": os.environ.get("SCREAM_MODEL_PATH", "/app/models/Resnet34_Model_2023-12-05--08-37-20.pt"),
    "email": {
        "sender": os.environ.get("EMAIL_SENDER"),
        "receiver": os.environ.get("EMAIL_RECEIVER"),
        "password": os.environ.get("EMAIL_PASSWORD")
    },
    "twilio": {
        "account_sid": os.environ.get("TWILIO_ACCOUNT_SID"),
        "auth_token": os.environ.get("TWILIO_AUTH_TOKEN"),
        "whatsapp_number": os.environ.get("TWILIO_WHATSAPP_NUMBER"),
        "recipient_whatsapp": os.environ.get("TWILIO_RECIPIENT_WHATSAPP"),
        "recipient_phone": os.environ.get("TWILIO_RECIPIENT_PHONE")
    }
}
