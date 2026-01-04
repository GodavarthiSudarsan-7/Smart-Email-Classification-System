import os
import joblib
from preprocessing import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "email_classifier.pkl")

model = joblib.load(MODEL_PATH)

def predict_email(text):
    cleaned = clean_text(text)
    return model.predict([cleaned])[0]

if __name__ == "__main__":
    test_email = "Your interview is scheduled tomorrow at 10 AM"
    print("📧 Email:", test_email)
    print("🔮 Prediction:", predict_email(test_email))
