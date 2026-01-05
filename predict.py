import os
import joblib
from preprocessing import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "email_classifier.pkl")

model = joblib.load(MODEL_PATH)

def predict_email(text):
    text = clean_text(text)
    return model.predict([text])[0]

if __name__ == "__main__":
    email = "Your interview is scheduled tomorrow"
    print("Prediction:", predict_email(email))
