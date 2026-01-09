from flask import Flask, request, jsonify
import os
import joblib
from preprocessing import clean_text

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "email_classifier.pkl")
model = joblib.load(MODEL_PATH)


URGENT_KEYWORDS = [
    "urgent", "asap", "immediately", "today",
    "tomorrow", "deadline", "exam", "interview",
    "meeting", "payment", "due"
]

@app.route("/")
def home():
    return "✅ Smart Email Classification API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    email_text = data.get("text", "").strip()

    if not email_text:
        return jsonify({"error": "Email text is empty"}), 400

    
    cleaned = clean_text(email_text)

    
    prediction = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    confidence = round(float(max(proba)), 2)

    lower_text = email_text.lower()
    urgent = any(word in lower_text for word in URGENT_KEYWORDS)

    return jsonify({
        "email": email_text,
        "category": prediction,
        "confidence": confidence,
        "urgent": urgent
    })

if __name__ == "__main__":
    app.run(debug=True)
