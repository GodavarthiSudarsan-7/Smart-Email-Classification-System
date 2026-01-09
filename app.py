from flask import Flask, request, jsonify, send_from_directory
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

AUTO_REPLIES = {
    "Important": "Thanks for the update. I will take care of this.",
    "Work": "Noted. I’ll review and get back to you.",
    "Promotions": "Thanks, I’ll check this out.",
    "Spam": None
}

def get_priority(category, urgent):
    if urgent:
        return 5
    if category == "Important":
        return 4
    if category == "Work":
        return 3
    if category == "Promotions":
        return 2
    return 1

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")

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

    urgent = any(word in email_text.lower() for word in URGENT_KEYWORDS)
    auto_reply = "Acknowledged. I’ll respond immediately." if urgent else AUTO_REPLIES.get(prediction)
    priority = get_priority(prediction, urgent)

    return jsonify({
        "email": email_text,
        "category": prediction,
        "confidence": confidence,
        "urgent": urgent,
        "priority": priority,
        "auto_reply": auto_reply
    })

if __name__ == "__main__":
    app.run(debug=True)
