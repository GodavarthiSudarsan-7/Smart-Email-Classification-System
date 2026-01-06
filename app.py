from flask import Flask, request, jsonify
import os
import joblib
from preprocessing import clean_text

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "email_classifier.pkl")


model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return "✅ Smart Email Classification API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    email_text = data.get("text", "")

    if not email_text.strip():
        return jsonify({"error": "Email text is empty"}), 400

    cleaned = clean_text(email_text)
    prediction = model.predict([cleaned])[0]

    return jsonify({
        "email": email_text,
        "category": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)
