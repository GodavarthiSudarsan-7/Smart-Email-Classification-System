from flask import Flask, request, jsonify
import os
import joblib
from preprocessing import clean_text

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "email_classifier.pkl")

model = joblib.load(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email_text = data.get("text", "")
    cleaned = clean_text(email_text)
    prediction = model.predict([cleaned])[0]

    return jsonify({
        "email": email_text,
        "category": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)
