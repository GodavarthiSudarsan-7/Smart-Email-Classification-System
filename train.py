import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from preprocessing import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "emails.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "email_classifier.pkl")


if not os.path.exists(DATA_PATH):
    df = pd.DataFrame({
        "text": [
            "Meeting scheduled tomorrow",
            "Your bank OTP is 123456",
            "Win a free iPhone now",
            "Limited offer buy now",
            "Exam timetable released",
            "Assignment deadline extended"
        ],
        "label": [
            "Important",
            "Important",
            "Spam",
            "Promotions",
            "Work",
            "Work"
        ]
    })
    df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)

df["text"] = df["text"].apply(clean_text)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

model.fit(df["text"], df["label"])
joblib.dump(model, MODEL_PATH)

print("✅ Model trained successfully")
