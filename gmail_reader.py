import os
import base64
import pickle
import requests
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def get_service():
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    return build("gmail", "v1", credentials=creds)

def get_or_create_label(service, label_name):
    labels = service.users().labels().list(userId="me").execute().get("labels", [])
    for label in labels:
        if label["name"] == label_name:
            return label["id"]

    label_body = {
        "name": label_name,
        "labelListVisibility": "labelShow",
        "messageListVisibility": "show"
    }

    label = service.users().labels().create(
        userId="me", body=label_body
    ).execute()

    return label["id"]

def extract_text(message):
    payload = message["payload"]
    parts = payload.get("parts", [])

    for part in parts:
        if part["mimeType"] == "text/plain":
            data = part["body"].get("data")
            if data:
                return base64.urlsafe_b64decode(data).decode()

    return ""

def classify_email(text):
    response = requests.post(
        "http://127.0.0.1:5000/predict",
        json={"text": text}
    )
    return response.json()

def main():
    service = get_service()

    important_label = get_or_create_label(service, "AI-Important")
    promo_label = get_or_create_label(service, "AI-Promotions")
    spam_label = get_or_create_label(service, "AI-Spam")

    results = service.users().messages().list(
        userId="me", maxResults=5
    ).execute()

    messages = results.get("messages", [])

    for msg in messages:
        full_msg = service.users().messages().get(
            userId="me", id=msg["id"]
        ).execute()

        text = extract_text(full_msg)
        if not text:
            continue

        result = classify_email(text)
        category = result["category"]

        label_id = None
        if category in ["Important", "Work"]:
            label_id = important_label
        elif category == "Promotions":
            label_id = promo_label
        elif category == "Spam":
            label_id = spam_label

        if label_id:
            service.users().messages().modify(
                userId="me",
                id=msg["id"],
                body={"addLabelIds": [label_id]}
            ).execute()

            print("Labeled:", category)

if __name__ == "__main__":
    main()
