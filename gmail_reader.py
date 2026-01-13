import os
import base64
import pickle
import requests
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

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

def extract_text(message):
    payload = message["payload"]
    parts = payload.get("parts", [])

    for part in parts:
        if part["mimeType"] == "text/plain":
            data = part["body"]["data"]
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
    results = service.users().messages().list(
        userId="me", maxResults=5
    ).execute()

    messages = results.get("messages", [])

    for msg in messages:
        full_msg = service.users().messages().get(
            userId="me", id=msg["id"]
        ).execute()

        text = extract_text(full_msg)
        if text:
            result = classify_email(text)
            print(result)

if __name__ == "__main__":
    main()
