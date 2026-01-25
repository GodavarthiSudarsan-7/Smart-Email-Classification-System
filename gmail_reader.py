import os
import base64
import pickle
import requests
import csv
from datetime import datetime
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
CSV_FILE = "email_history.csv"

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

    label = service.users().labels().create(
        userId="me",
        body={
            "name": label_name,
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show"
        }
    ).execute()

    return label["id"]

def extract_text(message):
    payload = message.get("payload", {})
    parts = payload.get("parts", [])

    for part in parts:
        if part.get("mimeType") == "text/plain":
            data = part.get("body", {}).get("data")
            if data:
                return base64.urlsafe_b64decode(data).decode(errors="ignore")

    return ""

def extract_headers(message):
    headers = message.get("payload", {}).get("headers", [])
    sender = ""
    subject = ""

    for h in headers:
        if h.get("name") == "From":
            sender = h.get("value")
        elif h.get("name") == "Subject":
            subject = h.get("value")

    return sender, subject

def classify_email(text):
    response = requests.post(
        "http://127.0.0.1:5000/predict",
        json={"text": text}
    )
    return response.json()

def create_reply_draft(service, message, reply_text):
    headers = message.get("payload", {}).get("headers", [])
    to = ""
    subject = ""

    for h in headers:
        if h.get("name") == "From":
            to = h.get("value")
        elif h.get("name") == "Subject":
            subject = h.get("value")

    raw = f"To: {to}\r\nSubject: Re: {subject}\r\n\r\n{reply_text}"
    encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("utf-8")

    service.users().drafts().create(
        userId="me",
        body={
            "message": {
                "raw": encoded
            }
        }
    ).execute()

def save_to_csv(message_id, result, sender, subject):
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "message_id",
                "sender",
                "subject",
                "category",
                "priority",
                "urgent",
                "confidence"
            ])
        writer.writerow([
            datetime.now().isoformat(),
            message_id,
            sender,
            subject,
            result.get("category"),
            result.get("priority"),
            result.get("urgent"),
            result.get("confidence")
        ])

def main():
    service = get_service()

    important_label = get_or_create_label(service, "AI-Important")
    promo_label = get_or_create_label(service, "AI-Promotions")
    spam_label = get_or_create_label(service, "AI-Spam")
    review_label = get_or_create_label(service, "AI-Review")

    p1_label = get_or_create_label(service, "AI-P1-Urgent")
    p2_label = get_or_create_label(service, "AI-P2-High")
    p3_label = get_or_create_label(service, "AI-P3-Normal")
    p4_label = get_or_create_label(service, "AI-P4-Low")

    results = service.users().messages().list(
        userId="me",
        maxResults=5
    ).execute()

    messages = results.get("messages", [])

    for msg in messages:
        full_msg = service.users().messages().get(
            userId="me",
            id=msg["id"]
        ).execute()

        text = extract_text(full_msg)
        if not text:
            continue

        sender, subject = extract_headers(full_msg)
        result = classify_email(text)

        category = result.get("category")
        confidence = result.get("confidence", 1)
        priority = result.get("priority", 4)
        auto_reply = result.get("auto_reply", "")

        body = {}

        if confidence < 0.4:
            body["addLabelIds"] = [review_label]
        else:
            labels = []

            if category in ["Important", "Work"]:
                labels.append(important_label)
            elif category == "Promotions":
                labels.append(promo_label)
            elif category == "Spam":
                labels.append(spam_label)
                body["removeLabelIds"] = ["INBOX"]

            if priority == 1:
                labels.append(p1_label)
            elif priority == 2:
                labels.append(p2_label)
            elif priority == 3:
                labels.append(p3_label)
            else:
                labels.append(p4_label)

            body["addLabelIds"] = labels

        if body:
            service.users().messages().modify(
                userId="me",
                id=msg["id"],
                body=body
            ).execute()

            if category in ["Important", "Work"] and priority <= 2 and auto_reply:
                create_reply_draft(service, full_msg, auto_reply)

            save_to_csv(msg["id"], result, sender, subject)
            print("Labeled:", category, "Priority:", priority)

if __name__ == "__main__":
    main()
