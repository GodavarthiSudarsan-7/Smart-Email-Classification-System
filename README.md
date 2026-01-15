# Smart Email Classification System

An AI-powered Gmail automation system that reads emails, classifies them using Machine Learning (NLP), automatically applies Gmail labels, archives spam, and stores classification history for analytics.

---

## Features

- Gmail OAuth integration (secure access)
- AI-based email classification (Important, Work, Promotions, Spam)
- Automatic Gmail labeling
- Automatic spam archiving
- Confidence-aware review labeling (AI-Review)
- Stores classification history in CSV
- Analytics dashboard to visualize results
- Safe on-demand execution (no background battery drain)

---

## Tech Stack

- Python
- Flask (API + Dashboard)
- Scikit-learn (NLP / ML)
- Gmail API
- OAuth 2.0
- HTML (Dashboard UI)

---

## System Architecture

Gmail Inbox  
→ Gmail API  
→ AI Classification Service (Flask + ML)  
→ Gmail Labels & Archive  
→ CSV Storage  
→ Dashboard View

---

## How It Works

1. Gmail emails are accessed using OAuth authentication.
2. Email content is sent to an AI model via a Flask API.
3. The AI model predicts category, priority, urgency, and confidence.
4. Gmail labels are applied automatically.
5. Spam emails are archived.
6. All actions are logged in a CSV file.
7. A dashboard displays analytics and recent activity.

---

## How to Run

### Step 1: Start AI API
```bash
python app.py


Step 2: Process Emails
python gmail_reader.py


Step 3: View Dashboard
python dashboard.py


opens :- http://127.0.0.1:5001


Safety & Efficiency

No continuous background execution

No frequent polling

Laptop-friendly design

Can be extended to cloud or push-based systems



Future Improvements

Gmail push notifications (event-driven)

Cloud deployment

Database storage

Advanced ML models

User preference learning