from flask import Flask, render_template
import csv

app = Flask(__name__)

@app.route("/")
def dashboard():
    rows = []
    counts = {}

    try:
        with open("email_history.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
                c = r["category"]
                counts[c] = counts.get(c, 0) + 1
    except FileNotFoundError:
        pass

    return render_template(
        "dashboard.html",
        rows=rows[-10:],
        counts=counts,
        total=len(rows)
    )

if __name__ == "__main__":
    app.run(port=5001, debug=True)
