# app.py
import os
import sqlite3
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import numpy as np

# --- App setup ---
app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET_KEY", "change_this_to_a_secure_random_key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "appdata.db")
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

# --- Database setup (SQLite) ---
# Using a single connection with check_same_thread=False for simplicity.
# For larger apps, prefer per-request connections via g/teardown_appcontext.
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    message TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id TEXT,
    amount REAL,
    result TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# --- Load ML model and encoders ---
fraud_model = None
encoders = None
state_options = []
type_options = []
channel_options = []

try:
    fraud_model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)

    # The encoders were trained on these exact columns:
    # "location_state", "Trans_type", "channel"
    state_options = list(encoders["location_state"].classes_)
    type_options = list(encoders["Trans_type"].classes_)
    channel_options = list(encoders["channel"].classes_)
except Exception as e:
    print(f"⚠️ Could not load model/encoders: {e}")

# --- Helpers ---
def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_email" not in session:
            flash("popup:Please login first.", "popup")
            return redirect(url_for("auth"))
        return func(*args, **kwargs)
    return wrapper

def admin_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("is_admin"):
            flash("popup:Admin access required.", "popup")
            return redirect(url_for("admin_login"))
        return func(*args, **kwargs)
    return wrapper

def safe_transform(enc_key: str, value: str) -> int:
    """Encode a categorical value using a saved LabelEncoder.
    Falls back to the first known class if the value is unseen."""
    try:
        le = encoders[enc_key]
        return int(le.transform([value])[0])
    except Exception:
        # Fallback: first class to avoid crashing on unseen inputs
        le = encoders[enc_key]
        return int(le.transform([le.classes_[0]])[0])

# --- Routes ---
@app.route("/")
def home():
    user_email = session.get("user_email")
    return render_template("home.html", user_email=user_email)

@app.route("/auth")
def auth():
    return render_template("auth.html")

@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()
    if email and password:
        user = cursor.execute(
            "SELECT * FROM users WHERE email=? AND password=?", (email, password)
        ).fetchone()
        if user:
            session["user_email"] = email
            flash(f"popup:Welcome, {email}!", "popup")
            return redirect(url_for("home"))
        else:
            flash("popup:Account not found. Please sign up first.", "popup")
            return redirect(url_for("auth"))
    flash("popup:Invalid credentials.", "popup")
    return redirect(url_for("auth"))

@app.route("/signup", methods=["POST"])
def signup():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()
    if name and email and password:
        try:
            cursor.execute(
                "INSERT INTO users (name,email,password) VALUES (?,?,?)",
                (name, email, password)
            )
            conn.commit()
            flash("popup:Account created successfully. Please login.", "popup")
        except sqlite3.IntegrityError:
            flash("popup:Email already registered. Please login.", "popup")
        return redirect(url_for("auth"))
    flash("popup:All fields required.", "popup")
    return redirect(url_for("auth"))

@app.route("/logout")
def logout():
    session.clear()
    flash("popup:Logged out successfully.", "popup")
    return redirect(url_for("home"))

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    # This route previews a CSV; it does not affect the model.
    import pandas as pd  # local import to avoid failure if pandas missing at startup
    preview = None
    if request.method == "POST":
        file = request.files.get("datafile")
        if file and file.filename:
            try:
                df = pd.read_csv(file)
                # Light cleanup for display
                df.columns = df.columns.str.strip()
                for col in df.columns:
                    if df[col].dtype == "object":
                        df[col] = df[col].astype(str).str.strip()
                preview = df.head().to_html(classes="table table-striped table-sm", index=False)
                flash("popup:File uploaded successfully.", "popup")
            except Exception as e:
                flash(f"popup:Error reading CSV: {e}", "popup")
        else:
            flash("popup:Please select a CSV file.", "popup")
    return render_template("upload.html", preview=preview)

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    # Render form with known options (keeps inputs consistent with encoders)
    result = None
    proba = None

    if request.method == "POST":
        txn_id = request.form.get("txn_id", "").strip()
        amount_str = request.form.get("amount", "").strip()
        state = (request.form.get("state", state_options[0]) or "").strip()
        ttype = (request.form.get("ttype", type_options[0]) or "").strip()
        channel = (request.form.get("channel", channel_options[0]) or "").strip()

        # Validate amount
        try:
            amount = float(amount_str)
        except (ValueError, TypeError):
            flash("popup:Enter a valid numeric amount.", "popup")
            return render_template(
                "predict.html",
                result=None,
                proba=None,
                states=state_options,
                types=type_options,
                channels=channel_options,
                last_inputs={"txn_id": txn_id, "amount": amount_str, "state": state, "ttype": ttype, "channel": channel}
            )

        # Predict using model (if loaded)
        if fraud_model and encoders:
            try:
                # Encode categoricals using saved encoders (trained on: location_state, Trans_type, channel)
                state_enc = safe_transform("location_state", state)
                type_enc = safe_transform("Trans_type", ttype)
                channel_enc = safe_transform("channel", channel)

                # Maintain training feature order:
                # [Transaction_amt, location_state, Trans_type, channel]
                features = np.array([[amount, state_enc, type_enc, channel_enc]])

                pred = int(fraud_model.predict(features)[0])
                label = "Fraudulent" if pred == 1 else "Clear"

                # Optional: show probabilities if classifier supports it
                try:
                    proba_arr = fraud_model.predict_proba(features)[0]
                    proba = {"Clear": round(float(proba_arr[0]), 3), "Fraudulent": round(float(proba_arr[1]), 3)}
                except Exception:
                    proba = None

                result = f"⚠️ Transaction {txn_id} is Fraudulent!" if pred == 1 else f"✅ Transaction {txn_id} looks Safe."
            except Exception as e:
                result = f"⚠️ Prediction error: {e}"
        else:
            result = "⚠️ Fraud model not loaded. Please train the model first."

        # Save to DB
        try:
            cursor.execute(
                "INSERT INTO predictions (txn_id, amount, result) VALUES (?,?,?)",
                (txn_id, amount, result)
            )
            conn.commit()
        except Exception as e:
            flash(f"popup:DB save error: {e}", "popup")

        flash(f"popup:{result}", "popup")

        return render_template(
            "predict.html",
            result=result,
            proba=proba,
            states=state_options,
            types=type_options,
            channels=channel_options,
            last_inputs={"txn_id": txn_id, "amount": amount, "state": state, "ttype": ttype, "channel": channel}
        )

    # GET: render empty form with options
    return render_template(
        "predict.html",
        result=None,
        proba=None,
        states=state_options,
        types=type_options,
        channels=channel_options,
        last_inputs=None
    )

@app.route("/faq")
def faq():
    return render_template("faq.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()
        if email and message:
            cursor.execute("INSERT INTO feedback (email,message) VALUES (?,?)", (email, message))
            conn.commit()
            flash("popup:Message sent successfully.", "popup")
        else:
            flash("popup:Both fields required.", "popup")
    return render_template("contact.html")

# --- Admin Routes ---
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "secure123")  # change this to your own

@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        password = request.form.get("password", "")
        if password == ADMIN_PASSWORD:
            session["is_admin"] = True
            flash("popup:Admin login successful.", "popup")
            return redirect(url_for("admin"))
        else:
            flash("popup:Invalid admin password.", "popup")
    return render_template("admin_login.html")

@app.route("/admin")
@admin_required
def admin():
    count_predictions = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    users = cursor.execute("SELECT name, email FROM users ORDER BY id DESC").fetchall()
    feedbacks = cursor.execute("SELECT email, message FROM feedback ORDER BY id DESC").fetchall()
    return render_template(
        "admin.html",
        count_predictions=count_predictions,
        users=users,
        feedbacks=feedbacks
    )

if __name__ == "__main__":
    # For local testing
    app.run(host="127.0.0.1", port=5000, debug=False)
