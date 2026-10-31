import os
import sqlite3
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
from transformers import pipeline

app = Flask(__name__)
app.secret_key = "supersecretkey"

MODEL_PATH = "delay_model.pkl"
SCALER_PATH = "scaler.pkl"

# -----------------------------
# Database Setup
# -----------------------------

def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Load Pretrained Models
# -----------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# -----------------------------
# Feature Engineering
# -----------------------------

def extract_features(df):

    features = []
    grouped = df.groupby("chat_id")

    for chat_id, group in grouped:

        group = group.sort_values("timestamp")
        total_messages = len(group)

        customer_msgs = len(group[group["speaker"] == "customer"])
        agent_msgs = len(group[group["speaker"] == "agent"])

        group["timestamp"] = pd.to_datetime(group["timestamp"])
        time_diffs = group["timestamp"].diff().dt.total_seconds().fillna(0)

        avg_response_time = time_diffs.mean()
        max_gap = time_diffs.max()

        escalation_keywords = ["refund", "cancel", "angry", "complaint"]
        escalation_count = sum(
            group["message"].str.lower().str.contains('|'.join(escalation_keywords))
        )

        full_text = " ".join(group["message"].astype(str).tolist())

        embedding = embedding_model.encode(full_text)

        emotion_output = emotion_classifier(full_text[:512])

        # Handle nested output safely
        if isinstance(emotion_output[0], list):
            emotions = emotion_output[0]
        else:
            emotions = emotion_output

        emotion_scores = {e["label"]: e["score"] for e in emotions}

        anger = emotion_scores.get("anger", 0)
        sadness = emotion_scores.get("sadness", 0)
        joy = emotion_scores.get("joy", 0)

        resolution_time = group["resolution_time_minutes"].iloc[0]
        label = 1 if resolution_time > 30 else 0

        combined = [
            total_messages,
            customer_msgs,
            agent_msgs,
            avg_response_time,
            max_gap,
            escalation_count,
            anger,
            sadness,
            joy,
        ] + list(embedding)

        features.append(combined + [label])

    base_cols = [
        "total_messages",
        "customer_msgs",
        "agent_msgs",
        "avg_response_time",
        "max_gap",
        "escalation_count",
        "anger",
        "sadness",
        "joy",
    ]

    embed_cols = [f"embed_{i}" for i in range(384)]
    columns = base_cols + embed_cols + ["label"]

    return pd.DataFrame(features, columns=columns)

# -----------------------------
# Train Model
# -----------------------------

def train_model(features_df):

    X = features_df.drop("label", axis=1)
    y = features_df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_scaled, y)

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(scaler, open(SCALER_PATH, "wb"))

# -----------------------------
# Authentication Routes
# -----------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
        except:
            return "User already exists"
        conn.close()

        return redirect("/login")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["user"] = username
            return redirect("/")
        else:
            return "Invalid Credentials"

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# -----------------------------
# Main Pages
# -----------------------------

@app.route("/")
def index():
    if "user" not in session:
        return redirect("/login")
    return render_template("index.html", user=session["user"])


@app.route("/train", methods=["POST"])
def train():
    file = request.files["file"]
    df = pd.read_csv(file)
    features_df = extract_features(df)
    train_model(features_df)
    return "Model Trained Successfully"


@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_PATH):
        return "Train model first"

    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))

    file = request.files["file"]
    df = pd.read_csv(file)
    features_df = extract_features(df)

    X = features_df.drop("label", axis=1)
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    return render_template("result.html",
                           prediction="Delayed" if predictions[0] == 1 else "On-Time",
                           risk=round(float(probabilities[0]), 3))


if __name__ == "__main__":
    app.run(debug=True, port=5050)