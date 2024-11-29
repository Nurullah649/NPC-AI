from flask import Flask, request, jsonify
import os
import json

app = Flask(__name__)

# Örnek veri depoları
USERS = {"test_user": "gUzm1vDdUsFx"}
SESSIONS = {}
FRAMES = [{"video_name": "example_video", "frame_id": 1, "data": "frame_data"}]
TRANSLATIONS = [{"video_name": "example_video", "frame_id": 1, "translation": "translation_data"}]
PREDICTIONS = []

# Kullanıcı kimlik doğrulama (token örnek bir string döndürür)
@app.route("/auth/", methods=["POST"])
def auth():
    data = request.form
    username = data.get("username")
    password = data.get("password")
    if username in USERS and USERS[username] == password:
        token = f"token_{username}"
        SESSIONS[token] = username
        return jsonify({"token": token}), 200
    return jsonify({"detail": "Invalid credentials"}), 401

# Frame'lerin gönderilmesi
@app.route("/frames/", methods=["GET"])
def get_frames():
    token = request.headers.get("Authorization", "").replace("Token ", "")
    if token in SESSIONS:
        return jsonify(FRAMES), 200
    return jsonify({"detail": "Unauthorized"}), 403

# Translation'ların gönderilmesi
@app.route("/translation/", methods=["GET"])
def get_translation():
    token = request.headers.get("Authorization", "").replace("Token ", "")
    if token in SESSIONS:
        return jsonify(TRANSLATIONS), 200
    return jsonify({"detail": "Unauthorized"}), 403

# Prediction gönderimi
@app.route("/prediction/", methods=["POST"])
def send_prediction():
    token = request.headers.get("Authorization", "").replace("Token ", "")
    if token in SESSIONS:
        data = request.get_json()
        PREDICTIONS.append(data)
        return jsonify({"detail": "Prediction received"}), 201
    return jsonify({"detail": "Unauthorized"}), 403

# Oturum bilgisi kontrolü
@app.route("/session/", methods=["GET"])
def get_session():
    token = request.headers.get("Authorization", "").replace("Token ", "")
    if token in SESSIONS:
        return jsonify({"session_name": f"session_{SESSIONS[token]}"}), 200
    return jsonify({"detail": "Unauthorized"}), 403

# Giriş sayfası (opsiyonel)
@app.route("/", methods=["GET"])
def index():
    return "Server is running.", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
