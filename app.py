"""
Visitor Entry System - Flask + DeepFace
Run: python app.py
"""

import os
import json
import base64
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from deepface import DeepFace

app = Flask(__name__)

# ─── Config ────────────────────────────────────────────────────────────────
FACES_DIR = "visitor_faces"          # Stores face images
DB_FILE   = "visitors.json"          # Stores visitor records
THRESHOLD = 0.6                      # Face match threshold (lower = stricter)

os.makedirs(FACES_DIR, exist_ok=True)

# ─── Database Helpers ───────────────────────────────────────────────────────
def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ─── Face Helpers ───────────────────────────────────────────────────────────
def decode_image(base64_str):
    """Convert base64 image from browser to OpenCV format."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_bytes = base64.b64decode(base64_str)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def get_embedding(img):
    """Extract face embedding from image using DeepFace."""
    result = DeepFace.represent(
        img_path=img,
        model_name="Facenet",
        enforce_detection=True,
        detector_backend="opencv"
    )
    return result[0]["embedding"]

def cosine_similarity(a, b):
    """Compute cosine similarity between two face embeddings."""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_match(new_embedding, db):
    """Search all stored visitors for a face match. Returns (visitor_id, score) or None."""
    best_id    = None
    best_score = -1

    for visitor_id, visitor in db.items():
        if "embedding" not in visitor:
            continue
        score = cosine_similarity(new_embedding, visitor["embedding"])
        if score > best_score:
            best_score = score
            best_id    = visitor_id

    if best_score >= THRESHOLD:
        return best_id, round(float(best_score), 3)
    return None, None

# ─── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scan", methods=["POST"])
def scan():
    """
    Receives a webcam snapshot, checks if face is known.
    Returns: { status: 'known'|'unknown', visitor: {...} } 
    """
    data     = request.json
    img      = decode_image(data["image"])
    db       = load_db()

    try:
        embedding = get_embedding(img)
    except Exception as e:
        return jsonify({"status": "error", "message": "No face detected. Please look at the camera."}), 400

    visitor_id, score = find_match(embedding, db)

    if visitor_id:
        visitor = db[visitor_id]
        # Log this visit
        visitor["visits"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        save_db(db)
        return jsonify({
            "status":     "known",
            "visitor_id": visitor_id,
            "name":       visitor["name"],
            "mobile":     visitor["mobile"],
            "purpose":    visitor["purpose"],
            "host":       visitor["host"],
            "total_visits": len(visitor["visits"]),
            "last_visit": visitor["visits"][-2] if len(visitor["visits"]) > 1 else "First time",
            "score":      score
        })
    else:
        # Unknown - return the embedding temporarily so frontend can use it for registration
        return jsonify({
            "status":    "unknown",
            "embedding": embedding  # Sent back to avoid re-processing on register
        })


@app.route("/register", methods=["POST"])
def register():
    """
    Registers a new visitor with their face embedding + details.
    """
    data      = request.json
    db        = load_db()
    visitor_id = str(uuid.uuid4())[:8]

    # Save face image to disk (for reference)
    img = decode_image(data["image"])
    img_path = os.path.join(FACES_DIR, f"{visitor_id}.jpg")
    cv2.imwrite(img_path, img)

    db[visitor_id] = {
        "name":      data["name"],
        "mobile":    data["mobile"],
        "purpose":   data["purpose"],
        "host":      data["host"],
        "embedding": data["embedding"],   # Already computed during /scan
        "visits":    [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    }
    save_db(db)

    return jsonify({
        "status":     "registered",
        "visitor_id": visitor_id,
        "name":       data["name"]
    })


@app.route("/visitors", methods=["GET"])
def get_visitors():
    """Returns all registered visitors (without embeddings for speed)."""
    db = load_db()
    result = []
    for vid, v in db.items():
        result.append({
            "id":           vid,
            "name":         v["name"],
            "mobile":       v["mobile"],
            "purpose":      v["purpose"],
            "host":         v["host"],
            "total_visits": len(v["visits"]),
            "last_visit":   v["visits"][-1] if v["visits"] else "—"
        })
    return jsonify(sorted(result, key=lambda x: x["last_visit"], reverse=True))


@app.route("/delete/<visitor_id>", methods=["DELETE"])
def delete_visitor(visitor_id):
    """Remove a visitor from the database."""
    db = load_db()
    if visitor_id not in db:
        return jsonify({"error": "Not found"}), 404
    del db[visitor_id]
    save_db(db)
    # Remove face image too
    img_path = os.path.join(FACES_DIR, f"{visitor_id}.jpg")
    if os.path.exists(img_path):
        os.remove(img_path)
    return jsonify({"status": "deleted"})


# ─── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Visitor Entry System running at http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
