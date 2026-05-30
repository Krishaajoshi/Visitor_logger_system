# Visitor Entry System
Face recognition based visitor management using Flask + DeepFace.

## Project Structure
```
visitor_system/
├── app.py              ← Flask backend (all logic here)
├── requirements.txt    ← Python dependencies
├── visitors.json       ← Auto-created: visitor database
├── visitor_faces/      ← Auto-created: stored face images
└── templates/
    └── index.html      ← Frontend UI (camera + forms)
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
> First run takes a few minutes — DeepFace downloads the Facenet model automatically.

### 2. Run the server
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```
> Use Chrome or Firefox. Allow camera permission when prompted.

## How It Works

| Scenario | What Happens |
|----------|-------------|
| New visitor | Face scanned → form appears → fill name, mobile, purpose → registered in DB |
| Returning visitor | Face scanned → auto-matched → entry granted, no form needed |
| Can't scan | Use "Manual Entry" fallback |

## API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Main UI |
| `/scan` | POST | Detect & identify face |
| `/register` | POST | Register new visitor |
| `/visitors` | GET | List all visitors |
| `/delete/<id>` | DELETE | Remove a visitor |

## Notes
- Face data stored in `visitors.json` as embeddings (not raw images)
- Match threshold: **0.6** (adjust `THRESHOLD` in `app.py` if needed)
  - Higher → stricter matching
  - Lower → more lenient matching
- Works best with good lighting and front-facing camera
