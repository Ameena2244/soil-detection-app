# SoilSense AI — Soil Detection & Crop Recommendation

> Upload a soil image → CNN detects soil type → SQLite returns crop recommendations

---

## Project Structure

```
soil_detection_app/
├── backend/
│   ├── app.py              # Flask REST API (POST /predict)
│   ├── model.py            # PyTorch SoilNet CNN
│   ├── preprocess.py       # Image validation & normalisation
│   ├── database.py         # SQLite init + crop lookup
│   ├── demo_predictor.py   # Heuristic fallback (no model needed)
│   ├── train.py            # Training script (bring your own data)
│   ├── requirements.txt
│   └── models/             # Drop soil_model.pth here after training
└── frontend/
    ├── index.html
    ├── css/style.css
    └── js/app.js
```

---

## Quick Start

### 1 — Backend

```bash
cd "soil_detection_app/backend"

# Create and activate virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
python app.py
# → API running at http://localhost:5000
```

### 2 — Frontend

Open `frontend/index.html` directly in your browser.

> No build step needed — plain HTML/CSS/JS.

---

## API Reference

### `POST /predict`

| Field | Type | Description |
|-------|------|-------------|
| `image` | `File` | Soil photo (JPEG or PNG, ≤ 16 MB) |

**Success Response (200)**
```json
{
  "success": true,
  "soil_type": "Loamy",
  "soil_description": "Loamy soil is a balanced mix…",
  "recommended_plants": ["Tomato", "Corn", "Spinach"],
  "all_crops": [{"name": "Tomato", "description": "…"}, …],
  "confidence": 0.92,
  "confidence_pct": "92.0%",
  "soil_nutrients": {
    "nitrogen": "High",
    "phosphorus": "Medium",
    "potassium": "High"
  },
  "demo_mode": false,
  "processing_time_ms": 42.3
}
```

**Error Response (400 / 500)**
```json
{ "success": false, "error": "Unsupported file type…" }
```

### `GET /health`
```json
{ "status": "ok", "model_mode": "demo", "version": "1.0.0" }
```

### `GET /soils`
Returns all 4 soil types with crops and nutrients.

---

## Training Your Own Model

```
1. Collect soil images and organise them:
   backend/data/train/Sandy/   *.jpg
   backend/data/train/Clay/    *.jpg
   backend/data/train/Loamy/   *.jpg
   backend/data/train/Silt/    *.jpg
   backend/data/val/…          (same structure, 20% of data)

2. Run training:
   python train.py --epochs 30 --batch_size 32

3. Model saved to: backend/models/soil_model.pth
4. Restart the API — it will load the trained model automatically.
```

---

## Supported Soil Types & Crops

| Soil Type | Top Crops | Nutrients |
|-----------|-----------|-----------|
| **Sandy** | Groundnut, Watermelon, Carrot, Potato, Barley | N:Low P:Low K:Med |
| **Clay**  | Rice, Wheat, Sugarcane, Broccoli, Cabbage | N:High P:High K:Med |
| **Loamy** | Tomato, Corn, Spinach, Soybean, Sunflower | N:High P:Med K:High |
| **Silt**  | Jute, Maize, Cucumber, Pepper, Mustard | N:Med P:Med K:Low |

---

## Demo Mode

If `models/soil_model.pth` is not found, the API automatically falls back to a **colour-histogram heuristic** predictor. Results are still realistic but less accurate. A yellow banner in the UI indicates demo mode.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5 · CSS3 · Vanilla JS |
| Backend  | Python 3.10+ · Flask 3.0 · Flask-CORS |
| ML Model | PyTorch 2.2 · SoilNet CNN |
| Database | SQLite 3 |
| Images   | Pillow · OpenCV (headless) |

---

## Validation Rules

- ✅ JPEG and PNG accepted
- ✅ Maximum 16 MB
- ✅ Corrupt / empty files rejected with clear error message
- ✅ Non-image files detected by MIME type
- ✅ 30-second client timeout with helpful error message

---

## License

MIT — VAC Project 2026
