"""
app.py
------
SoilSense AI – Flask REST API

Endpoints
---------
  POST /predict   – Accept soil image, return prediction JSON
  GET  /health    – Health check
  GET  /soils     – List all supported soil types and their crops

Run
---
  python app.py          # development  (debug mode, port 5000)
  gunicorn app:app       # production

Environment Variables
---------------------
  MODEL_PATH   Path to trained weights (default: models/soil_model.pth)
  PORT         Server port                (default: 5000)
  FLASK_ENV    'development' or 'production'
"""

import logging
import os
import time
import traceback
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

from database import init_db, get_crops_for_soil
from preprocess import preprocess_image, validate_image_file, pil_image_from_file

# ------------------------------------------------------------------ #
# Logging                                                             #
# ------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("soilsense")


# ------------------------------------------------------------------ #
# Flask App                                                           #
# ------------------------------------------------------------------ #
app = Flask(__name__)
CORS(app, origins="*")           # Allow all origins for local dev

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB


# ------------------------------------------------------------------ #
# Model Loading (lazy – done once at startup)                         #
# ------------------------------------------------------------------ #
BASE_DIR   = Path(__file__).parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR / "models" / "soil_model.pth"))

_model          = None     # SoilNet instance (if weights found)
_demo_predictor = None     # DemoPredictor instance (fallback)
_using_demo     = False    # Flag shown in responses


def _load_model():
    """Attempt to load the trained SoilNet model; fall back to DemoPredictor."""
    global _model, _demo_predictor, _using_demo

    if os.path.isfile(MODEL_PATH):
        try:
            from model import load_model
            _model     = load_model(MODEL_PATH)
            _using_demo = False
            logger.info("✔ Loaded trained model from %s", MODEL_PATH)
        except Exception as exc:
            logger.warning("Failed to load model: %s – using demo predictor.", exc)
            _using_demo = True
    else:
        logger.warning(
            "Model weights not found at '%s'. Using demo (heuristic) predictor. "
            "Run train.py to train the model.",
            MODEL_PATH,
        )
        _using_demo = True

    if _using_demo:
        from demo_predictor import DemoPredictor
        _demo_predictor = DemoPredictor()


# ------------------------------------------------------------------ #
# Prediction helper                                                   #
# ------------------------------------------------------------------ #
def _predict(file) -> dict:
    """
    Core prediction pipeline.

    1. Validate image file
    2. Preprocess image
    3. Run model or demo predictor
    4. Fetch crops from database
    5. Return structured result dict
    """
    if _using_demo:
        # Heuristic predictor only needs PIL image
        pil_img    = pil_image_from_file(file)
        soil_type, confidence = _demo_predictor.predict_pil(pil_img)
    else:
        # Full CNN inference
        tensor = preprocess_image(file)
        soil_type, confidence = _model.predict(tensor)

    # Fetch crop & nutrient data from SQLite
    db_data = get_crops_for_soil(soil_type)

    result = {
        "soil_type":           soil_type,
        "soil_description":    db_data.get("description", ""),
        "recommended_plants":  [c["name"] for c in db_data.get("crops", [])[:3]],
        "all_crops":           db_data.get("crops", []),
        "confidence":          confidence,
        "confidence_pct":      f"{confidence * 100:.1f}%",
        "soil_nutrients":      db_data.get("nutrients", {}),
        "demo_mode":           _using_demo,
    }
    return result


# ------------------------------------------------------------------ #
# Routes                                                              #
# ------------------------------------------------------------------ #
@app.route("/health", methods=["GET"])
def health():
    """Health check – confirms API is running."""
    return jsonify({
        "status":     "ok",
        "model_mode": "demo" if _using_demo else "trained",
        "version":    "1.0.0",
    })


@app.route("/soils", methods=["GET"])
def list_soils():
    """Return all supported soil types with their crop and nutrient data."""
    from database import SOIL_DATA
    soils = []
    for name, data in SOIL_DATA.items():
        soils.append({
            "name":        name,
            "description": data["description"],
            "nutrients":   data["nutrients"],
            "crops":       [c[0] for c in data["crops"]],
        })
    return jsonify({"soils": soils, "count": len(soils)})


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    -------------
    Accepts: multipart/form-data  field name: 'image'
    Returns: JSON prediction result

    Success (200):
    {
        "success": true,
        "soil_type": "Loamy",
        "soil_description": "...",
        "recommended_plants": ["Tomato", "Corn", "Spinach"],
        "all_crops": [{"name": ..., "description": ...}, ...],
        "confidence": 0.92,
        "confidence_pct": "92.0%",
        "soil_nutrients": {"nitrogen": "High", "phosphorus": "Medium", "potassium": "High"},
        "demo_mode": false,
        "processing_time_ms": 45
    }

    Error (400 / 500):
    { "success": false, "error": "..." }
    """
    t_start = time.perf_counter()

    # ── 1. Validate form field ─────────────────────────────────────
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error":   "No 'image' field in the request. Send file as multipart/form-data."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error":   "No file selected. Please choose an image to upload."
        }), 400

    # ── 2. Validate image format / size ───────────────────────────
    try:
        validate_image_file(file)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    # ── 3. Run prediction pipeline ────────────────────────────────
    try:
        result = _predict(file)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception:
        logger.exception("Unexpected error during prediction")
        return jsonify({
            "success": False,
            "error":   "An internal error occurred. Please try again."
        }), 500

    # ── 4. Build response ─────────────────────────────────────────
    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
    response   = {
        "success":            True,
        "processing_time_ms": elapsed_ms,
        **result,
    }

    logger.info(
        "Predicted: soil_type=%s confidence=%.2f mode=%s time=%.1fms",
        result["soil_type"], result["confidence"],
        "demo" if _using_demo else "model", elapsed_ms,
    )
    return jsonify(response), 200


# ------------------------------------------------------------------ #
# Error handlers                                                      #
# ------------------------------------------------------------------ #
@app.errorhandler(413)
def request_entity_too_large(_):
    return jsonify({
        "success": False,
        "error":   "File too large. Maximum allowed size is 16 MB."
    }), 413


@app.errorhandler(404)
def not_found(_):
    return jsonify({
        "success": False,
        "error":   "Endpoint not found. Available: POST /predict, GET /health, GET /soils"
    }), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({
        "success": False,
        "error":   "Method not allowed."
    }), 405


# ------------------------------------------------------------------ #
# Startup                                                             #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("=" * 60)
    logger.info("  SoilSense AI – Backend API")
    logger.info("  http://localhost:%d", port)
    logger.info("=" * 60)

    # Initialise database
    init_db()

    # Load ML model
    _load_model()

    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_ENV") == "development")
