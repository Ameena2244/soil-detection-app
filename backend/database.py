"""
database.py
-----------
SQLite database initialisation and soil-crop lookup.
Creates soil_crops.db on first run, seeds it with soil types and
recommended crops, and provides a query function used by the API.
"""

import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

# Database file lives in the same directory as this script
DB_PATH = os.path.join(os.path.dirname(__file__), "soil_crops.db")

# ------------------------------------------------------------------ #
# Seed data – 4 soil types × 5 crops each + nutrient profiles        #
# ------------------------------------------------------------------ #
SOIL_DATA = {
    "Sandy": {
        "description": "Sandy soil has large particles, drains quickly, and is low in nutrients.",
        "nutrients": {"nitrogen": "Low", "phosphorus": "Low", "potassium": "Medium"},
        "crops": [
            ("Groundnut",   "Groundnut thrives in sandy soil with good drainage."),
            ("Watermelon",  "Watermelons love the warmth and drainage of sandy soil."),
            ("Carrot",      "Carrots grow straight and long in loose sandy soil."),
            ("Potato",      "Potatoes prefer loose soil for tuber expansion."),
            ("Barley",      "Barley is drought-tolerant and suits sandy conditions."),
        ],
    },
    "Clay": {
        "description": "Clay soil has fine particles, retains water well, and is nutrient-rich.",
        "nutrients": {"nitrogen": "High", "phosphorus": "High", "potassium": "Medium"},
        "crops": [
            ("Rice",       "Rice thrives in water-retaining clay soil."),
            ("Wheat",      "Wheat benefits from the nutrient richness of clay."),
            ("Sugarcane",  "Sugarcane needs water-retentive clay for best yield."),
            ("Broccoli",   "Broccoli loves the moisture and nutrients in clay."),
            ("Cabbage",    "Cabbage prefers moist, nutrient-dense clay soil."),
        ],
    },
    "Loamy": {
        "description": "Loamy soil is a balanced mix of sand, silt, and clay – ideal for most crops.",
        "nutrients": {"nitrogen": "High", "phosphorus": "Medium", "potassium": "High"},
        "crops": [
            ("Tomato",   "Tomatoes thrive in well-drained, nutrient-rich loamy soil."),
            ("Corn",     "Corn grows vigorously in fertile loamy soil."),
            ("Spinach",  "Spinach benefits from the balanced moisture of loamy soil."),
            ("Soybean",  "Soybeans fix nitrogen well in loamy soil."),
            ("Sunflower","Sunflowers flourish in deep, well-drained loamy soil."),
        ],
    },
    "Silt": {
        "description": "Silt soil has medium-sized particles, retains moisture, and is moderately fertile.",
        "nutrients": {"nitrogen": "Medium", "phosphorus": "Medium", "potassium": "Low"},
        "crops": [
            ("Jute",      "Jute grows best in moist, silty river-valley soil."),
            ("Maize",     "Maize yields well in silty, moisture-retaining soil."),
            ("Cucumber",  "Cucumbers love the fine texture of silty soil."),
            ("Pepper",    "Pepper prefers fertile, moisture-retentive silt."),
            ("Mustard",   "Mustard adapts well to silty soil conditions."),
        ],
    },
}


def init_db():
    """Create tables and seed data if they don't already exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Create tables
    c.executescript("""
        CREATE TABLE IF NOT EXISTS soil_types (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT UNIQUE NOT NULL,
            description TEXT,
            nitrogen    TEXT,
            phosphorus  TEXT,
            potassium   TEXT
        );

        CREATE TABLE IF NOT EXISTS crop_recommendations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            soil_id     INTEGER NOT NULL,
            crop_name   TEXT NOT NULL,
            description TEXT,
            FOREIGN KEY (soil_id) REFERENCES soil_types(id)
        );
    """)

    # Seed only if empty
    c.execute("SELECT COUNT(*) FROM soil_types")
    if c.fetchone()[0] == 0:
        for soil_name, data in SOIL_DATA.items():
            n = data["nutrients"]
            c.execute(
                "INSERT INTO soil_types (name, description, nitrogen, phosphorus, potassium) VALUES (?,?,?,?,?)",
                (soil_name, data["description"], n["nitrogen"], n["phosphorus"], n["potassium"]),
            )
            soil_id = c.lastrowid
            for crop_name, crop_desc in data["crops"]:
                c.execute(
                    "INSERT INTO crop_recommendations (soil_id, crop_name, description) VALUES (?,?,?)",
                    (soil_id, crop_name, crop_desc),
                )
        logger.info("Database seeded with %d soil types.", len(SOIL_DATA))

    conn.commit()
    conn.close()
    logger.info("Database initialised at %s", DB_PATH)


def get_crops_for_soil(soil_type: str) -> dict:
    """
    Return crops and nutrient data for the given soil type.

    Returns a dict:
    {
        "crops": [{"name": ..., "description": ...}, ...],
        "nutrients": {"nitrogen": ..., "phosphorus": ..., "potassium": ...},
        "description": ...,
    }
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute(
        "SELECT * FROM soil_types WHERE LOWER(name) = LOWER(?)", (soil_type,)
    )
    soil_row = c.fetchone()

    if soil_row is None:
        conn.close()
        logger.warning("Soil type '%s' not found in database.", soil_type)
        return {"crops": [], "nutrients": {}, "description": "Unknown soil type."}

    c.execute(
        "SELECT crop_name, description FROM crop_recommendations WHERE soil_id = ? LIMIT 5",
        (soil_row["id"],),
    )
    crop_rows = c.fetchall()
    conn.close()

    return {
        "crops": [{"name": r["crop_name"], "description": r["description"]} for r in crop_rows],
        "nutrients": {
            "nitrogen":    soil_row["nitrogen"],
            "phosphorus":  soil_row["phosphorus"],
            "potassium":   soil_row["potassium"],
        },
        "description": soil_row["description"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("Sandy soil crops:", get_crops_for_soil("Sandy")["crops"])
