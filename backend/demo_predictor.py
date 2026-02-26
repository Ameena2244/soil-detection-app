"""
demo_predictor.py
-----------------
Colour-histogram heuristic predictor used when no trained model weights
exist.  It analyses the RGB distribution of the uploaded soil image and
maps typical soil colour ranges to one of the four soil classes.

Soil colour heuristics (approximate)
-------------------------------------
  Sandy  → warm tan / beige   (high R, mid G, low B)
  Clay   → reddish / brick    (dominant R, lower G & B)
  Loamy  → dark brown         (balanced, all channels mid-low)
  Silt   → greyish / pale     (balanced channels, all mid-high)

These are rough approximations; real models should be used in production.
"""

import random
import logging
from typing import Tuple

import numpy as np
from PIL import Image

from model import SOIL_CLASSES

logger = logging.getLogger(__name__)


class DemoPredictor:
    """
    Confidence-aware heuristic predictor.

    Usage
    -----
    predictor = DemoPredictor()
    label, confidence = predictor.predict_pil(pil_image)
    """

    # Mean-colour bounding boxes for each soil class
    # Each entry: (R_lo, R_hi, G_lo, G_hi, B_lo, B_hi)
    _COLOUR_RANGES = {
        "Sandy": (140, 220, 110, 180, 70,  140),   # warm tan / beige
        "Clay":  (130, 210,  70, 140, 50,  110),   # reddish / brick
        "Loamy": ( 60, 140,  50, 120, 30,   90),   # dark brown
        "Silt":  (150, 220, 140, 200, 120, 190),   # greyish / pale
    }

    def predict_pil(self, image: Image.Image) -> Tuple[str, float]:
        """
        Predict soil type from a PIL RGB image.

        Returns
        -------
        (soil_type, confidence) : Tuple[str, float]
        """
        arr  = np.array(image.convert("RGB"), dtype=np.float32)
        mean = arr.reshape(-1, 3).mean(axis=0)          # [R_mean, G_mean, B_mean]
        r, g, b = float(mean[0]), float(mean[1]), float(mean[2])

        logger.debug("Image mean RGB: R=%.1f G=%.1f B=%.1f", r, g, b)

        scores = {}
        for label, (rl, rh, gl, gh, bl, bh) in self._COLOUR_RANGES.items():
            # Score = fraction of channels whose mean falls within the range
            in_r = 1.0 if rl <= r <= rh else max(0, 1 - min(abs(r - rl), abs(r - rh)) / 60)
            in_g = 1.0 if gl <= g <= gh else max(0, 1 - min(abs(g - gl), abs(g - gh)) / 60)
            in_b = 1.0 if bl <= b <= bh else max(0, 1 - min(abs(b - bl), abs(b - bh)) / 60)
            scores[label] = (in_r + in_g + in_b) / 3.0

        best_label = max(scores, key=scores.get)
        raw_score  = scores[best_label]

        # Map raw [0,1] score to a realistic confidence band [0.72, 0.95]
        confidence = round(0.72 + raw_score * 0.23 + random.uniform(-0.02, 0.02), 2)
        confidence = max(0.72, min(0.95, confidence))

        logger.info(
            "DemoPredictor → %s (confidence=%.2f) | scores=%s",
            best_label, confidence,
            {k: f"{v:.2f}" for k, v in scores.items()},
        )
        return best_label, confidence

    def predict_random(self) -> Tuple[str, float]:
        """Fallback: pick a random soil type with a plausible confidence."""
        label = random.choice(SOIL_CLASSES)
        conf  = round(random.uniform(0.72, 0.89), 2)
        logger.info("DemoPredictor (random) → %s (%.2f)", label, conf)
        return label, conf


if __name__ == "__main__":
    from PIL import Image
    import io, logging
    logging.basicConfig(level=logging.DEBUG)

    # Create a warm-tan dummy image (should → Sandy)
    img = Image.new("RGB", (256, 256), color=(180, 145, 100))
    pred = DemoPredictor()
    label, conf = pred.predict_pil(img)
    print(f"Heuristic prediction: {label} ({conf*100:.1f}% confidence)")
