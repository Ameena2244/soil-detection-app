"""
preprocess.py
-------------
Image validation and preprocessing pipeline.

Steps
-----
1. Validate file extension (JPEG / PNG only)
2. Open with Pillow, convert to RGB
3. Resize to 256 × 256
4. Convert to tensor and normalise with ImageNet mean/std
5. Return (1, 3, 256, 256) batch tensor ready for SoilNet
"""

import io
import logging
from typing import Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

# Model expected input size
IMAGE_SIZE: Tuple[int, int] = (256, 256)

# ImageNet normalisation stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Accepted MIME types and extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}

# Maximum file size (16 MB)
MAX_FILE_SIZE_BYTES = 16 * 1024 * 1024

# ------------------------------------------------------------------ #
# Torchvision transform chain                                         #
# ------------------------------------------------------------------ #
_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def allowed_file(filename: str) -> bool:
    """Return True if *filename* has an accepted extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXTENSIONS


def validate_image_file(file) -> None:
    """
    Raise ValueError with a user-friendly message if the upload is invalid.

    Checks
    ------
    - Filename extension (jpg / jpeg / png)
    - File size ≤ 16 MB
    - File can actually be opened as an image (catches corrupt files)
    """
    filename = getattr(file, "filename", "")
    if not allowed_file(filename):
        raise ValueError(
            f"Unsupported file type '{filename}'. "
            "Only JPEG and PNG images are accepted."
        )

    # Check size without consuming the stream
    file.seek(0, 2)          # seek to end
    size = file.tell()
    file.seek(0)             # reset
    if size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File size {size / 1024 / 1024:.1f} MB exceeds the 16 MB limit."
        )

    # Try actually opening the image
    try:
        img = Image.open(file)
        img.verify()         # checks headers without decoding all pixels
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError(f"Cannot read image file: {exc}") from exc
    finally:
        file.seek(0)         # reset for later reading


def preprocess_image(file) -> torch.Tensor:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    file : werkzeug.datastructures.FileStorage or file-like
        Raw uploaded file object. Stream is reset before/after use.

    Returns
    -------
    torch.Tensor
        Shape (1, 3, 256, 256), float32, normalised.

    Raises
    ------
    ValueError
        If the file is invalid or cannot be processed.
    """
    validate_image_file(file)

    try:
        file.seek(0)
        image = Image.open(file).convert("RGB")
        logger.debug("Opened image size=%s mode=%s", image.size, image.mode)

        # Apply transform chain
        tensor = _transform(image)                   # (3, 256, 256)
        batch  = tensor.unsqueeze(0)                 # (1, 3, 256, 256)
        logger.debug("Preprocessed tensor shape: %s", batch.shape)
        return batch

    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during preprocessing")
        raise ValueError(f"Image preprocessing failed: {exc}") from exc


def pil_image_from_file(file) -> Image.Image:
    """
    Open and return a PIL Image (RGB) without the tensor transform.
    Useful for colour-histogram analysis in demo_predictor.
    """
    file.seek(0)
    img = Image.open(file).convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
    file.seek(0)
    return img


if __name__ == "__main__":
    # Quick smoke test with a dummy in-memory image
    import io
    logging.basicConfig(level=logging.DEBUG)

    buf = io.BytesIO()
    dummy_img = Image.new("RGB", (512, 512), color=(120, 80, 40))
    dummy_img.save(buf, format="JPEG")
    buf.seek(0)
    buf.filename = "test.jpg"

    tensor = preprocess_image(buf)
    print(f"Tensor shape : {tensor.shape}")
    print(f"Tensor dtype : {tensor.dtype}")
    print(f"Value range  : [{tensor.min():.3f}, {tensor.max():.3f}]")
