"""
train.py
--------
Training script for SoilNet – a CNN for soil type classification.

Dataset layout expected on disk
--------------------------------
data/
  train/
    Sandy/    *.jpg *.png …
    Clay/     …
    Loamy/    …
    Silt/     …
  val/
    Sandy/    …
    Clay/     …
    Loamy/    …
    Silt/     …

Quick-start
-----------
1. Place images in the folder structure above.
2. Run:
       python train.py --epochs 30 --batch_size 32
3. Best weights are saved to  models/soil_model.pth

The script prints per-epoch accuracy and macro F1, and keeps the
checkpoint with the best validation accuracy.
"""

import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

from model import SoilNet, SOIL_CLASSES, NUM_CLASSES

# ------------------------------------------------------------------ #
# Logging                                                             #
# ------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Paths                                                               #
# ------------------------------------------------------------------ #
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "soil_model.pth"


# ------------------------------------------------------------------ #
# Data Augmentation                                                   #
# ------------------------------------------------------------------ #
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((270, 270)),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #
def evaluate(model, loader, device):
    """Run model on loader, return (accuracy, macro_f1)."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


def train(args):
    # Validate data directories
    for split in ("train", "val"):
        d = DATA_DIR / split
        if not d.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {d}\n"
                "Please create  data/train/<class>/  and  data/val/<class>/  "
                "directories and populate them with soil images."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load datasets
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=TRAIN_TRANSFORM)
    val_ds   = datasets.ImageFolder(DATA_DIR / "val",   transform=VAL_TRANSFORM)

    logger.info("Train: %d images | Val: %d images", len(train_ds), len(val_ds))
    logger.info("Classes: %s", train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # Model, loss, optimiser, scheduler
    model     = SoilNet(num_classes=NUM_CLASSES, dropout=0.5).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training for %d epochs …", args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Validation
        train_acc, train_f1 = evaluate(model, train_loader, device)
        val_acc,   val_f1   = evaluate(model, val_loader,   device)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %3d/%d | Loss: %.4f | Train Acc: %.2f%% F1: %.3f | "
            "Val Acc: %.2f%% F1: %.3f | LR: %.6f | %.1fs",
            epoch, args.epochs,
            running_loss / len(train_loader),
            train_acc * 100, train_f1,
            val_acc   * 100, val_f1,
            scheduler.get_last_lr()[0],
            elapsed,
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info("  ✔ New best model saved (val_acc=%.2f%%)", val_acc * 100)

    logger.info("Training complete. Best Val Accuracy: %.2f%%", best_val_acc * 100)
    logger.info("Model saved to: %s", MODEL_PATH)


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SoilNet CNN")
    parser.add_argument("--epochs",     type=int,   default=30,   help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=32,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--workers",    type=int,   default=2,    help="DataLoader worker threads")
    args = parser.parse_args()
    train(args)
