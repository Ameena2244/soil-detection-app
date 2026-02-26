"""
model.py
--------
SoilNet – A lightweight 3-block CNN for soil type classification.

Input : RGB image tensor of shape (batch, 3, 256, 256)
Output: Logits over 4 soil classes (Sandy, Clay, Loamy, Silt)

Architecture
------------
Block 1:  Conv2d(3→32,   3×3) → BN → ReLU → MaxPool(2)  → 128×128
Block 2:  Conv2d(32→64,  3×3) → BN → ReLU → MaxPool(2)  → 64×64
Block 3:  Conv2d(64→128, 3×3) → BN → ReLU → MaxPool(2)  → 32×32
GlobalAvgPool                                              → 128-dim
FC1: 128 → 256, ReLU, Dropout(0.5)
FC2: 256 → 4   (logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ordered class labels – index 0 → Sandy, 1 → Clay, etc.
SOIL_CLASSES = ["Sandy", "Clay", "Loamy", "Silt"]
NUM_CLASSES  = len(SOIL_CLASSES)


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → MaxPool2d."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class SoilNet(nn.Module):
    """
    Lightweight CNN for soil image classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 4).
    dropout : float
        Dropout probability before the final FC layer.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        # Feature extractor
        self.block1 = ConvBlock(3,   32)   # 256 → 128
        self.block2 = ConvBlock(32,  64)   # 128 → 64
        self.block3 = ConvBlock(64,  128)  # 64  → 32

        # Global average pooling collapses spatial dims → (batch, 128)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x)

    # ---------------------------------------------------------------- #
    # Helper – run inference on a single preprocessed tensor           #
    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def predict(self, tensor: torch.Tensor):
        """
        Run forward pass and return (class_label, confidence).

        Parameters
        ----------
        tensor : torch.Tensor
            Shape (1, 3, 256, 256), already normalised.

        Returns
        -------
        tuple[str, float]
            Predicted soil type label and confidence score in [0, 1].
        """
        self.eval()
        logits      = self(tensor)
        probs       = F.softmax(logits, dim=1)
        conf, idx   = probs.max(dim=1)
        label       = SOIL_CLASSES[idx.item()]
        confidence  = round(conf.item(), 4)
        return label, confidence


def load_model(model_path: str, device: str = "cpu") -> SoilNet:
    """
    Load SoilNet weights from *model_path*.

    Returns an eval-mode model ready for inference.
    Raises FileNotFoundError if the weights file does not exist.
    """
    if not __import__("os").path.isfile(model_path):
        raise FileNotFoundError(
            f"Model weights not found at '{model_path}'. "
            "Please train the model first using train.py."
        )
    model = SoilNet(num_classes=NUM_CLASSES)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick sanity check
    net   = SoilNet()
    dummy = torch.randn(2, 3, 256, 256)
    out   = net(dummy)
    print(f"SoilNet output shape: {out.shape}")   # Expected: torch.Size([2, 4])
    print(f"Classes: {SOIL_CLASSES}")
    total = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total:,}")
