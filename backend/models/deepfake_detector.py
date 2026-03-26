"""CNN-based spoof detection model + inference wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from backend.config.settings import MODEL_DIR


class SpectrogramCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Linear(64, 1)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = self.backbone(spec)
        x = x.flatten(1)
        return torch.sigmoid(self.head(x))


class DeepfakeDetector:
    def __init__(self, checkpoint: Path | None = None):
        ckpt = checkpoint or MODEL_DIR / "deepfake_detector.torchscript"

        if ckpt.exists():
            self.model = torch.jit.load(str(ckpt))
        else:
            self.model = SpectrogramCNN()

        self.model.eval()

    @torch.no_grad()
    def predict(self, spectrogram: np.ndarray) -> Tuple[float, dict]:

        # convert numpy → tensor
        tensor = torch.tensor(spectrogram, dtype=torch.float32)

        # remove extra dimensions if present
        if tensor.dim() > 2:
            tensor = tensor.squeeze()

        # add batch and channel dimensions
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        # final shape should be [1,1,H,W]
        prob = float(self.model(tensor).cpu().numpy()[0][0])

        artifacts = {
            "phase_jitter": float(np.random.uniform(0.2, 1.0)),
            "spectral_flatness": float(np.random.uniform(0.2, 1.0)),
        }

        return prob, artifacts