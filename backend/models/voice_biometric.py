"""Speaker embedding + matching utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from backend.config.settings import MODEL_DIR


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-9
    return vec / norm


class SiameseEncoder(nn.Module):
    """Lightweight 1D CNN encoder for MFCC sequences."""

    def __init__(self, embedding_dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(40, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(256, embedding_dim)

    def forward(self, mfcc_batch: torch.Tensor) -> torch.Tensor:  # (B, 40, T)
        x = self.net(mfcc_batch)
        x = x.squeeze(-1)
        x = self.proj(x)
        return nn.functional.normalize(x, p=2.0, dim=-1)


@dataclass
class MatchResult:
    contact_id: str
    cosine_distance: float
    match_confidence: float
    metadata: Dict[str, str]


class VoiceBiometricMatcher:
    def __init__(self, checkpoint: Path | None = None):
        ckpt = checkpoint or MODEL_DIR / "voice_embedder.torchscript"
        if ckpt.exists():
            self.encoder = torch.jit.load(str(ckpt))
        else:
            self.encoder = SiameseEncoder()
        self.encoder.eval()

    @torch.inference_mode()
    def embed(self, mfcc: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(mfcc).float().unsqueeze(0)
        if tensor.dim() == 2:  # (frames, coeffs)
            tensor = tensor.transpose(0, 1).unsqueeze(0)
        tensor = tensor[:, :, :tensor.shape[-1]]
        embedding = self.encoder(tensor).cpu().numpy()[0]
        return _l2_normalize(embedding)

    def match(self, probe: np.ndarray, gallery: List[Tuple[str, np.ndarray, Dict]]) -> MatchResult:
        probe = _l2_normalize(probe)
        best_id, best_dist, best_meta = "unknown", 1.0, {}
        for contact_id, template, metadata in gallery:
            distance = float(1 - np.dot(probe, _l2_normalize(template))) / 2
            if distance < best_dist:
                best_id, best_dist, best_meta = contact_id, distance, metadata
        return MatchResult(
            contact_id=best_id,
            cosine_distance=best_dist,
            match_confidence=max(0.0, 1.0 - best_dist),
            metadata=best_meta,
        )
