"""Feature extraction routines shared across biometric and spoof modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import librosa
import numpy as np


@dataclass
class FeaturePacket:
    mfcc: np.ndarray
    spectral_contrast: np.ndarray
    chroma: np.ndarray
    delta: np.ndarray
    meta: Dict[str, float]


def compute_mfcc(samples: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    return librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=n_mfcc)


def compute_packet(samples: np.ndarray, sr: int) -> FeaturePacket:
    mfcc = compute_mfcc(samples, sr)
    spectral = librosa.feature.spectral_contrast(y=samples, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=samples, sr=sr)
    delta = librosa.feature.delta(mfcc)
    meta = {
        "duration": len(samples) / sr,
        "energy": float(np.mean(samples ** 2)),
        "zcr": float(np.mean(librosa.feature.zero_crossing_rate(samples))),
    }
    return FeaturePacket(mfcc=mfcc, spectral_contrast=spectral, chroma=chroma, delta=delta, meta=meta)


def make_spectrogram(samples: np.ndarray, sr: int) -> np.ndarray:
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(samples)), ref=np.max)
    return np.expand_dims(spec, axis=0)
