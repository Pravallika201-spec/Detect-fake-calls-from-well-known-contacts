"""Offline audio preprocessing: denoise, normalize, VAD trimming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


@dataclass
class AudioFrame:
    samples: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float


class AudioPreprocessor:

    def __init__(self, target_sr: int = 16000, frame_dur_ms: int = 800):
        self.target_sr = target_sr
        self.frame_len = int(target_sr * frame_dur_ms / 1000)

    def load(self, path: str) -> Tuple[np.ndarray, int]:

        samples, sr = sf.read(path)

        # convert stereo → mono
        if len(samples.shape) > 1:
            samples = np.mean(samples, axis=1)

        if sr != self.target_sr:
            samples = librosa.resample(samples, orig_sr=sr, target_sr=self.target_sr)

        return samples.astype(np.float32), self.target_sr

    def normalize(self, samples: np.ndarray) -> np.ndarray:

        if len(samples) == 0:
            return samples

        peak = np.abs(samples).max() + 1e-9
        return samples / peak

    def denoise(self, samples: np.ndarray) -> np.ndarray:

        # prevent crashes for extremely short audio
        if len(samples) < 512:
            return samples

        try:

            spec = librosa.stft(samples, n_fft=512)
            magnitude = np.abs(spec)

            spectral = librosa.decompose.nn_filter(
                magnitude,
                aggregate=np.median,
                metric="cosine",
            )

            mask = librosa.util.softmask(spectral, magnitude)

            clean = librosa.istft(mask * spec)

            return librosa.util.fix_length(clean, size=len(samples))

        except Exception:
            # fallback if librosa fails
            return samples

    def voice_activity_trim(self, samples: np.ndarray, top_db: int = 25) -> np.ndarray:

        if len(samples) < 512:
            return samples

        try:
            trimmed, _ = librosa.effects.trim(samples, top_db=top_db)
            return trimmed
        except Exception:
            return samples

    def preprocess(self, samples: np.ndarray) -> np.ndarray:

        samples = np.array(samples)

        # prevent extremely short audio crashes
        if len(samples) < 512:
            return samples

        samples = self.normalize(samples)
        samples = self.denoise(samples)
        samples = self.voice_activity_trim(samples)

        return samples

    def preprocess_stream(self, samples: np.ndarray) -> Tuple[np.ndarray, bool]:

        clean = self.preprocess(samples)

        if len(clean) == 0:
            return np.zeros((1, self.frame_len)), False

        pad = self.frame_len - (len(clean) % self.frame_len)

        if pad != self.frame_len:
            clean = np.pad(clean, (0, pad))

        reshaped = clean.reshape(-1, self.frame_len)

        return reshaped, pad != self.frame_len