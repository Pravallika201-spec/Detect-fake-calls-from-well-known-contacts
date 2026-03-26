"""Offline speech-to-text wrapper using Whisper or Wav2Vec2."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from faster_whisper import WhisperModel

from backend.config.settings import MODEL_DIR


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    confidence: float


class OfflineSpeechToText:
    def __init__(self, model_size: str = "tiny", device: str = "cpu"):
        checkpoint_dir = MODEL_DIR / "whisper"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model = WhisperModel(model_size, device=device, download_root=str(checkpoint_dir))

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> List[TranscriptSegment]:
        segments, _ = self.model.transcribe(audio, language="en", beam_size=3)
        output: List[TranscriptSegment] = []
        for segment in segments:
            output.append(
                TranscriptSegment(
                    text=segment.text.strip(),
                    start=float(segment.start),
                    end=float(segment.end),
                    confidence=float(segment.avg_logprob),
                )
            )
        return output
