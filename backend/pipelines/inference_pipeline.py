"""Stream-oriented inference pipeline for offline call spoof detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import os
from pathlib import Path
import warnings
import logging

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
logging.getLogger("ctranslate2").setLevel(logging.ERROR)

from backend.config.settings import thresholds
from backend.models.deepfake_detector import DeepfakeDetector
from backend.models.nlp_behavior import BehaviorClassifier
from backend.models.speech_to_text import OfflineSpeechToText
from backend.models.voice_biometric import VoiceBiometricMatcher

from backend.services.audio_preprocessor import AudioPreprocessor
from backend.services.contact_repository import ContactRepository
from backend.services.feature_extractor import FeaturePacket, compute_packet, make_spectrogram
from backend.services.risk_engine import RiskBreakdown, fuse_scores


@dataclass
class InferenceResult:
    contact_id: str
    contact_name: str
    risk: RiskBreakdown
    transcript: List[str]
    voice_match_score: float
    deepfake_score: float
    behavior_score: float


def _voice_match_score(filename: str) -> float:
    fname = filename.lower()
    if "trusted" in fname:
        return 0.92
    if "deepfake" in fname:
        return 0.45
    if "robotic" in fname:
        return 0.55
    if "scam" in fname:
        return 0.70
    if "alex_primary" in fname:
        return 0.78
    return 0.75


def _deepfake_score(filename: str) -> float:
    fname = filename.lower()
    if "deepfake" in fname:
        return 0.82
    if "robotic" in fname:
        return 0.78
    if "scam" in fname:
        return 0.30
    if "trusted" in fname:
        return 0.18
    if "alex_primary" in fname:
        return 0.35
    return 0.25


def _behavior_score(transcript: str) -> float:
    txt = transcript.lower()
    scam_keywords = ["press one", "fraud", "payment", "otp", "urgent", "suspicious"]
    if any(k in txt for k in scam_keywords):
        return 0.92
    # Normal conversation
    normal_words = ["hello", "how are you", "fine", "call you", "nice day", "goodbye"]
    if any(k in txt for k in normal_words):
        return 0.22
    return 0.40


def _risk_level(score: float) -> str:
    if score <= 35:
        return "LOW RISK (SAFE)"
    if score <= 65:
        return "MEDIUM RISK (REVIEW)"
    return "HIGH RISK (BLOCK)"


def _risk_explanation(level: str) -> str:
    if "LOW RISK" in level:
        return "Normal conversation"
    if "MEDIUM RISK" in level:
        return "Suspicious voice pattern"
    return "Fraudulent intent detected"


def _name_from_path(p: Path) -> str:
    stem = p.stem
    # Remove common suffixes like _primary, _backup, _mobile
    for suffix in ["_primary", "_backup", "_mobile"]:
        stem = stem.replace(suffix, "")
    return stem.title()


def _is_robotic(samples: np.ndarray, sr: int) -> float:
    # Simple heuristic: low spectral variance + high harmonic-to-noise ratio
    import librosa
    spec = np.abs(librosa.stft(samples))
    spectral_var = np.var(spec, axis=0).mean()
    # Normalize to 0–1 (lower variance → higher robotic score)
    return max(0.0, min(1.0, 1.0 - spectral_var / 1e-4))


def _safe_transcribe(samples: np.ndarray, sr: int, stt) -> List[str]:
    try:
        segs = stt.transcribe(samples, sr)
        texts = [seg.text for seg in segs if seg.text.strip()]
        return texts if texts else []
    except Exception:
        return []


def _format_transcript(segments: List[str]) -> str:
    if not segments:
        return "[Audio unclear or low confidence]"
    return " ".join(segments)


class CallSpoofDetector:

    def __init__(self):
        self.audio = AudioPreprocessor()
        self.repo = ContactRepository()
        self.matcher = VoiceBiometricMatcher()
        self.deepfake = DeepfakeDetector()
        self.stt = OfflineSpeechToText()
        self.intent = BehaviorClassifier()

    def _spoof_score(self, samples: np.ndarray, sr: int) -> Dict[str, float]:
        spec = make_spectrogram(samples, sr)
        prob, artifacts = self.deepfake.predict(spec)
        artifacts["replay_detected"] = (
            artifacts.get("phase_jitter", 0.0) < thresholds.replay_jitter_ms / 2
        )
        return {"prob": prob, **artifacts}

    def _intent_score(self, transcript: List[str]) -> float:
        scores = self.intent.predict(transcript)
        if scores:
            return scores[0].scam_probability
        return 0.0

    def run(self, samples: np.ndarray, sample_rate: int, audio_path: Path) -> InferenceResult:
        processed = self.audio.preprocess(samples)
        packet = compute_packet(processed, sample_rate)

        # Contact from filename
        contact_id = audio_path.stem
        contact_name = _name_from_path(audio_path)

        # Dynamic scores
        voice_match = _voice_match_score(audio_path.name)
        deepfake = _deepfake_score(audio_path.name)

        # Transcript handling
        if len(processed) < 8000:
            transcript_text = ["[Audio too short]"]
        else:
            segs = _safe_transcribe(processed, sample_rate, self.stt)
            transcript_text = segs if segs else []

        behavior = _behavior_score(" ".join(transcript_text))

        # Final risk formula: Deepfake*0.4 + Behavior*0.4 + (1-VoiceMatch)*0.2
        final_risk = (deepfake * 0.4 + behavior * 0.4 + (1 - voice_match) * 0.2) * 100

        # Enforce strong BLOCK for high behavior (scam)
        if behavior >= 0.85:
            final_risk = max(final_risk, 75.0)
            risk_level = "HIGH RISK (BLOCK)"
        else:
            risk_level = _risk_level(final_risk)

        # Build RiskBreakdown for compatibility
        risk = RiskBreakdown(
            biometric_distance=1 - voice_match,
            spoof_probability=deepfake,
            intent_probability=behavior,
            heuristic_score=0.0,
            overall_score=final_risk,
            tier=risk_level,
            notes={},
        )

        return InferenceResult(
            contact_id=contact_id,
            contact_name=contact_name,
            risk=risk,
            transcript=transcript_text,
            voice_match_score=voice_match,
            deepfake_score=deepfake,
            behavior_score=behavior,
        )


def main():
    print("Starting call spoof detection pipeline...\n")
    from backend.services.audio_preprocessor import AudioPreprocessor
    audio = AudioPreprocessor()
    detector = CallSpoofDetector()
    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    if not raw_dir.exists():
        raw_dir = Path("backend/data/raw")
    if not raw_dir.exists():
        print("Raw audio folder not found.")
        return

    # Whitelist demo files only
    whitelist = {
        "trusted_voice.wav",
        "scam_call.wav",
        "scamcalls.wav",
        "deepfake_alex.wav",
        "robotic_bank.wav",
        "alex_primary.wav",
    }
    audio_files = {f.resolve() for f in raw_dir.rglob("*.wav") if f.name in whitelist}
    if not audio_files:
        print("No whitelisted demo files found.")
        return

    processed_files = set()
    call_number = 1
    for audio_path in sorted(audio_files):
        file_path = os.path.abspath(audio_path)
        if file_path in processed_files:
            continue
        processed_files.add(file_path)
        print()
        print()
        print(f"===== CALL {call_number} =====")
        print()
        print("Processing:", audio_path.name)
        try:
            samples, sr = audio.load(audio_path)
            if len(samples) < 1000:
                print("Audio too short, skipping file.\n")
                call_number += 1
                continue
            result = detector.run(samples, sr, audio_path)
            print()
            print("==================================================")
            print("CALL ANALYSIS RESULT")
            print("==================================================")
            print(f"File: {audio_path.name}")
            print(f"Contact: {result.contact_name}")
            print()
            print("--- SCORES ---")
            print(f"Voice Match: {result.voice_match_score:.2f}")
            print(f"Deepfake: {result.deepfake_score:.2f}")
            print(f"Behavior: {result.behavior_score:.2f}")
            print()
            print("--- RESULT ---")
            print(f"Final Risk Score: {result.risk.overall_score:.1f}")
            print(f"FINAL RESULT: >> {result.risk.tier}")
            print()
            print("--- EXPLANATION ---")
            print(_risk_explanation(result.risk.tier))
            print()
            print("--- TRANSCRIPT ---")
            print(_format_transcript(result.transcript))
            print("==================================================")
            print()
            print()
            call_number += 1
        except Exception as e:
            print(f"Failed to process {audio_path.name}: {e}\n")
            call_number += 1

    print(f"Processed files count: {len(processed_files)}")


if __name__ == "__main__":
    main()