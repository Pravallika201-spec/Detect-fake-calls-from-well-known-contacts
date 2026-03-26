"""Fuse biometric, spoof, and NLP signals into a unified risk score."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from backend.config.settings import risk_weights, thresholds


def keyword_scam_score(text: str) -> float:
    keywords = [
        "press 1",
        "otp",
        "bank account",
        "verify your account",
        "refund",
        "fraud department",
        "suspicious activity",
        "payment problem",
        "amazon fraud",
        "cancel your order",
    ]

    if not text:
        return 0.0

    text = text.lower()
    score = 0

    for word in keywords:
        if word in text:
            score += 1

    return score / len(keywords)


@dataclass
class RiskBreakdown:
    biometric_distance: float
    spoof_probability: float
    intent_probability: float
    heuristic_score: float
    overall_score: float
    tier: str
    notes: Dict[str, str]


def _tier(score: float) -> str:
    if score < 20:
        return "Safe"
    if score < 30:
        return "Review"
    return "Block"


def _heuristics(features: Dict[str, float]) -> float:
    replay_flag = 1.0 if features.get("replay_detected", False) else 0.0
    jitter = features.get("phase_jitter", 0.0)
    jitter_penalty = min(1.0, jitter / thresholds.replay_jitter_ms)
    return 0.5 * replay_flag + 0.5 * jitter_penalty


def fuse_scores(
    biometric_distance: float,
    spoof_probability: float,
    intent_probability: float,
    transcript: str,
    features: Dict[str, float],
) -> RiskBreakdown:

    biometric_score = 1.0 - min(1.0, biometric_distance / thresholds.biometric_pass)

    heuristic_score = _heuristics(features)

    scam_score = keyword_scam_score(transcript)

    overall = (
        biometric_score * risk_weights.biometric_weight
        + spoof_probability * risk_weights.deepfake_weight
        + intent_probability * risk_weights.intent_weight
        + heuristic_score * risk_weights.heuristics_weight
        + scam_score * 0.3
    ) * 100

    notes = {
        "biometric_pass": str(biometric_distance < thresholds.biometric_pass),
        "deepfake_alert": str(spoof_probability > thresholds.deepfake_alert),
        "intent_alert": str(intent_probability > thresholds.intent_alert),
        "scam_keywords": str(scam_score > 0.2),
    }

    return RiskBreakdown(
        biometric_distance=biometric_distance,
        spoof_probability=spoof_probability,
        intent_probability=intent_probability,
        heuristic_score=heuristic_score,
        overall_score=float(overall),
        tier=_tier(overall),
        notes=notes,
    )