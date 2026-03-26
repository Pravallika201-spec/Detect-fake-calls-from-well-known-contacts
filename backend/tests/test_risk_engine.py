"""Minimal tests for the risk fusion logic."""
from backend.services import risk_engine


def test_safe_risk_bucket():
    result = risk_engine.fuse_scores(
        biometric_distance=0.3,
        spoof_probability=0.1,
        intent_probability=0.05,
        features={"replay_detected": False, "phase_jitter": 5.0},
    )
    assert result.tier == "Safe"
    assert result.overall_score < 40


def test_block_risk_bucket():
    result = risk_engine.fuse_scores(
        biometric_distance=0.9,
        spoof_probability=0.95,
        intent_probability=0.8,
        features={"replay_detected": True, "phase_jitter": 60.0},
    )
    assert result.tier == "Block"
    assert result.overall_score >= 70
