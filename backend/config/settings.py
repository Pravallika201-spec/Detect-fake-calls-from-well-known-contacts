"""Centralized configuration for the Offline Edge AI Call Spoof Detection System."""
from pathlib import Path
from pydantic import BaseModel

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data and model directories
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "checkpoints"

# Contact store
CONTACT_STORE = DATA_DIR / "trusted_contacts.json"


class Thresholds(BaseModel):
    biometric_pass: float = 0.65
    deepfake_alert: float = 0.6
    intent_alert: float = 0.55
    replay_jitter_ms: float = 35.0


class RiskWeights(BaseModel):
    biometric_weight: float = 0.4
    deepfake_weight: float = 0.2
    intent_weight: float = 0.3
    heuristics_weight: float = 0.1


thresholds = Thresholds()
risk_weights = RiskWeights()



EDGE_DEVICE = {
    "device": "Jetson Orin Nano",
    "precision": "fp16",
    "sample_rate": 16000,
    "frame_ms": 800,
}
