"""Generate synthetic trusted/spoof audio samples for quick demos."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000
BASE = Path(__file__).resolve().parents[1] / "data" / "raw"


def synth_voice(freq: float, duration: float, phase: float, vibrato: float = 4.0) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    waveform = 0.6 * np.sin(2 * np.pi * freq * t + phase)
    mod = 0.05 * np.sin(2 * np.pi * vibrato * t)
    noise = 0.02 * np.random.randn(len(t))
    return (waveform + mod + noise).astype(np.float32)


SCENARIOS = [
    ("trusted", "alex_primary.wav", 185, 3.2, 0.0),
    ("trusted", "jordan_backup.wav", 205, 2.8, 0.4),
    ("trusted", "maria_mobile.wav", 195, 3.5, 0.9),
    ("spoof", "deepfake_alex.wav", 185, 3.2, 0.0),
    ("spoof", "robotic_bank.wav", 150, 3.0, 1.2),
    ("spoof", "replay_threat.wav", 165, 4.0, 0.3),
]


def main() -> None:
    for subdir, name, freq, duration, phase in SCENARIOS:
        target_dir = BASE / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        audio = synth_voice(freq, duration, phase)
        sf.write(target_dir / name, audio, SR)
        print(f"Wrote {subdir}/{name} ({duration:.1f}s at {freq}Hz)")


if __name__ == "__main__":
    main()
