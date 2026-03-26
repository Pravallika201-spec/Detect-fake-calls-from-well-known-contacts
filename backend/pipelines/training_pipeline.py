"""CLI-friendly training orchestrator for biometric, spoof, and intent models."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from backend.config.settings import DATA_DIR, MODEL_DIR
from backend.models.deepfake_detector import SpectrogramCNN
from backend.models.nlp_behavior import BehaviorClassifier
from backend.models.voice_biometric import SiameseEncoder
from backend.services.feature_extractor import make_spectrogram


class AudioFeatureDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        blob = np.load(path)

        mfcc = blob["mfcc"]
        label = blob["label"].item()
        spec = blob["spectrogram"]

        return (
            torch.from_numpy(mfcc).float(),
            torch.tensor(label).float(),
            torch.from_numpy(spec).float(),
        )


def train_biometric(dataset: AudioFeatureDataset, output: Path) -> None:
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = SiameseEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.TripletMarginLoss(margin=0.3)

    anchors = []

    for mfcc, label, _ in loader:
        anchors.append(mfcc)

    if not anchors:
        raise RuntimeError("No training samples found for biometric model")

    mfcc_tensor = torch.cat(anchors, dim=0)

    for epoch in range(3):
        optimizer.zero_grad()

        emb = model(mfcc_tensor)

        loss = loss_fn(
            emb,
            emb.roll(1, 0),
            emb.roll(2, 0),
        )

        loss.backward()
        optimizer.step()

    output.parent.mkdir(parents=True, exist_ok=True)

    scripted = torch.jit.script(model)
    scripted.save(str(output))


def train_deepfake(dataset: AudioFeatureDataset, output: Path) -> None:
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SpectrogramCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(5):
        for _, labels, specs in loader:

            preds = model(specs.unsqueeze(1))

            loss = loss_fn(preds.squeeze(), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    scripted = torch.jit.script(model)
    scripted.save(str(output))


def train_intent(texts: list[str], labels: list[int]) -> None:
    classifier = BehaviorClassifier()
    classifier.fit(texts, labels)


def preprocess_raw_audio(raw_dir: Path, processed_dir: Path) -> list[Path]:

    processed_dir.mkdir(parents=True, exist_ok=True)

    feature_paths: list[Path] = []

    import librosa

    max_height = 256
    max_frames = 200

    for wav in raw_dir.glob("*.wav"):

        # load audio
        samples, sr = librosa.load(wav, sr=16000)

        # create spectrogram
        spec = make_spectrogram(samples, sr=16000)

        # ensure spectrogram is 2D
        if spec.ndim == 3:
            spec = spec.squeeze()

        # FIX HEIGHT (frequency bins)
        if spec.shape[0] > max_height:
            spec = spec[:max_height, :]
        elif spec.shape[0] < max_height:
            pad = max_height - spec.shape[0]
            spec = np.pad(spec, ((0, pad), (0, 0)), mode="constant")

        # FIX WIDTH (time frames)
        if spec.shape[1] > max_frames:
            spec = spec[:, :max_frames]
        elif spec.shape[1] < max_frames:
            pad = max_frames - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad)), mode="constant")

        # create MFCC-like feature
        mfcc = librosa.feature.mfcc(
            y=samples,
            sr=16000,
            n_mfcc=40
        )
        if mfcc.shape[1] > 200:
            mfcc = mfcc[:, :200]
        elif mfcc.shape[1] < 200:
            pad = 200 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode="constant")

        # ensure MFCC always (1,200)
        if mfcc.shape[1] > 200:
            mfcc = mfcc[:, :200]
        elif mfcc.shape[1] < 200:
            pad = 200 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")

        payload = {
            "mfcc": mfcc,
            "spectrogram": spec,
            "label": np.array(0),
        }

        out_path = processed_dir / f"{wav.stem}.npz"

        np.savez(out_path, **payload)

        feature_paths.append(out_path)

    return feature_paths
def main() -> None:
    parser = argparse.ArgumentParser(description="Training pipeline")

    parser.add_argument(
        "--stage",
        choices=["preprocess", "biometric", "deepfake", "intent", "all"],
        default="all",
    )

    args = parser.parse_args()

    raw_dir = DATA_DIR / "raw"
    processed_dir = DATA_DIR / "processed"

    feature_paths = []

    if args.stage in ("preprocess", "all"):

        feature_paths = preprocess_raw_audio(raw_dir, processed_dir)

        print(f"Preprocessed {len(feature_paths)} files")

    if not feature_paths:
        feature_paths = list(processed_dir.glob("*.npz"))

    dataset = AudioFeatureDataset(feature_paths)

    if args.stage in ("biometric", "all"):
        train_biometric(dataset, MODEL_DIR / "voice_embedder.torchscript")

    if args.stage in ("deepfake", "all"):
        train_deepfake(dataset, MODEL_DIR / "deepfake_detector.torchscript")

    if args.stage in ("intent", "all"):

        # FIX: need two classes
        texts = [
            "Your bank account is locked please send OTP",
            "This is bank support verify your account number",
            "Hello how are you today",
            "Call me when you are free",
        ]

        labels = [
            1,
            1,
            0,
            0,
        ]

        train_intent(texts, labels)


if __name__ == "__main__":
    main()