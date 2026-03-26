"""In-memory store for trusted contact voice embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np

from backend.config.settings import CONTACT_STORE


@dataclass
class ContactProfile:
    contact_id: str
    name: str
    embedding: np.ndarray
    device: str
    notes: str = ""


class ContactRepository:
    def __init__(self, store_path: Path | None = None):
        self.path = store_path or CONTACT_STORE
        self.cache: Dict[str, ContactProfile] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for entry in payload:
            self.cache[entry["contact_id"]] = ContactProfile(
                contact_id=entry["contact_id"],
                name=entry["name"],
                embedding=np.array(entry["embedding"], dtype=np.float32),
                device=entry.get("device", "unknown"),
                notes=entry.get("notes", ""),
            )

    def save(self) -> None:
        serializable = [
            {
                "contact_id": profile.contact_id,
                "name": profile.name,
                "embedding": profile.embedding.tolist(),
                "device": profile.device,
                "notes": profile.notes,
            }
            for profile in self.cache.values()
        ]
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def list_gallery(self) -> List[Tuple[str, np.ndarray, Dict[str, str]]]:
        if not self.cache:
            # Return a dummy entry so the matcher can still run
            dummy_embedding = np.zeros(192, dtype=np.float32)
            return [("dummy", dummy_embedding, {"name": "Demo Contact", "device": "demo"})]
        return [
            (profile.contact_id, profile.embedding, {"name": profile.name, "device": profile.device})
            for profile in self.cache.values()
        ]

    def upsert(self, profile: ContactProfile) -> None:
        self.cache[profile.contact_id] = profile
        self.save()
