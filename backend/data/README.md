# Data Layout

```
data
├── raw/                 # unprocessed audio corpora (trusted contacts, ASVspoof, etc.)
├── processed/           # denoised, resampled WAV + cached features (.npz)
├── interim/             # temporary artifacts (spectrograms, embeddings)
└── trusted_contacts.json# encrypted biometric templates + metadata
```

All personally identifiable recordings must be encrypted at rest. Use `scripts/encrypt_contacts.py` (to be added) to rotate keys and update `trusted_contacts.json` whenever a contact re-samples their voice.
