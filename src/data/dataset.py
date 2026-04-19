"""
Dataset classes for accent-robust ASR training.
Supports: LibriSpeech, CommonVoice, L2-ARCTIC, AESRC2020.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from .preprocessing import AudioPreprocessor


# Accent label mapping across datasets
ACCENT_LABELS = {
    "american": 0,
    "british": 1,
    "indian": 2,
    "chinese": 3,
    "arabic": 4,
    "korean": 5,
    "russian": 6,
    "spanish": 7,
}

ACCENT_NAMES = {v: k for k, v in ACCENT_LABELS.items()}
NUM_ACCENTS = len(ACCENT_LABELS)


class AccentedSpeechDataset(Dataset):
    """
    Unified dataset class for accented speech corpora.

    Expected manifest format (JSON lines):
        {"audio_path": "...", "text": "...", "accent": "indian", "gender": "F", "age": "25-34"}
    """

    def __init__(
        self,
        manifest_path: str,
        preprocessor: AudioPreprocessor,
        split: str = "train",
        max_duration: float = 20.0,
        accent_filter: Optional[List[str]] = None,
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.split = split
        self.max_duration = max_duration

        self.samples = self._load_manifest(manifest_path, accent_filter)
        print(f"[{split}] Loaded {len(self.samples)} samples from {manifest_path}")

    def _load_manifest(
        self,
        path: str,
        accent_filter: Optional[List[str]],
    ) -> List[Dict]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if accent_filter and item.get("accent") not in accent_filter:
                    continue
                if "accent" in item and item["accent"] not in ACCENT_LABELS:
                    item["accent_id"] = -1  # unknown accent
                else:
                    item["accent_id"] = ACCENT_LABELS.get(item.get("accent", "american"), 0)
                samples.append(item)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        try:
            waveform, sr = self.preprocessor.load_audio(sample["audio_path"])
            input_values = self.preprocessor.process(waveform, sr)
        except Exception as e:
            print(f"Error loading {sample['audio_path']}: {e}")
            input_values = torch.zeros(16000)  # 1 sec silence fallback

        text = sample.get("text", "").strip().upper()
        label_ids = self.preprocessor.encode_text([text])[0]

        return {
            "input_values": input_values,
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "accent_id": torch.tensor(sample["accent_id"], dtype=torch.long),
            "text": text,
            "accent": sample.get("accent", "unknown"),
            "gender": sample.get("gender", "unknown"),
            "age": sample.get("age", "unknown"),
            "audio_path": sample["audio_path"],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function with padding for variable-length audio.
    """
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad input_values
    max_input_len = max(iv.shape[0] for iv in input_values)
    padded_inputs = torch.zeros(len(batch), max_input_len)
    attention_mask = torch.zeros(len(batch), max_input_len, dtype=torch.long)

    for i, iv in enumerate(input_values):
        padded_inputs[i, : iv.shape[0]] = iv
        attention_mask[i, : iv.shape[0]] = 1

    # Pad labels with -100 (ignored in CTC loss)
    max_label_len = max(l.shape[0] for l in labels)
    padded_labels = torch.full((len(batch), max_label_len), fill_value=-100, dtype=torch.long)
    for i, l in enumerate(labels):
        padded_labels[i, : l.shape[0]] = l

    return {
        "input_values": padded_inputs,
        "attention_mask": attention_mask,
        "labels": padded_labels,
        "accent_ids": torch.stack([item["accent_id"] for item in batch]),
        "texts": [item["text"] for item in batch],
        "accents": [item["accent"] for item in batch],
        "genders": [item["gender"] for item in batch],
        "ages": [item["age"] for item in batch],
    }


def create_dataloader(
    manifest_path: str,
    preprocessor: AudioPreprocessor,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    accent_filter: Optional[List[str]] = None,
    sampler=None,
) -> DataLoader:
    dataset = AccentedSpeechDataset(
        manifest_path=manifest_path,
        preprocessor=preprocessor,
        split=split,
        accent_filter=accent_filter,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
