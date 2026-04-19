"""
Audio preprocessing utilities.
Handles loading, resampling, normalization, and feature extraction.
"""

import torch
import numpy as np
import librosa
import torchaudio
from transformers import Wav2Vec2Processor
from typing import List, Tuple, Optional


class AudioPreprocessor:
    """
    Preprocesses raw audio for Wav2Vec 2.0 input.
    - Resamples to 16kHz
    - Normalizes amplitude
    - Pads/truncates to max length
    """

    TARGET_SAMPLE_RATE = 16000
    MAX_DURATION_SECONDS = 20.0

    def __init__(self, processor_name: str = "facebook/wav2vec2-base"):
        self.processor = Wav2Vec2Processor.from_pretrained(processor_name)
        self.max_length = int(self.MAX_DURATION_SECONDS * self.TARGET_SAMPLE_RATE)

    def load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return waveform + sample rate."""
        waveform, sr = librosa.load(path, sr=self.TARGET_SAMPLE_RATE, mono=True)
        return waveform, sr

    def normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Peak normalization."""
        peak = np.abs(waveform).max()
        if peak > 0:
            waveform = waveform / peak
        return waveform

    def process(
        self,
        waveform: np.ndarray,
        sample_rate: int = None,
        truncate: bool = True,
    ) -> torch.Tensor:
        """
        Full preprocessing pipeline: resample -> normalize -> tokenize.

        Returns:
            input_values: (T,) float tensor ready for Wav2Vec 2.0
        """
        if sample_rate and sample_rate != self.TARGET_SAMPLE_RATE:
            waveform = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=self.TARGET_SAMPLE_RATE
            )

        waveform = self.normalize(waveform)

        if truncate and len(waveform) > self.max_length:
            waveform = waveform[: self.max_length]

        inputs = self.processor(
            waveform,
            sampling_rate=self.TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )
        return inputs.input_values.squeeze(0)  # (T,)

    def process_batch(
        self,
        waveforms: List[np.ndarray],
        padding: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch of waveforms with padding.

        Returns:
            input_values: (B, T_max) padded tensor
            attention_mask: (B, T_max) mask
        """
        processed = [self.normalize(w) for w in waveforms]
        inputs = self.processor(
            processed,
            sampling_rate=self.TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=padding,
        )
        return inputs.input_values, inputs.attention_mask

    def decode_labels(self, label_ids: torch.Tensor) -> List[str]:
        """Decode CTC output token IDs to text."""
        return self.processor.batch_decode(label_ids, skip_special_tokens=True)

    def encode_text(self, texts: List[str]) -> List[List[int]]:
        """Encode text transcriptions to token IDs."""
        return [
            self.processor.tokenizer.encode(text.upper())
            for text in texts
        ]


def compute_melspectrogram(
    waveform: np.ndarray,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
) -> np.ndarray:
    """Compute mel-spectrogram for Grad-CAM visualization."""
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return librosa.power_to_db(mel, ref=np.max)
