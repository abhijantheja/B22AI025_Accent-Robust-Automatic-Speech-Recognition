"""
Accent-stratified sampling to prevent majority accent dominance during training.
Ensures balanced representation across all accent groups.
"""

import torch
import numpy as np
from torch.utils.data import Sampler
from typing import Dict, List, Iterator, Optional
from collections import Counter


class AccentStratifiedSampler(Sampler):
    """
    Stratified sampler that balances accent distribution in each batch.

    Strategy:
    - Computes class weights inversely proportional to accent frequency.
    - Oversamples minority accents; undersamples majority accents.
    - Ensures each epoch sees approximately balanced accent distribution.
    """

    def __init__(
        self,
        dataset,
        replacement: bool = True,
        num_samples: Optional[int] = None,
    ):
        self.dataset = dataset
        self.replacement = replacement
        self._num_samples = num_samples

        # Extract accent labels
        self.accent_ids = [sample["accent_id"] for sample in dataset.samples]
        self.weights = self._compute_weights()

    def _compute_weights(self) -> torch.Tensor:
        """Inverse frequency weighting per accent class."""
        counter = Counter(self.accent_ids)
        total = len(self.accent_ids)
        weights = []
        for accent_id in self.accent_ids:
            weight = total / (len(counter) * counter[accent_id])
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.double)

    @property
    def num_samples(self) -> int:
        return self._num_samples or len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(
            self.weights,
            num_samples=self.num_samples,
            replacement=self.replacement,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def get_accent_distribution(self) -> Dict[int, float]:
        """Report expected accent distribution under this sampler."""
        total_weight = self.weights.sum().item()
        per_accent = {}
        accent_ids_unique = set(self.accent_ids)
        for accent_id in accent_ids_unique:
            mask = [w for a, w in zip(self.accent_ids, self.weights.tolist()) if a == accent_id]
            per_accent[accent_id] = sum(mask) / total_weight
        return per_accent


