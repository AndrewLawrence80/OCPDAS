"""
Copy from [Ruptures](https://centre-borelli.github.io/ruptures-docs/) 
"""
from __future__ import annotations
from .base_cost import BaseCost
import numpy as np


class CostL2(BaseCost):
    """Least squared deviation."""

    model = "l2"

    def __init__(self):
        """Initialize the object."""
        self.signal = None
        self.min_size = 1

    def fit(self, signal: np.ndarray) -> CostL2:
        """Set parameters of the instance.

        Args:
            signal (array): array of shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        return self

    def error(self, start: int, end: int) -> float:
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            segment cost

        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size` samples).
        """
        if end - start < self.min_size:
            raise ValueError("Segment too short")

        return (end-start)*np.sum(np.var(self.signal[start:end], axis=0))

    def get_min_size(self):
        return self.min_size
