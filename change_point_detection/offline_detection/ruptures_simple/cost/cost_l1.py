"""
Copy from [Ruptures](https://centre-borelli.github.io/ruptures-docs/) 
"""
from __future__ import annotations
from .base_cost import BaseCost
import numpy as np


class CostL1(BaseCost):
    """Least absolute deviation."""

    model = "l1"

    def __init__(self) -> None:
        """Initialize the object."""
        self.signal = None
        self.min_size = 1

    def fit(self, signal: np.ndarray) -> CostL1:
        """Set parameters of the instance.

        Args:
            signal (array): signal. Shape (n_samples,) or (n_samples, n_features)

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

        """
        if end - start < self.min_size:
            raise ValueError("Segment too short")
        sub = self.signal[start:end]
        med = np.median(sub, axis=0)
        return np.sum(np.abs(sub-med))

    def get_min_size(self):
        return self.min_size
