import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import torch


def min_max_normalization(x: np.ndarray):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_x = scaler.fit_transform(x)
    return scaled_x, scaler


class VanillaDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Params
            x: np.ndarray of shape (n_samples, 1)
            y: np.ndarray of shape (n_samples, 1) 
        """
        super().__init__()
        if len(x) != len(y):
            raise ValueError("length of x and y not equal")
        self.x = x.reshape((-1, 1))
        self.y = y.reshape((-1, 1))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index) -> torch.Tensor:
        x = torch.tensor(self.x[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.float32)
        return x, y
