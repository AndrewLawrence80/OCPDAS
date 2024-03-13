import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import torch


def split_data(x: np.ndarray, p_train: float, p_val: float, p_test: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if 1 != p_train+p_val+p_test:
        raise ValueError("p_train, p_val, p_test must sum up to 1")
    len_dataset = len(x)
    len_train = int(np.floor(p_train*len_dataset))
    len_val = int(np.floor((p_train+p_val)*len_dataset))-len_train
    x_train = x[:len_train]
    x_val = x[len_train:len_train+len_val]
    x_test = x[len_train+len_val:]
    return x_train, x_val, x_test


class TimeSeriesClassificationDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index) -> torch.Tensor:
        x = self.x[index]
        y = self.y[index]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

    def get(self, index) -> torch.Tensor:
        return self.__getitem__(index)
