from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Iterable, Tuple
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def split_data(x: np.ndarray, n_lookback: int, n_predict: int) -> Tuple[np.ndarray, np.ndarray]:
    len_dataset = len(x)
    if len_dataset < 2*(n_lookback+n_predict):
        raise ValueError("History too short, must have 1 sample for train and 1 sample for test")
    x_train = x[:len_dataset-n_predict-n_lookback, :]
    x_test = x[len_dataset-n_predict-n_lookback:, :]
    return x_train, x_test


def min_max_normalization(x: np.ndarray):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_x = scaler.fit_transform(x)
    return scaled_x, scaler


def standard_normalization(x: np.ndarray):
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x)
    return scaled_x, scaler


class TimeSeriesAutoEncoderDataset(Dataset):
    def __init__(self, data: np.ndarray, len_window: int) -> None:
        super().__init__()
        self.data = data
        self.len_window = len_window

    def __len__(self):
        return len(self.data)-self.len_window+1

    def __getitem__(self, index) -> torch.Tensor:
        x = self.data[index:index+self.len_window, :]
        y = self.data[index:index+self.len_window, :]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

    def get(self, index) -> torch.Tensor:
        return self.__getitem__(index)
