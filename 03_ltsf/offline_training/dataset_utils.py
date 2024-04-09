import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import torch


def split_data(x: np.ndarray, p_train: float, p_val: float, p_test: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test set.
    """
    if 1 != p_train+p_val+p_test:
        raise ValueError("p_train, p_val, p_test must sum up to 1")
    len_dataset = len(x)
    len_train = int(np.floor(p_train*len_dataset))
    len_val = int(np.floor((p_train+p_val)*len_dataset))-len_train
    x_train = x[:len_train]
    x_val = x[len_train:len_train+len_val]
    x_test = x[len_train+len_val:]
    return x_train, x_val, x_test


def min_max_normalization(x: np.ndarray):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_x = scaler.fit_transform(x)
    return scaled_x, scaler


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, n_lookback: int, n_predict: int) -> None:
        super().__init__()
        self.data = data
        self.n_lookback = n_lookback
        self.n_predict = n_predict

    def __len__(self):
        # if data length is 4, n_lookback is 2, and n_predict=1
        # then total training sample will be 4-2+1=2
        return len(self.data)-self.n_lookback-self.n_predict+1

    def __getitem__(self, index) -> torch.Tensor:
        x = self.data[index:index+self.n_lookback, :]
        y = self.data[index+self.n_lookback:index+self.n_lookback+self.n_predict, :]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

    def get(self, index) -> torch.Tensor:
        return self.__getitem__(index)
