from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Iterable, Tuple
import torch
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler


def split_data(x: np.ndarray, n_lookback: int, n_predict: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data into train, test set.
    The file is different from `dataset_utils.py` in `offline_training`.
    For online training, experiments using the last example for testing and the others for trainning.
    """
    len_dataset = len(x)
    if len_dataset < 2*(n_lookback+n_predict):
        raise ValueError("History too short, must have 1 sample for train and 1 sample for test")
    x_train = x[:len_dataset-n_predict-n_lookback]
    x_test = x[len_dataset-n_predict-n_lookback:]
    return x_train, x_test


def min_max_normalization(x: np.ndarray):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_x = scaler.fit_transform(x)
    return scaled_x, scaler


def standard_normalization(x: np.ndarray):
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x)
    return scaled_x, scaler


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray,  n_lookback: int, n_predict: int) -> None:
        super().__init__()
        self.data = data
        self.n_lookback = n_lookback
        self.n_predict = n_predict

    def __len__(self):
        return len(self.data)-self.n_lookback-self.n_predict+1

    def __getitem__(self, index) -> torch.Tensor:
        x = self.data[index:index+self.n_lookback, :]
        y = self.data[index+self.n_lookback:index+self.n_lookback+self.n_predict, :]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

    def get(self, index) -> torch.Tensor:
        return self.__getitem__(index)
