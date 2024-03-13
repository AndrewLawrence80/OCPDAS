import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Optional


class Evaluator:
    def __init__(self, model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: str) -> None:
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.device = device
        self.gt = None
        self.pd = None
        self.loss = None

    def evaluate(self):
        self.model.eval()
        self.gt = []
        self.pd = []
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in self.dataloader:

                self.gt.extend([y for y in batch_y.numpy()])

                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                prediction = self.model(batch_x)
                loss = self.loss_fn(prediction, batch_y)
                total_loss += loss.item()/len(self.dataloader)

                self.pd.extend([y for y in prediction.cpu().numpy()])

        self.gt = np.array(self.gt)
        self.pd = np.array(self.pd)

        self.loss = total_loss

    def get_gt(self) -> Optional[np.ndarray]:
        return self.gt

    def get_pd(self) -> Optional[np.ndarray]:
        return self.pd

    def get_loss(self) -> Optional[np.ndarray]:
        return self.loss


def MAE(gt: np.ndarray, pd: np.ndarray) -> float:
    return np.mean(np.abs(gt-pd))


def RMSE(gt: np.ndarray, pd: np.ndarray) -> float:
    return np.sqrt(np.mean(np.power(gt-pd, 2)))
