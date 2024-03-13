import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt


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
        self.emb = []
        self.pd = []
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in self.dataloader:

                self.gt.extend([y for y in batch_y.numpy()])

                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                embedding = self.model.encode(batch_x)
                self.emb.extend([e for e in embedding.cpu().numpy()])
                prediction = self.model(batch_x)
                loss = self.loss_fn(prediction, batch_y)
                total_loss += loss.item()/len(self.dataloader)

                self.pd.extend([y for y in prediction.cpu().numpy()])

        self.gt = np.array(self.gt)
        self.emb = np.array(self.emb)
        self.pd = np.array(self.pd)

        self.loss = total_loss

    def get_gt(self) -> Optional[np.ndarray]:
        return self.gt

    def get_pd(self) -> Optional[np.ndarray]:
        return self.pd

    def get_emb(self) -> Optional[np.ndarray]:
        return self.emb

    def get_loss(self) -> Optional[np.ndarray]:
        return self.loss
