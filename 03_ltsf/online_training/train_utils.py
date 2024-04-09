import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_utils import TimeSeriesDataset
import numpy as np


class Trainer:
    def __init__(self, model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, num_epochs: Optional[int], early_stop_gain: Optional[float], early_stop_loss: Optional[float], lr_scheduler: torch.optim.lr_scheduler._LRScheduler, device: str) -> None:
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.early_stop_gain = early_stop_gain
        self.early_stop_loss = early_stop_loss
        self.lr_scheduler = lr_scheduler
        self.device = device

    def train_by_early_stop(self):
        """
        Train until delta loss percentage no more than EARLY_STOP_GAIN or loss no more than EARLY_STOP_LOSS
        """
        self.model.train()
        epoch = 0

        loss_delta_percentage = 1
        loss_prev = 1e-5
        loss_current = 1

        while loss_delta_percentage > self.early_stop_gain and loss_current > self.early_stop_loss:
            total_loss = 0.0
            for batch_x, batch_y in self.dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                prediction = self.model(batch_x)
                loss = self.loss_fn(prediction, batch_y)
                total_loss += loss.item()/len(self.dataloader)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            loss_current = total_loss
            loss_delta_percentage = np.abs(loss_prev-loss_current)/loss_prev
            loss_prev = loss_current

            epoch += 1
            print("epoch: %d, loss: %f" % (epoch, loss))

    def train_one_epoch(self):
        total_loss = 0.0

        self.model.train()

        for batch_x, batch_y in self.dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            prediction = self.model(batch_x)
            loss = self.loss_fn(prediction, batch_y)
            total_loss += loss.item()/len(self.dataloader)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return total_loss

    def train_by_epoch(self):
        for epoch in range(self.num_epochs):
            loss = self.train_one_epoch()
            # print("epoch: %d, loss: %f" % (epoch+1, loss))
