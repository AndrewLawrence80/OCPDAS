import torch.nn as nn
import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, num_epochs: int, lr_scheduler: torch.optim.lr_scheduler.LRScheduler, device: str) -> None:
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.lr_scheduler = lr_scheduler
        self.device = device

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

    def train(self):
        for epoch in range(self.num_epochs):
            loss = self.train_one_epoch()
            print("epoch: %d, loss: %f" % (epoch+1, loss))
