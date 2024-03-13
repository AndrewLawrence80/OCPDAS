import torch
import torch.nn as nn


class LSTMQNet(nn.Module):
    def __init__(self, n_features, hidden_size, num_actions, num_layers) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        out = self.linear(h_n[-1])
        return out
