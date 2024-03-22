import torch
import torch.nn as nn


class LSTMMdel(nn.Module):
    def __init__(self, n_features, hidden_size, output_size, num_layers) -> None:
        super().__init__()
        self.output_size = output_size
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        out = self.linear(h_n[-1])
        return out

def get_model():
    return LSTMMdel(1, 64, 2, 1)