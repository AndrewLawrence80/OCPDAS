import torch.nn as nn


class NLinear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, n_lookback, n_predict):
        super().__init__()
        self.linear = nn.Linear(n_lookback, n_predict)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]
