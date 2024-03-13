import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SELU())
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.SELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.permute(0, 2, 1)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        encoded = self.encoder(x)
        encoded = encoded.permute(0, 2, 1)
        return encoded


class LSTMAutoEncoder(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, n_features, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        encoded_seq, _ = self.encoder(x)
        # Decoder
        decoded_seq, _ = self.decoder(encoded_seq)
        return decoded_seq

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        return encoded
