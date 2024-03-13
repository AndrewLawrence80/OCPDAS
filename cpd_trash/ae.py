import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Generate synthetic time series data with a change point
np.random.seed(0)
n_samples = 200
time = np.arange(n_samples)
data = np.concatenate([np.random.normal(10, 1, n_samples // 2),
                       np.random.normal(15, 1, n_samples // 2)])

# Normalize the data
data_mean = np.mean(data)
data_std = np.std(data)
data_normalized = (data - data_mean) / data_std

# Convert data to PyTorch tensors
data_tensor = torch.FloatTensor(data_normalized).view(-1, 1)

# Define Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.SELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the Autoencoder model
model = Autoencoder()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the Autoencoder
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(data_tensor)
    loss = criterion(outputs, data_tensor)
    loss.backward()
    optimizer.step()

# Reconstruct the data using the trained Autoencoder
model.eval()
with torch.no_grad():
    reconstructed_data_normalized = model(data_tensor).numpy()

# Calculate reconstruction error
reconstruction_error = np.mean(np.square(data_normalized - reconstructed_data_normalized), axis=1)

# Set a threshold for anomaly detection
threshold = np.mean(reconstruction_error) + 2 * np.std(reconstruction_error)

# Detect change points
change_points = np.where(reconstruction_error > threshold)[0]

# Plot the time series data, reconstruction, and detected change points
plt.figure(figsize=(10, 6))
plt.plot(time, data, label='Original Data')
plt.plot(time, reconstructed_data_normalized * data_std + data_mean, color='red', label='Reconstructed Data')
plt.scatter(change_points, data[change_points], color='green', label='Change Points')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Change Point Detection with Autoencoder (PyTorch)')
plt.legend()
plt.grid(True)
plt.show()