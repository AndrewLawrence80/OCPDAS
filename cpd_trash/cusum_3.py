import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic time series data with a change in distribution
np.random.seed(0)
n = 200
time = np.arange(n)
mu1, mu2 = 0, 5  # means before and after the change point
sigma = 1
change_point = 100
data = np.concatenate([np.random.normal(mu1, sigma, change_point),
                       np.random.normal(mu2, sigma, n - change_point)])

# Define CUSUM of ranks algorithm
def cusum_ranks(data, window_size, threshold):
    """Detects change points using CUSUM of ranks algorithm."""
    n = len(data)
    cusum_values = np.zeros(n)
    rank_values = np.zeros(n)

    for i in range(n - window_size):
        window = data[i:i+window_size]
        ranks = np.argsort(window)
        rank_values[i] = np.mean(ranks)
        cusum_values[i] = max(0, cusum_values[i-1] + (rank_values[i] - np.mean(rank_values[:i+1])))

    return np.where(cusum_values > threshold * np.max(cusum_values))[0]

# Set window size and threshold for change point detection
window_size = 10
threshold = 0.5

# Detect change points using CUSUM of ranks
change_points = cusum_ranks(data, window_size, threshold)

# Plot the time series data with detected change points
plt.figure(figsize=(10, 6))
plt.plot(time, data, label='Time Series Data')
plt.scatter(change_points, data[change_points], color='red', marker='o', label='Change Points')
plt.axvline(x=change_point, color='green', linestyle='--', label='True Change Point')
plt.title('Change Point Detection of Data Distribution using CUSUM of Ranks')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()