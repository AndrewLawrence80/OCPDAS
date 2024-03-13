import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic time series data with a change point
np.random.seed(0)
n = 200
time = np.arange(n)
mu1, mu2 = 0, 5  # means before and after the change point
sigma = 1
change_point = 100
data = np.concatenate([np.random.normal(mu1, sigma, change_point),
                       np.random.normal(mu2, sigma, n - change_point)])

# Define CUSUM algorithm


def cusum(data, threshold):
    """Detects change points using CUSUM algorithm."""
    n = len(data)
    cusum_values = np.zeros(n)
    max_val = 0

    for i in range(1, n):
        cusum_values[i] = max(0, cusum_values[i-1] + (data[i] - np.mean(data[:i+1])))
        max_val = max(max_val, cusum_values[i])

    return cusum_values, np.where(cusum_values > threshold*max_val)[0]


# Set threshold for change point detection
threshold = 0.1

# Detect change points using CUSUM
cusum_values, change_points = cusum(data, threshold)

# Plot the time series data with detected change points
plt.figure(figsize=(10, 6))
plt.plot(time, data, label='Time Series Data')
plt.plot(time, cusum_values, label="CUSUM Values")
plt.scatter(change_points, data[change_points], color='red', marker='o', label='Change Points')
plt.axvline(x=change_point, color='green', linestyle='--', label='True Change Point')
plt.title('Change Point Detection using CUSUM')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
