import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic time series data with a change point
np.random.seed(0)
n_samples = 200
time = np.arange(n_samples)
data = np.concatenate([np.random.normal(10, 1, n_samples // 2),
                       np.random.normal(15, 1, n_samples // 2)])

# Define the CUSUM algorithm
def cusum(data, threshold):
    cusum_values = np.zeros(len(data))
    change_points = []
    s = 0
    for i, x in enumerate(data):
        s = max(0, s + x - threshold)  # Update the CUSUM value
        cusum_values[i] = s
        if s > 0:
            change_points.append(i)
    return cusum_values, change_points

# Set the threshold for the CUSUM algorithm
threshold = 12.5  # Adjust as needed

# Apply the CUSUM algorithm to detect change points
cusum_values, change_points = cusum(data, threshold)

# Plot the time series data and the CUSUM values
plt.figure(figsize=(10, 6))
plt.plot(time, data, label='Time Series Data')
plt.plot(time, cusum_values, color='red', label='CUSUM')
plt.scatter(change_points, data[change_points], color='green', label='Change Points')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Change Point Detection with CUSUM')
plt.legend()
plt.grid(True)
plt.show()
