import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic time series data with a change point
np.random.seed(0)
n_samples = 200
time = np.arange(n_samples)
data = np.concatenate([np.random.normal(10, 1, n_samples // 2),
                       np.random.normal(15, 1, n_samples // 2)])

# Define parameters for the EWMA control chart
alpha = 0.2  # Smoothing parameter (0 < alpha < 1)
L = 3       # Number of standard deviations for control limits

# Compute the EWMA
ewma = [data[0]]  # Initialize the EWMA
for i in range(1, len(data)):
    ewma.append(alpha * data[i] + (1 - alpha) * ewma[-1])

# Compute the control limits
mean_ewma = np.mean(ewma)
std_ewma = np.std(ewma)
UCL = mean_ewma + L * std_ewma
LCL = mean_ewma - L * std_ewma

# Plot the time series data and the EWMA control chart
plt.figure(figsize=(10, 6))
plt.plot(time, data, label='Time Series Data')
plt.plot(time, ewma, color='red', label='EWMA')
plt.axhline(UCL, color='green', linestyle='--', label='UCL')
plt.axhline(LCL, color='green', linestyle='--', label='LCL')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('EWMA Control Chart for Change Point Detection')
plt.legend()
plt.grid(True)
plt.show()