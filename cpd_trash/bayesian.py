import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

# Generate synthetic time series data with a change point
np.random.seed(0)
n_samples = 200
time = np.arange(n_samples)
data = np.concatenate([np.random.normal(10, 1, n_samples // 2),
                       np.random.normal(15, 1, n_samples // 2)])

# Apply Bayesian Online Changepoint Detection (BOCPD) algorithm
model = "rbf"  # Change to "l2" for linear cost
algo = rpt.Binseg(model=model).fit(data)
result = algo.predict(pen=np.log(n_samples) ** 2)

# Plot the time series data and detected change points
plt.figure(figsize=(10, 6))
plt.plot(time, data, label='Time Series Data')
for cp in result[:-1]:
    plt.axvline(x=cp, color='red', linestyle='--', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Change Point Detection with Bayesian Methods (BOCPD)')
plt.legend()
plt.grid(True)
plt.show()