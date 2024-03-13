import numpy as np
import matplotlib.pyplot as plt

# Define the probability density function f(tau)
def f(tau):
    # Define your probability density function here
    return np.exp(-tau)  # Example: Exponential distribution

# Define the range of tau values
tau_values = np.linspace(0, 10, 100)

# Calculate the denominator sum
denominator_sum = np.sum(f(tau_values))

# Calculate H(T) for each tau
H_T_values = f(tau_values) / denominator_sum

# Plot the distribution of H(T) against tau
plt.plot(tau_values, H_T_values)
plt.xlabel('Tau')
plt.ylabel('H(T)')
plt.title('Distribution of H(T)')
plt.grid(True)
plt.show()