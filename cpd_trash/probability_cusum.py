import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 1: Generate data points for the Gaussian (normal) distribution
mean = 0
std_dev = 1
x_values = np.linspace(-5, 5, 1000)  # Adjust range as needed
pdf = norm.pdf(x_values, mean, std_dev)  # Probability density function values

# Step 2: Plot the Gaussian distribution
plt.plot(x_values, pdf, label='Gaussian Distribution', color='blue')

# Step 3: Plot a vertical line at a custom x value with its cumulative distribution function (CDF)
custom_x = 1.5  # Custom x value
pdf_at_custom_x = norm.pdf(custom_x, mean, std_dev)  # PDF at custom x

# Find the index of the x-value closest to the custom_x
index = np.abs(x_values - custom_x).argmin()

# Plot the vertical line
plt.plot([custom_x, custom_x], [0, pdf_at_custom_x], color='red', linestyle='--')
plt.scatter(custom_x, pdf_at_custom_x, color='red', zorder=5)  # Mark the point where the line meets the curve

# Step 4: Add labels and title
plt.title('Normal Distribution with Custom CDF')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)
plt.legend()
plt.show()