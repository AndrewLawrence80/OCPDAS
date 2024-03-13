import numpy as np
import matplotlib.pyplot as plt

# Function to compute the likelihood of data given the mean and variance
def likelihood(data, mean, variance):
    return np.exp(-0.5 * ((data - mean) ** 2) / variance) / np.sqrt(2 * np.pi * variance)

# Function to update the posterior distribution
def update_posterior(data, prior_mean, prior_variance):
    likelihoods = likelihood(data, prior_mean, prior_variance)
    posterior_mean = (prior_variance * np.sum(data) + prior_mean) / (len(data) * prior_variance + 1)
    posterior_variance = 1 / (len(data) * (1 / prior_variance + 1))
    return posterior_mean, posterior_variance

# Function to detect changepoints
def detect_changepoints(data, threshold=1.0):
    changepoints = []
    prior_mean, prior_variance = 0, 1  # Initial prior parameters
    for i, datum in enumerate(data):
        posterior_mean, posterior_variance = update_posterior(np.array([datum]), prior_mean, prior_variance)
        likelihood_ratio = likelihood(np.array(data[i]), posterior_mean, posterior_variance) / likelihood(np.array(data[i]), prior_mean, prior_variance)
        if likelihood_ratio < threshold:
            changepoints.append(i)
            prior_mean, prior_variance = 0, 1  # Reset prior after detecting a changepoint
        else:
            prior_mean, prior_variance = posterior_mean, posterior_variance  # Update prior for next iteration
    return changepoints

# Generate synthetic data with changepoints
np.random.seed(0)
data = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)])

# Perform changepoint detection
changepoints = detect_changepoints(data)

# Plot the data and detected changepoints
plt.plot(data)
for cp in changepoints:
    plt.axvline(cp, color='r', linestyle='--')
plt.title('Bayesian Online Changepoint Detection')
plt.xlabel('Time')
plt.ylabel('Data')
plt.show()