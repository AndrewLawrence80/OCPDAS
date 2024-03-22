import torch
import matplotlib.pyplot as plt

# Number of samples
n = 15

# Calculate degrees of freedom
nu = n - 1

# Create a Student's t-distribution object
t_dist = torch.distributions.StudentT(df=nu)

# Generate random samples from the t-distribution
samples = t_dist.sample([1000])  # Generate 1000 samples

# Plot the histogram of the generated samples
plt.hist(samples.numpy(), bins=50, density=True, alpha=0.6, color='g')

# Plot the probability density function (PDF) of the t-distribution
x = torch.linspace(-5, 5, 1000)
pdf = t_dist.log_prob(x).exp()
plt.plot(x.numpy(), pdf.numpy(), 'r-', lw=2)

plt.title("Student's t-distribution with df={}".format(nu))
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend(["PDF", "Samples"])
plt.show()