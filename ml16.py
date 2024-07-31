import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Generate synthetic data
np.random.seed(42)
n_samples = 300
mean1 = [0, 0]
cov1 = [[1, 0.1], [0.1, 1]]
mean2 = [5, 5]
cov2 = [[1, -0.1], [-0.1, 1]]

X = np.vstack([
    np.random.multivariate_normal(mean1, cov1, n_samples),
    np.random.multivariate_normal(mean2, cov2, n_samples)
])

# Number of components
k = 2

# Initialize the parameters
np.random.seed(42)
pi = np.ones(k) / k  # Mixing coefficients
means = np.random.rand(k, 2)  # Means of the Gaussians
covariances = np.array([np.eye(2)] * k)  # Covariances of the Gaussians

def e_step(X, pi, means, covariances):
    N = X.shape[0]
    r = np.zeros((N, k))
    for i in range(k):
        r[:, i] = pi[i] * multivariate_normal.pdf(X, mean=means[i], cov=covariances[i])
    r = r / r.sum(axis=1, keepdims=True)
    return r

def m_step(X, r):
    N, D = X.shape
    pi = r.sum(axis=0) / N
    means = np.dot(r.T, X) / r.sum(axis=0)[:, np.newaxis]
    covariances = np.zeros((k, D, D))

    for i in range(k):
        diff = X - means[i]
        covariances[i] = np.dot(r[:, i] * diff.T, diff) / r[:, i].sum()

    return pi, means, covariances

def log_likelihood(X, pi, means, covariances):
    N = X.shape[0]
    log_likelihood = 0

    for i in range(k):
        log_likelihood += pi[i] * multivariate_normal.pdf(X, mean=means[i], cov=covariances[i])
    return np.log(log_likelihood).sum()

# Run the EM algorithm
max_iter = 100
tol = 1e-6
log_likelihoods = []

for iteration in range(max_iter):
    r = e_step(X, pi, means, covariances)
    pi, means, covariances = m_step(X, r)
    log_likelihoods.append(log_likelihood(X, pi, means, covariances))

    # Check for convergence
    if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
        break

# Plot the log-likelihoods
plt.figure(figsize=(10, 6))
plt.plot(log_likelihoods)
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood Convergence')
plt.show()

# Plot the final clusters
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=r.argmax(axis=1), cmap='viridis', marker='o')
plt.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=100)
plt.title('Clusters after EM Algorithm')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()