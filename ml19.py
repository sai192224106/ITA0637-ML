import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Step 1: Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 3 + 4 * X + np.random.randn(100, 1)

# Step 2: Implement Linear Regression
def linear_regression(X, y):
    # Add bias term
    X_b = np.c_[np.ones((len(X), 1)), X]
    # Normal equation: theta = (X^T * X)^-1 * X^T * y
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best

# Fit linear regression model
theta_best = linear_regression(X, y)

# Extract coefficients
intercept, slope = theta_best[0], theta_best[1]

# Step 3: Show Performance
# Predictions
y_predict = np.dot(np.c_[np.ones((len(X), 1)), X], theta_best)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y, y_predict)
# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_predict, color='red', label='Linear Regression')
plt.title(f'Linear Regression\nMSE: {mse:.2f}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

print(f'Intercept (theta_0): {intercept[0]:.2f}')
print(f'Slope (theta_1): {slope[0]:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')