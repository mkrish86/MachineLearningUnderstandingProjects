import numpy as np

# Dataset
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 2.8, 3.6, 4.5, 5.1])

# Number of training examples
m = len(x)

# Initialize parameters
theta_0 = 0
theta_1 = 0

# Learning rate
alpha = 0.01

# Number of iterations
iterations = 4500

# Gradient Descent
for _ in range(iterations):
    # Compute the hypothesis
    h = theta_0 + theta_1 * x
    
    # Compute the gradients
    gradient_0 = (1/m) * np.sum(h - y)
    gradient_1 = (1/m) * np.sum((h - y) * x)
    
    # Update the parameters
    theta_0 -= alpha * gradient_0
    theta_1 -= alpha * gradient_1

# Output the results
print(f"theta_0: {theta_0}, theta_1: {theta_1}")

# Predictions
predictions = theta_0 + theta_1 * x
print(f"Predictions: {predictions}")