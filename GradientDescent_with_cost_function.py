import numpy as np
import matplotlib.pyplot as plt

# Define the cost function and its derivative
def f(x):
    return x**2

def f_prime(x):
    return 2 * x

# Gradient Descent parameters
x = 10  # Initial guess
alpha = 0.1  # Learning rate
iterations = 20  # Number of iterations

# Lists to store the progress
x_values = [x]
f_values = [f(x)]

# Perform Gradient Descent
for _ in range(iterations):
    x = x - alpha * f_prime(x)
    x_values.append(x)
    f_values.append(f(x))

# Plotting the cost function
x_plot = np.linspace(-10, 10, 400)
y_plot = f(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label='f(x) = x^2')
plt.scatter(x_values, f_values, color='red')
plt.plot(x_values, f_values, color='red', linestyle='--', marker='o', label='Gradient Descent')
plt.title('Gradient Descent on f(x) = x^2')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()

# Print the final value
print(f"The value of x that minimizes f(x) is approximately: {x_values[-1]}")