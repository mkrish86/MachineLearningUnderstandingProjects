import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input values
x = np.linspace(-10, 10, 100)
# Apply sigmoid function
y = sigmoid(x)
#y = sigmoid(y)

# Plot
plt.plot(x, y)
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()