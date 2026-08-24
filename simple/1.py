# ***

# ## Deep Learning & Generative AI - Exam Practice

# ## PRACTICAL 1 : Activation Functions

# ***

# #### Objective: Implement and visualize common activation functions used in deep learning.
# #### Theory Notes: What is an activation function?
#
# An activation function is a mathematical function applied to the output of a neuron.
# It introduces non-linearity into a neural network.
# Without activation functions, multiple neural network layers would behave like a single linear transformation.
# Different activation functions are chosen depending on the type of problem and network layer.

# ***

# ### 1] import Libraries
#
# only the libraries required for implementation and visualization are imported.

import numpy as np
import matplotlib.pyplot as plt

# Input values
x = np.linspace(-5, 5, 100)
print(x)

# ### 2] common plotting function
#
# this function plots an activation function and its derivative.

def plot_function(f, df, name):
    y = f(x)
    dy = df(x)

    plt.plot(x, y, label='f(x)')
    plt.plot(x, dy, label="f'(x)")
    plt.title(name)
    plt.xlabel('x')
    plt.ylabel('Output')
    plt.grid()
    plt.legend()
    plt.show()

# ### 3] binary step function

def binary_step(x):
    return np.where(x >= 0, 1, 0)

def binary_step_derivative(x):
    # Derivative is taken as 0 for visualization
    return np.zeros_like(x)

plot_function(binary_step, binary_step_derivative, 'Binary Step Function')

# ### 4] linear / identity function

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

plot_function(linear, linear_derivative, 'Linear / Identity Function')

# ### 5] sigmoid function

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

plot_function(sigmoid, sigmoid_derivative, 'Sigmoid Function')

# ### 6] tanh function

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

plot_function(tanh, tanh_derivative, 'Tanh Function')

# ### 7] ReLU function

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

plot_function(relu, relu_derivative, 'ReLU Function')

# ### 8] leaky ReLU function

alpha = 0.01

def leaky_relu(x):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x):
    return np.where(x > 0, 1, alpha)

plot_function(leaky_relu, leaky_relu_derivative, 'Leaky ReLU Function')

# ### 9] ELU function

alpha = 1

def elu(x):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x):
    return np.where(x > 0, 1, alpha * np.exp(x))

plot_function(elu, elu_derivative, 'ELU Function')

# ### 10] softplus function

def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_derivative(x):
    return sigmoid(x)

plot_function(softplus, softplus_derivative, 'Softplus Function')

# ### 11] swish / SiLU function

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

plot_function(swish, swish_derivative, 'Swish / SiLU Function')

# ### 12] GELU function

from scipy.special import erf

def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

# GELU derivative is omitted here to keep the exam implementation simple.
y = gelu(x)
plt.plot(x, y)
plt.title('GELU Function')
plt.xlabel('x')
plt.ylabel('Output')
plt.grid()
plt.show()

# ### 13] softmax function
#
# Softmax converts a set of logits into probabilities whose sum is 1.

# Example logits
logits = np.array([2.0, 1.0, 0.1])

# Numerically stable Softmax
exp_values = np.exp(logits - np.max(logits))
probabilities = exp_values / np.sum(exp_values)

print('Softmax:', probabilities)
print('Sum:', np.sum(probabilities))

# ***

# ### quick revision:
#
# | Activation | Main formula / idea |
# |---|---|
# | Binary Step | 0 or 1 |
# | Linear | `x` |
# | Sigmoid | `1 / (1 + exp(-x))` |
# | Tanh | `tanh(x)` |
# | ReLU | `max(0, x)` |
# | Leaky ReLU | `x` or `alpha*x` |
# | ELU | `x` or `alpha*(exp(x)-1)` |
# | Softplus | `log(1 + exp(x))` |
# | Swish | `x * sigmoid(x)` |
# | GELU | Gaussian-error-function based activation |
# | Softmax | Converts logits to probabilities; sum = 1 |

# ***
