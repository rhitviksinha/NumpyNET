""" Activation functions and their respective gradients
"""
import numpy as np
from numpy import ndarray as Vector

def sigmoid(x: Vector) -> Vector:
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x: Vector) -> Vector:
    val = sigmoid(x)
    return val * (1.0 - val)

def relu(x: Vector) -> Vector:
    return (np.abs(x) + x) / 2.0

def relu_prime(x: Vector) -> Vector:
    x_ = np.array(x)
    x_[x_>0] = 1.0
    return x_

def tanh(x: Vector) -> Vector:
    return np.tanh(x)

def tanh_prime(x: Vector) -> Vector:
    val = tanh(x)
    return 1.0-(val**2)

def softmax(x: Vector) -> Vector:
    # return (np.exp(x) / np.sum(np.exp(x)))
    raise NotImplementedError
