""" Neural Network layer to forward pass the inputs.
    And back-propagate the gradients.
"""
import numpy as np
from typing import Dict, Callable
from numpy import ndarray as Vector
from numpyNET.activation import relu, relu_prime, sigmoid, sigmoid_prime, tanh, tanh_prime

class Layer:
    def __init__(self) -> None:
        self.parameters: Dict[str, Vector] = {}
        self.gradients: Dict[str, Vector] = {}

    def forward_pass(self, inputs: Vector) -> Vector:
        raise NotImplementedError

    def backward_pass(self, grad: Vector) -> Vector:
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.parameters["w"] = np.random.randn(input_size, output_size)
        self.parameters["b"] = np.random.randn(output_size)
    
    def forward_pass(self, inputs: Vector) -> Vector:
        self.inputs = inputs
        return (inputs @ self.parameters["w"]) + self.parameters["b"]
    
    def backward_pass(self, grad: Vector) -> Vector:
        self.gradients["b"] = np.sum(grad, axis=0)
        self.gradients["w"] = self.inputs.T @ grad
        return grad @ self.parameters["w"].T

# Implement Activation Functions

Func = Callable[[Vector], Vector]

class Activation(Layer):
    def __init__(self, func: Func, func_prime: Func) -> None:
        super().__init__()
        self.func = func
        self.func_prime = func_prime
    
    def forward_pass(self, inputs: Vector) -> Vector:
        self.inputs = inputs
        return self.func(inputs)
    
    def backward_pass(self, grad: Vector) -> Vector:
        return self.func_prime(self.inputs) * grad

class ReLU(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)

# from numpy import tanh
class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)
