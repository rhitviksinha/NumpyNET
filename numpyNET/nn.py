""" Setting up an Artificial Neural Network
"""
import numpy as np
from typing import Iterator, Sequence, Tuple
from numpyNET.layers import Layer
from numpy import ndarray as Vector

class Model:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def feed_forward(self, inputs: Vector) -> Vector:
        for layer in self.layers:
            inputs = layer.forward_pass(inputs)
        return inputs

    def back_propagate(self, grad: Vector) -> Vector:
        for layer in reversed(self.layers):
            grad = layer.backward_pass(grad)
        return grad
    
    def params_and_grads(self) -> Iterator[Tuple[Vector, Vector]]:
        for layer in self.layers:
            for name, param in layer.parameters.items():
                grad = layer.gradients[name]
                yield param, grad
