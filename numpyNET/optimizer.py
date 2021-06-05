""" Adjust parameters using computed gradients
"""
from numpyNET.nn import Model

class Optimizer:
    def step(self, model: Model) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learn_rate: float = 0.01) -> None:
        self.learn_rate = learn_rate
    
    def step(self, model: Model) -> None:
        for param, grad in model.params_and_grads():
            param -= grad * self.learn_rate

class Adam(Optimizer):
    def __init__(self, learn_rate: float = 0.01) -> None:
        raise NotImplementedError
