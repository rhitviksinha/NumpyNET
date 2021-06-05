""" Error functions and their respective derivatives
"""
import numpy as np
from numpy import ndarray as Vector

class Error:
    def error(self, y_pred: Vector, y: Vector) -> float:
        raise NotImplementedError
    def derivative(self, y_pred: Vector, y: Vector) -> float:
        raise NotImplementedError

class MSE(Error):
    """ Mean Squared Error
    """
    def error(self, y_pred: Vector, y: Vector) -> float:
        err = np.sum((y-y_pred) @ (y-y_pred).T / y.shape[0])
        return err

    def derivative(self, y_pred: Vector, y: Vector) -> float:
        # grad = 2*(y_pred-y) / y.shape[0]
        grad = 2*(y_pred-y)
        return grad

class BinaryCrossEntropy(Error):
    """ Binary Cross Entropy
    """
    def error(self, y_pred: Vector, y: Vector) -> float:
        err = np.sum( ((y-1.0)*np.log(1.0-y_pred) - y*np.log(y_pred)) / y.shape[0] )
        return err

    def derivative(self, y_pred: Vector, y: Vector) -> float:
        # grad = (((1-y) / (1-y_pred)) - (y / y_pred)) / y.shape[0]
        grad = (((1-y) / (1-y_pred)) - (y / y_pred))
        return grad

class MAE(Error):
    """ Mean Absolute Error
    """
    def error(self, y_pred: Vector, y: Vector) -> float:
        err = np.sum( np.abs((y-y_pred)) / y.shape[0] )
        return err

    def derivative(self, y_pred: Vector, y: Vector) -> float:
        # grad = np.sign(y_pred-y) / y.shape[0]
        grad = np.sign(y_pred-y)
        return grad
