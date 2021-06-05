""" Design and train the neural network. Also,
    Test the Neural Net against previously unseen data.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray as Vector
from numpyNET.nn import Model
from numpyNET.error import Error, MSE
from numpyNET.optimizer import Optimizer, SGD
from numpyNET.data_loader import DataIterator, BatchIterator

def train(model: Model,
          features: Vector,
          labels: Vector,
          plotting: bool = False,
          epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          err: Error = MSE(),
          optimizer: Optimizer = SGD()) -> Vector:
    history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in iterator(features, labels):
            predicted = model.feed_forward(batch.features)
            epoch_loss += err.error(predicted, batch.labels)
            grad = err.derivative(predicted, batch.labels)
            model.back_propagate(grad)
            optimizer.step(model)
        if epoch % (epochs // 10) == 0:
            print(epoch, epoch_loss)
        history.append(np.array([epoch, epoch_loss]))
    history = np.array(history)

    if plotting:
        plt.figure(figsize=(8,8), dpi=96)
        plt.plot(history[:, 0], history[:, 1], 'r')
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss")
        plt.show()
    
    return history

def predict(model: Model, features: Vector) -> Vector:
    return model.feed_forward(features)
