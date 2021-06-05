# NumpyNET

A self-designed neural network, written entirely in Python and the NumPy library. Heavily inspired by [Joel Grus' `joelnet`](https://github.com/joelgrus/joelnet/). Check the .ipynb notebooks out for implementation.
* [testing_real_data](./testing_real_data.ipynb) has a neural network successfully implemented on the [Banknote Authentication Data Set](https://archive.ics.uci.edu/ml/datasets/banknote+authentication).
* [testing_custom_data](./testing_real_data.ipynb) contains a neural network implemented on the `XOR` logic function. And a custom dataset with an `XOR`-like distribution.

## Repository Distribution

    .
    ├── dataset/
    |   └── data_banknote_authentication.txt
    ├── numpyNET/
    |   ├── activation.py
    |   ├── data_loader.py
    |   ├── error.py
    |   ├── layers.py
    |   ├── nn.py
    |   ├── optimizer.py
    |   └── train.py
    ├── README.md
    ├── requirements.txt
    ├── testing_custom_data.ipynb
    └── testing_real_data.ipynb
---
## Usage

```py
from numpyNET.nn import Model
from numpyNET.layers import Dense, Sigmoid
from numpyNET.optimizer import SGD
from numpyNET.error import Error, MSE, BinaryCrossEntropy
from numpyNET.data_loader import BatchIterator
from numpyNET.train import train, predict

# Designing a Model
model = Model([
    Dense(input_size=..., output_size=...),
    Sigmoid(),
    Dense(input_size=..., output_size=...),
    Sigmoid()
])

# Model Hyperparameters
num_epochs = ...
optim = SGD(learn_rate=...)
batch_size = ...
err = MSE()

# Training
train_features = ...
train_labels = ...
history = train(
    model, train_features, train_labels, plotting=True,
    epochs=num_epochs, optimizer=optim, err=err,
    iterator=BatchIterator(batch_size=batch_size)
)

# Predictions
unknown_features = ...
prediction = predict(model, unknown_features)
```

---
## Layers Implemented in [layers.py](./numpyNET/layers.py)

### Dense

The classic linear layer that takes an `input` matrix and returns its matrix product with a `weight` matrix and a `bias` term added. Each neuron in the `Dense` layer receives `input` from all neurons of its previous layer. *Activations are implemented as separate layers.*

### ReLU *(Activation Layer)*

The rectified linear activation function, or `ReLU` activation function, is perhaps the most common function used for hidden layers. In a `ReLU` unit, if the `input` value is negative, then a value `0.0` is obtained, otherwise, the `input` is returned.

### Sigmoid *(Activation Layer)*

Another classic, it applies a `Sigmoid` function to the `input` such that the output is bounded in the interval `(0, 1)`. The larger the `input` (more positive), the closer the output value will be to `1.0`, whereas the smaller the `input` (more negative), the closer the output will be to `0.0`.

### Tanh *(Activation Layer)*

This layer applies a `Tanh` function to the `input`. The output is bounded in the interval `(-1, 1)`. The larger the `input` (more positive), the closer the output value will be to `1.0`, whereas the smaller the `input` (more negative), the closer the output will be to `-1.0`.

---
## Error Functions Implemented in [error.py](./numpyNET/error.py)

### Binary Cross Entropy Error

*Incomplete.* Might have some issues, as the gradient either explodes, or vanishes to `nan`.

### Mean Absolute Error

*Untested.*

### Mean Squared Error

Implmented. Well tested.

---
## Optimizers Implemented in [optimzer.py](./numpyNET/optimizer.py)

### Stochastic Gradient Descent

The batch sizes are implemented in [data_loader.py](./numoyNET/data_loader.py). Will work on implementing Momentum / Adagrad / Adam optimizers.

