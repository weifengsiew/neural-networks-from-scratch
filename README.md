# Neural Networks From Scratch

An educational feed-forward neural network implemented with NumPy. The code keeps the main mechanics visible: sigmoid activations, bias terms, forward propagation, backpropagation, mini-batch gradient descent, L2 regularization, label encoding, and classification metrics.

## Project layout

```text
algorithms/neural_network.py   neural-network implementation
tests/test_neural_network.py   focused implementation tests
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Example

```python
import numpy as np

from algorithms.neural_network import NeuralNetwork

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([0, 1, 1, 0])

network = NeuralNetwork(num_hidden_layers=1, num_neurons_per_hidden_layer=4)
network.configure_for_fit(regularization_strength=0.0, step_size=1.0)
network.fit(
    X,
    y,
    num_iterations=1000,
    batch_size=4,
    random_seed=7,
    shuffle=True,
    record_history=False,
)
predictions = network.predict_label(X)
```

Run the test suite with:

```bash
pytest -q
```
