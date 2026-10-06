# Neural Networks From Scratch

This repository contains a NumPy feed-forward neural network and a notebook that applies it to the Parkinson's disease dataset.

## What is a neural network?

A neural network is a sequence of layers that transforms inputs into predictions. Each neuron computes a weighted sum of its inputs, adds a bias, and applies an activation function:

```text
activation = sigmoid(weights · inputs + bias)
```

The weights and biases are the model's parameters. They determine how strongly each input contributes to the prediction. Hidden layers combine simple transformations into representations that can model nonlinear relationships.

The implementation stores one weight matrix, called `theta`, for each connection between layers. It initializes those matrices with reproducible random values:

```python
thetas = initialize_thetas(
    num_input_neurons=X_train.shape[1],
    num_hidden_layers=self.num_hidden_layers,
    num_neurons_per_hidden_layer=self.num_neurons_per_hidden_layer,
    num_output_neurons=y_train.shape[1],
    random_seed=random_seed,
)
```

## From inputs to outputs: forward propagation

For one instance, the feature vector is first augmented with a bias term. Each layer then performs two operations:

1. Multiply the incoming activations by the layer's weight matrix and add the bias weights.
2. Apply the sigmoid activation function to produce the next layer's activations.

The output of one layer becomes the input to the next. The final layer produces a probability for binary classification or one probability per class for multiclass classification. This sequence is called forward propagation.

In `NeuralNetwork._hidden_layer`, the matrix multiplication, bias handling, and activation are explicit:

```python
preactivation = theta @ input
activation = sigmoid(preactivation)
activation = prepend_bias_term(activation)
```

The complete `_forward_propagate` method saves the intermediate activations because backpropagation needs them later:

```python
input_layer_activation = self._input_layer(x)
layers_activations = [input_layer_activation]

hidden_layers_preactivations, hidden_layers_activations, final_hidden_activation = (
    self._hidden_layers(input_layer_activation)
)
layers_activations.extend(hidden_layers_activations)

output_layer_preactivation, y_pred = self._output_layer(final_hidden_activation)
layers_activations.append(y_pred)
```

## Comparing predictions with actual values: loss

The network's output is compared with the target value or target class. For binary classification, the implementation uses binary cross-entropy:

```text
loss = -[y log(p) + (1 - y) log(1 - p)]
```

Here, `y` is the actual class (`0` or `1`) and `p` is the predicted probability of class `1`. Confident correct predictions have low loss; confident incorrect predictions have high loss. For multiple classes, the targets are one-hot encoded and the same output-layer formulation is applied across the class outputs. The training objective is to minimize the average loss over the training instances, optionally with an L2 penalty on non-bias weights.

The corresponding implementation for one instance is:

```python
def compute_cost(y_pred, y_true):
    return -np.sum(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )
```

Across a batch, these individual losses are averaged. When regularization is enabled, the squared non-bias weights are added to the average cost:

```python
regularization = (regularization_strength / (2 * num_instances)) * (
    sum(np.sum(theta[:, 1:] ** 2) for theta in thetas)
)
regularised_cost = total_cost / num_instances + regularization
```

## Backpropagation and gradients

Backpropagation computes how much each parameter contributed to the loss. Starting at the output layer, the network calculates an error signal, or delta, from the difference between the prediction and the target. It then moves backward through the hidden layers, applying the chain rule and each activation function's derivative.

For a connection, the gradient is the outer product of:

- the delta of the receiving neuron, and
- the activation that entered the connection.

The result has the same shape as the weight matrix. A gradient describes how the loss changes when each weight changes: a positive value means increasing that weight would increase the loss locally, while a negative value means it would decrease the loss locally.

The implementation starts with the output error and propagates it backward through the hidden layers:

```python
output_layer_delta = y_pred - y_true
deltas = [output_layer_delta]

for theta_next, activation_current in zip(
    reversed(thetas[1:]), reversed(layers_activations[1:-1])
):
    delta_next = deltas[0]
    delta_current = (
        (theta_next.T @ delta_next)
        * activation_current
        * (1 - activation_current)
    )
    delta_current = delta_current[1:]
    deltas.insert(0, delta_current)
```

Once the deltas are known, each weight gradient is formed from the receiving layer's delta and the previous layer's activation:

```python
for delta_next, activation_current in zip(deltas, activations[:-1]):
    gradient_current = np.outer(delta_next, activation_current)
    gradients.append(gradient_current)
```

## Weight updates

After gradients have been accumulated for a batch, the implementation averages them, adds the regularization gradient when configured, and updates each weight with gradient descent:

```text
new_weight = old_weight - step_size × gradient
```

The step size, also called the learning rate, controls how far each update moves. Subtracting the gradient moves the parameters in the direction that locally reduces the loss. Bias weights are excluded from L2 regularization.

The update itself is implemented directly as:

```python
for theta, gradient in zip(self.thetas, gradients):
    updated_theta = theta - step_size * gradient
    updated_thetas.append(updated_theta)

self.thetas = updated_thetas
```

## Instances, batches, epochs, and weight updates

- An **instance** is one training example: one feature vector and its target.
- A **batch** is a group of instances processed before the weights are updated. With batch size `32`, the network accumulates gradients from up to 32 instances, averages them, and performs one weight update.
- An **epoch** is one complete pass through the training set. If there are `N` instances and a batch size of `B`, one epoch produces approximately `ceil(N / B)` weight updates.
- A **weight update** is one application of the gradient-descent formula to every weight matrix.

For example, 156 training instances with a batch size of 32 produce five updates per epoch: four full batches of 32 and one final batch of 28. After 100 epochs, the network has processed each instance 100 times and performed 500 updates. Shuffling changes which instances share a batch, but not the number of instances processed in an epoch.

The training loop shows the complete relationship:

```python
for iteration in range(num_iterations):       # epochs
    batches = create_batches(X_train, y_train, batch_size, ...)

    for X_batch, y_batch in batches:            # one batch
        for x_i, y_i in zip(X_batch, y_batch):  # individual instances
            ...
            gradient_totals = accumulate_gradients(
                gradient_totals, gradients
            )

        final_gradients = regularize_and_average_gradients(
            self.thetas, gradient_totals, self.regularization_strength,
            len(X_batch),
        )
        self._update_weights(final_gradients, self.step_size)
```

The inner instance loop accumulates information; the weight update occurs only after the whole batch has been processed.

## Project layout

```text
neural_network.py          neural-network implementation
parkinsons_demo.ipynb      training and evaluation demonstration
data/                      fixed Parkinson's train/test split
```

## Setup

```bash
python -m pip install -r requirements.txt
```

Open [`parkinsons_demo.ipynb`](parkinsons_demo.ipynb) and run all cells. The notebook standardizes features using training data only, trains the network, and reports test accuracy and F1 score.
