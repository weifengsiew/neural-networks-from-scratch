# Neural Networks From Scratch

This repository contains feed-forward neural network implemented from scratch and a notebook that applies it to the Parkinson's disease dataset. The sections below explain the theory underlying the feed-forward neural network implementation in [`neural_network.py`](neural_network.py).

## 1. What is a neural network?

A neural network is a composition of parameterized functions. Each layer receives an activation vector from the previous layer, computes an affine transformation—a weighted sum of the activations plus a bias—and then applies a nonlinear activation function to the result.

For layer $l$, let:

- $\mathbf{a}^{(l-1)}$ be the incoming activations;
- $\Theta^{(l)}$ be the layer's weight matrix, including bias weights;
- $\mathbf{z}^{(l)}$ be the preactivation values; and
- $\mathbf{a}^{(l)}$ be the output activations.

The layer computes:

$$
\mathbf{z}^{(l)} = \Theta^{(l)}\mathbf{a}^{(l-1)},
\qquad
\mathbf{a}^{(l)} = \sigma\left(\mathbf{z}^{(l)}\right),
$$

where the sigmoid activation is:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

The weights and biases are the parameters learned from data. The implementation uses a bias convention in which a leading `1.0` is inserted into every input activation. The first column of each weight matrix therefore contains the bias weights.

~~~python
def sigmoid(preactivation: np.ndarray) -> np.ndarray:
    # Convert each preactivation z into a value between 0 and 1.
    return 1 / (1 + np.exp(-preactivation))


def prepend_bias_term(activation: np.ndarray) -> np.ndarray:
    # The leading 1 lets the first column of theta act as a bias vector.
    return np.insert(activation, 0, 1.0)
~~~

## 2. Forward propagation: inputs become outputs

Forward propagation evaluates the network from left to right. For one instance, the input features are augmented with a bias term, passed through every hidden layer, and finally transformed into an output probability.

For a hidden layer, the equations are:

$$
\mathbf{z}^{(l)} = \Theta^{(l)}\mathbf{a}^{(l-1)},
\qquad
\mathbf{a}^{(l)} = \begin{bmatrix}1 \\ \sigma(\mathbf{z}^{(l)})\end{bmatrix}.
$$

The leading $1$ in the second equation is the bias activation passed to the next layer. This is the code for one hidden layer:

~~~python
def _hidden_layer(self, theta: np.ndarray, input: np.ndarray):
    # theta contains one row per receiving neuron and one column per input,
    # including column 0 for the bias.
    preactivation = theta @ input

    # Apply the nonlinear function element by element:
    # activation_without_bias = sigmoid(preactivation).
    activation = sigmoid(preactivation)

    # Add the leading 1 so the next layer can include its bias weights.
    activation = prepend_bias_term(activation)
    return preactivation, activation
~~~

The complete forward pass stores all intermediate activations. They are needed later because backpropagation applies the chain rule through each layer.

~~~python
def _forward_propagate(self, x: np.ndarray):
    # a^(0): input features with the bias activation prepended.
    input_layer_activation = self._input_layer(x)
    layers_activations = [input_layer_activation]
    layers_preactivations = []

    # Compute and save every hidden layer's z^(l) and a^(l).
    hidden_preactivations, hidden_activations, final_hidden_activation = (
        self._hidden_layers(input_layer_activation)
    )
    layers_preactivations.extend(hidden_preactivations)
    layers_activations.extend(hidden_activations)

    # The final sigmoid output is the prediction p.
    output_preactivation, y_pred = self._output_layer(final_hidden_activation)
    layers_preactivations.append(output_preactivation)
    layers_activations.append(y_pred)

    return layers_preactivations, layers_activations, y_pred
~~~

For binary classification, the final output $p$ is interpreted as the predicted probability that $y=1$. The class prediction is $1$ when $p > 0.5$, and $0$ otherwise.

## 3. Loss: comparing predictions with actual values

The network learns by comparing its prediction with the known target. For binary classification, the implementation uses binary cross-entropy:

$$
\mathcal{L}(p,y)
= -\left[y\log(p) + (1-y)\log(1-p)\right],
$$

where $y\in\{0,1\}$ is the actual class and $p\in(0,1)$ is the predicted probability. A confident correct prediction has low loss; a confident incorrect prediction has high loss.

~~~python
def compute_cost(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    # y_pred is p and y_true is y in the binary cross-entropy equation.
    cost = -np.sum(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )
    return cost
~~~

For a batch $\mathcal{B}$ containing $m$ instances, the data loss is averaged:

$$
J_{\text{data}} = \frac{1}{m}\sum_{i\in\mathcal{B}} \mathcal{L}(p_i,y_i).
$$

The implementation can add L2 regularization to discourage large non-bias weights:

$$
J = J_{\text{data}}
+ \frac{\lambda}{2m}\sum_l\sum_{j,k>0}
\left(\Theta^{(l)}_{jk}\right)^2.
$$

The condition $k>0$ excludes the first, bias-weight column.

~~~python
def regularize_and_average_cost(thetas, total_cost, regularization_strength, num_instances):
    # theta[:, 1:] excludes each layer's bias-weight column.
    regularization = (regularization_strength / (2 * num_instances)) * (
        sum(np.sum(theta[:, 1:] ** 2) for theta in thetas)
    )

    # Combine the average data loss with the L2 penalty.
    return total_cost / num_instances + regularization
~~~

## 4. Backpropagation: computing the error signals

Backpropagation computes the derivative of the loss with respect to every layer's preactivation. Define the delta for layer $l$ as:

$$
\boldsymbol{\delta}^{(l)}
= \frac{\partial J}{\partial \mathbf{z}^{(l)}}.
$$

For a sigmoid output layer with binary cross-entropy, the output delta simplifies to:

$$
\boldsymbol{\delta}^{(L)} = \mathbf{p} - \mathbf{y}.
$$

For a hidden layer, the chain rule gives:

$$
\boldsymbol{\delta}^{(l)}
= \left((\Theta^{(l+1)})^T
\boldsymbol{\delta}^{(l+1)}\right)
\odot \sigma'\left(\mathbf{z}^{(l)}\right),
$$

and the sigmoid derivative is:

$$
\sigma'(z) = \sigma(z)(1-\sigma(z)).
$$

The code applies this equation from the output layer backward. `delta_current[1:]` removes the bias position because bias activations are not neurons that need a previous-layer error signal.

~~~python
def compute_deltas(thetas, y_pred, y_true, layers_activations):
    # For sigmoid plus binary cross-entropy: delta^(L) = p - y.
    output_layer_delta = y_pred - y_true
    deltas = [output_layer_delta]

    # Walk backward through the hidden layers.
    for theta_next, activation_current in zip(
        reversed(thetas[1:]), reversed(layers_activations[1:-1])
    ):
        delta_next = deltas[0]

        # theta_next.T moves the next-layer error back to this layer.
        # activation_current * (1 - activation_current) is sigma'(z).
        delta_current = (
            (theta_next.T @ delta_next)
            * activation_current
            * (1 - activation_current)
        )

        # The first entry corresponds to the bias activation; remove it.
        delta_current = delta_current[1:]
        deltas.insert(0, delta_current)

    return deltas
~~~

## 5. Gradients: how deltas become weight derivatives

For a weight connecting activation $a_k^{(l-1)}$ to neuron $j$ in layer $l$, the chain rule gives:

$$
\frac{\partial J}{\partial \Theta^{(l)}_{jk}}
= \delta^{(l)}_j a^{(l-1)}_k.
$$

In matrix form, the gradient for a layer is an outer product:

$$
\nabla_{\Theta^{(l)}}J
= \boldsymbol{\delta}^{(l)}
\left(\mathbf{a}^{(l-1)}\right)^T.
$$

The implementation uses `np.outer` to create exactly that matrix:

~~~python
def compute_gradients(activations, deltas):
    gradients = []

    # activations[:-1] supplies a^(l-1); deltas supplies delta^(l).
    for delta_next, activation_current in zip(deltas, activations[:-1]):
        # Each entry is delta_j * activation_k.
        gradient_current = np.outer(delta_next, activation_current)
        gradients.append(gradient_current)

    return gradients
~~~

For a batch, the per-instance gradients are summed and then divided by the batch size. The regularization gradient $\lambda\Theta/m$ is added to non-bias weights before the update.

## 6. Weight updates: gradient descent

Once the gradient is known, gradient descent changes every parameter according to:

$$
\Theta^{(l)}_{\text{new}}
= \Theta^{(l)}_{\text{old}}
- \eta\nabla_{\Theta^{(l)}}J,
$$

where $\eta$ is the step size, or learning rate. The minus sign moves the parameters opposite to the direction in which the loss increases.

~~~python
def _update_weights(self, gradients, step_size):
    updated_thetas = []

    for theta, gradient in zip(self.thetas, gradients):
        # Move each weight opposite its loss gradient.
        updated_theta = theta - step_size * gradient
        updated_thetas.append(updated_theta)

    # The new matrices become the parameters used by the next batch.
    self.thetas = updated_thetas
~~~

## 7. Instances, batches, epochs, and updates

- An **instance** is one feature vector and its target $(\mathbf{x}_i,y_i)$.
- A **batch** is a group of instances processed before one update.
- An **epoch** is one complete pass through all training instances.
- A **weight update** changes every weight matrix once.

If there are $N$ training instances and the batch size is $B$, one epoch contains:

$$
\text{updates per epoch} = \left\lceil\frac{N}{B}\right\rceil.
$$

For the demonstration, $N=156$ and $B=32$, so each epoch has $\lceil156/32\rceil=5$ updates: four batches of 32 and one final batch of 28. One instance contributes a gradient; the batch combines those gradients; the update changes the weights; the next epoch repeats the process with the changed weights.

~~~python
for iteration in range(num_iterations):       # One iteration is one epoch.
    batches = create_batches(X_train, y_train, batch_size, ...)

    for X_batch, y_batch in batches:            # One weight update per batch.
        gradient_totals = [np.zeros_like(theta) for theta in self.thetas]

        for x_i, y_i in zip(X_batch, y_batch):  # Process each instance.
            _, activations, y_pred_i = self._forward_propagate(x_i)
            deltas = compute_deltas(self.thetas, y_pred_i, y_i, activations)
            gradients = compute_gradients(activations, deltas)
            gradient_totals = accumulate_gradients(gradient_totals, gradients)

        # Average the batch's gradients, then update all weight matrices once.
        final_gradients = regularize_and_average_gradients(
            self.thetas,
            gradient_totals,
            self.regularization_strength,
            len(X_batch),
        )
        self._update_weights(final_gradients, self.step_size)
~~~

Shuffling changes which instances share a batch, but not the number of instances processed per epoch. Increasing the batch size usually means fewer updates per epoch; increasing the number of epochs means more complete passes over the same training data.

## Project layout

~~~text
neural_network.py          neural-network implementation
parkinsons_demo.ipynb      training and evaluation demonstration
data/                      fixed Parkinson's train/test split
~~~

## Setup

~~~bash
python -m pip install -r requirements.txt
~~~

Open [`parkinsons_demo.ipynb`](parkinsons_demo.ipynb) and run all cells. The notebook standardizes features using training data only, trains the network, and reports test accuracy and F1 score.
