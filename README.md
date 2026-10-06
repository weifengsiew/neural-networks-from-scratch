# Neural Networks From Scratch

This repository contains a NumPy feed-forward neural network and a notebook that applies it to the Parkinson's disease dataset.

## What is a neural network?

A neural network is a sequence of layers that transforms inputs into predictions. Each neuron computes a weighted sum of its inputs, adds a bias, and applies an activation function:

```text
activation = sigmoid(weights · inputs + bias)
```

The weights and biases are the model's parameters. They determine how strongly each input contributes to the prediction. Hidden layers combine simple transformations into representations that can model nonlinear relationships.

## From inputs to outputs: forward propagation

For one instance, the feature vector is first augmented with a bias term. Each layer then performs two operations:

1. Multiply the incoming activations by the layer's weight matrix and add the bias weights.
2. Apply the sigmoid activation function to produce the next layer's activations.

The output of one layer becomes the input to the next. The final layer produces a probability for binary classification or one probability per class for multiclass classification. This sequence is called forward propagation.

## Comparing predictions with actual values: loss

The network's output is compared with the target value or target class. For binary classification, the implementation uses binary cross-entropy:

```text
loss = -[y log(p) + (1 - y) log(1 - p)]
```

Here, `y` is the actual class (`0` or `1`) and `p` is the predicted probability of class `1`. Confident correct predictions have low loss; confident incorrect predictions have high loss. For multiple classes, the targets are one-hot encoded and the same output-layer formulation is applied across the class outputs. The training objective is to minimize the average loss over the training instances, optionally with an L2 penalty on non-bias weights.

## Backpropagation and gradients

Backpropagation computes how much each parameter contributed to the loss. Starting at the output layer, the network calculates an error signal, or delta, from the difference between the prediction and the target. It then moves backward through the hidden layers, applying the chain rule and each activation function's derivative.

For a connection, the gradient is the outer product of:

- the delta of the receiving neuron, and
- the activation that entered the connection.

The result has the same shape as the weight matrix. A gradient describes how the loss changes when each weight changes: a positive value means increasing that weight would increase the loss locally, while a negative value means it would decrease the loss locally.

## Weight updates

After gradients have been accumulated for a batch, the implementation averages them, adds the regularization gradient when configured, and updates each weight with gradient descent:

```text
new_weight = old_weight - step_size × gradient
```

The step size, also called the learning rate, controls how far each update moves. Subtracting the gradient moves the parameters in the direction that locally reduces the loss. Bias weights are excluded from L2 regularization.

## Instances, batches, epochs, and weight updates

- An **instance** is one training example: one feature vector and its target.
- A **batch** is a group of instances processed before the weights are updated. With batch size `32`, the network accumulates gradients from up to 32 instances, averages them, and performs one weight update.
- An **epoch** is one complete pass through the training set. If there are `N` instances and a batch size of `B`, one epoch produces approximately `ceil(N / B)` weight updates.
- A **weight update** is one application of the gradient-descent formula to every weight matrix.

For example, 156 training instances with a batch size of 32 produce five updates per epoch: four full batches of 32 and one final batch of 28. After 100 epochs, the network has processed each instance 100 times and performed 500 updates. Shuffling changes which instances share a batch, but not the number of instances processed in an epoch.

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
