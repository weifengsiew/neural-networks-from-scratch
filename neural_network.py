"""Functions and stateful model implementation for a feed-forward neural network."""

from __future__ import annotations

from typing import Any

import numpy as np


def sigmoid(preactivation: np.ndarray) -> np.ndarray:
    """Compute activation using sigmoid.

    Args:
        preactivation (np.ndarray): Weighted sum of input values for a neural network layer.

    Returns:
        activation (np.ndarray): Sigmoid-transformed weighted sum.
    """
    activation = 1 / (1 + np.exp(-preactivation))
    return activation


def prepend_bias_term(activation: np.ndarray) -> np.ndarray:
    """Prepend bias term to activation.

    Args:
        activation (np.ndarray): Sigmoid-transformed weighted sum of input values.

    Returns:
        activation_w_bias_term (np.ndarray): Bias term (1) prepended to activation.
    """
    activation_w_bias_term = np.insert(activation, 0, 1.0)
    return activation_w_bias_term


def compute_cost(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute binary cross-entropy loss for one training instance.

    Args:
        y_pred (np.ndarray): Predicted output values.
        y_true (np.ndarray): Expected output values.

    Returns:
        cost (float): Binary cross-entropy loss for one training instance.
    """
    cost = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return cost


def compute_deltas(
    thetas: list[np.ndarray],
    y_pred: np.ndarray,
    y_true: np.ndarray,
    layers_activations: list[np.ndarray],
) -> list[np.ndarray]:
    """Compute delta values.

    Args:
        thetas (list): Weights for each network layer.
        y_pred (np.ndarray): Predicted output values.
        y_true (np.ndarray): Expected output values.
        layers_activations (list): Activations from each layer.

    Returns:
        deltas (list): Delta values for hidden and output layers. Delta value of a neuron measures how much its output contributed to the network error.
    """
    output_layer_delta = y_pred - y_true
    deltas = [output_layer_delta]

    for theta_next, activation_current in zip(
        reversed(thetas[1:]), reversed(layers_activations[1:-1])
    ):
        delta_next = deltas[0]
        delta_current = (
            (theta_next.T @ delta_next) * activation_current * (1 - activation_current)
        )
        delta_current = delta_current[1:]
        deltas.insert(0, delta_current)

    return deltas


def compute_gradients(
    activations: list[np.ndarray], deltas: list[np.ndarray]
) -> list[np.ndarray]:
    """Compute gradients for one training instance.

    Args:
        activations (list): Activations from each layer.
        deltas (list): Delta values for hidden and output layers.

    Returns:
        gradients (list): Gradients for each weight based on one training instance.
    """
    gradients = []

    for delta_next, activation_current in zip(deltas, activations[:-1]):
        gradient_current = np.outer(delta_next, activation_current)
        gradients.append(gradient_current)

    return gradients


def accumulate_gradients(
    gradient_totals: list[np.ndarray], gradients: list[np.ndarray]
) -> list[np.ndarray]:
    """Add gradients from one training instance to gradient totals.

    Args:
        gradient_totals (list): Sum of gradients for each weight across training instances processed so far.
        gradients (list): Gradients for each weight based on one training instance.

    Returns:
        gradient_totals (list): Sum of gradients for each weight after adding gradients from the current training instance.
    """
    for gradient_total, gradient in zip(gradient_totals, gradients):
        gradient_total += gradient

    return gradient_totals


def regularize_and_average_gradients(
    thetas: list[np.ndarray],
    gradient_totals: list[np.ndarray],
    regularization_strength: float,
    num_instances: int,
) -> list[np.ndarray]:
    """Compute L2-regularized gradients averaged across training instances.

    Args:
        thetas (list): Weights for each network layer.
        gradient_totals (list): Sum of gradients for each weight across all training instances.
        regularization_strength (float): Strength of L2 regularization for non-bias weights.
        num_instances (int): Number of training instances.

    Returns:
        regularized_gradients (list): L2-regularized gradients averaged across training instances.
    """
    regularized_gradients = []

    for theta, gradient_total in zip(thetas, gradient_totals):
        regularization = regularization_strength * theta
        regularization[:, 0] = 0.0
        regularized_gradient = (gradient_total + regularization) / num_instances
        regularized_gradients.append(regularized_gradient)

    return regularized_gradients


def regularize_and_average_cost(
    thetas: list[np.ndarray],
    total_cost: float,
    regularization_strength: float,
    num_instances: int,
) -> float:
    """Compute average L2-regularized binary cross-entropy loss.

    Args:
        thetas (list): Weights for each network layer, including bias weights.
        total_cost (float): Sum of costs across all training instances.
        regularization_strength (float): Strength of L2 regularization for non-bias weights.
        num_instances (int): Number of training instances.

    Returns:
        regularised_cost (float): Average L2-regularized binary cross-entropy loss.
    """
    regularization = (regularization_strength / (2 * num_instances)) * (
        sum(np.sum(theta[:, 1:] ** 2) for theta in thetas)
    )
    regularised_cost = total_cost / num_instances + regularization
    return regularised_cost


def compute_average_regularized_cost_over_instances(
    network: NeuralNetwork, X: np.ndarray, y: np.ndarray, regularization_strength: float
) -> float:
    """Compute average L2-regularized binary cross-entropy loss over instances.

    Args:
        network (NeuralNetwork): Neural network.
        X (list): Input values.
        y (list): Expected output values.
        regularization_strength (float): Strength of L2 regularization for non-bias weights.

    Returns:
        cost (float): Average L2-regularized binary cross-entropy loss.
    """
    total_cost = 0.0

    for x_i, y_i in zip(X, y):
        _, _, y_pred_i = network._forward_propagate(x_i)
        cost = compute_cost(y_pred_i, y_i)
        total_cost += cost

    cost = regularize_and_average_cost(
        network.thetas, total_cost, regularization_strength, len(X)
    )
    return cost


def compute_binary_classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, positive_label: Any
) -> tuple[float, float, float, float]:
    """Compute classification metrics for one positive label.

    Args:
        labels (np.ndarray): True labels.
        predictions (np.ndarray): Predicted labels.
        positive_label: Label to treat as the positive class.

    Returns:
        accuracy (float): Fraction of predictions that match true labels.
        precision (float): Fraction of predicted positive labels that are correct.
        recall (float): Fraction of true positive labels that are recovered.
        f1 (float): Harmonic mean of precision and recall.
    """
    tp = np.sum((labels == positive_label) & (predictions == positive_label))
    tn = np.sum((labels != positive_label) & (predictions != positive_label))
    fp = np.sum((labels != positive_label) & (predictions == positive_label))
    fn = np.sum((labels == positive_label) & (predictions != positive_label))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return accuracy, precision, recall, f1


def compute_multiclass_classification_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> tuple[float, float, float, float]:
    """Compute accuracy and macroaveraged classification metrics.

    Args:
        labels (np.ndarray): True labels.
        predictions (np.ndarray): Predicted labels.

    Returns:
        accuracy (float): Fraction of predictions that match true labels.
        precision (float): Macroaveraged precision.
        recall (float): Macroaveraged recall.
        f1 (float): Macroaveraged F1 score.
    """
    labels = np.array(labels)
    predictions = np.array(predictions)
    unique_labels = np.unique(labels)

    accuracy = np.sum(labels == predictions) / max(len(labels), 1)
    precisions = []
    recalls = []
    f1_scores = []

    for label in unique_labels:
        _, precision, recall, f1 = compute_binary_classification_metrics(
            labels,
            predictions,
            positive_label=label,
        )
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    precision = np.mean(precisions)
    recall = np.mean(recalls)
    f1 = np.mean(f1_scores)

    return accuracy, precision, recall, f1


def compute_metrics_over_instances(
    network: NeuralNetwork, X: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    """Compute classification metrics over instances.

    Args:
        network (NeuralNetwork): Neural network.
        X (list): Input values.
        y (list): True class labels.

    Returns:
        accuracy (float): Fraction of predictions that match true class labels.
        f1 (float): Macroaveraged F1 score.
    """
    predictions = np.array(network.predict_label(X))
    accuracy, _, _, f1 = compute_multiclass_classification_metrics(y, predictions)
    return accuracy, f1


def create_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    random_generator: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create batches of training data.

    Args:
        X (list): Input values.
        y (list): True class labels, represented as binary labels or one-hot encoded vectors.
        batch_size (int): Number of training instances in each batch.
        shuffle (bool): Whether to shuffle training instances before creating batches.
        random_generator (np.random.Generator): Random number generator.

    Returns:
        batches (list): Training instances divided into batches.
    """
    if batch_size < 1:
        raise ValueError("batch_size should be at least 1")

    if batch_size > len(X):
        raise ValueError(
            "batch_size should not exceed the number of training instances"
        )

    X = np.array(X)
    y = np.array(y)

    if shuffle:
        indices = random_generator.permutation(len(X))
    else:
        indices = np.arange(len(X))

    batches = []

    for start_index in range(0, len(X), batch_size):
        batch_indices = indices[start_index : start_index + batch_size]
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        batches.append((X_batch, y_batch))

    return batches


def initialize_thetas(
    num_input_neurons: int,
    num_hidden_layers: int,
    num_neurons_per_hidden_layer: int,
    num_output_neurons: int,
    random_seed: int | None = None,
) -> list[np.ndarray]:
    """Initialize theta matrices using Gaussian distribution.

    Args:
        num_input_neurons (int): Number of input neurons, excluding bias term.
        num_hidden_layers (int): Number of hidden layers.
        num_neurons_per_hidden_layer (int): Number of neurons in each hidden layer, excluding bias term.
        num_output_neurons (int): Number of output neurons.
        random_seed (int): Random seed for reproducibility.

    Returns:
        thetas (list): Theta matrices initialized using Gaussian distribution with mean 0 and variance 1.
    """
    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers should be at least 1")

    if num_neurons_per_hidden_layer < 1:
        raise ValueError("num_neurons_per_hidden_layer should be at least 1")

    random_generator = np.random.default_rng(random_seed)
    thetas = []

    theta_input_to_hidden = random_generator.normal(
        0.0, 1.0, size=(num_neurons_per_hidden_layer, num_input_neurons + 1)
    )

    thetas.append(theta_input_to_hidden)

    for _ in range(num_hidden_layers - 1):
        theta_hidden_to_hidden = random_generator.normal(
            0.0,
            1.0,
            size=(num_neurons_per_hidden_layer, num_neurons_per_hidden_layer + 1),
        )

        thetas.append(theta_hidden_to_hidden)

    theta_hidden_to_output = random_generator.normal(
        0.0, 1.0, size=(num_output_neurons, num_neurons_per_hidden_layer + 1)
    )

    thetas.append(theta_hidden_to_output)

    return thetas


class NeuralNetwork:
    """Train a feed-forward neural network with gradient descent."""

    def __init__(
        self,
        thetas: list[np.ndarray] | None = None,
        num_hidden_layers: int | None = None,
        num_neurons_per_hidden_layer: int | None = None,
    ) -> None:
        """Initialize neural network.

        Args:
            thetas (list): Weights for each network layer, including bias weights.
            num_hidden_layers (int): Number of hidden layers.
            num_neurons_per_hidden_layer (int): Number of neurons in each hidden layer.
        """
        self.thetas = thetas
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_hidden_layer = num_neurons_per_hidden_layer
        self.regularization_strength = 0
        self.step_size = 0
        self.label_to_index = None
        self.index_to_label = None

    def _input_layer(self, x: np.ndarray) -> np.ndarray:
        """Compute output values from input layer.

        Args:
            x (np.ndarray): Input values for one instance.

        Returns:
            activation (np.ndarray): Output values from input layer, including bias term.
        """
        activation = prepend_bias_term(x)
        return activation

    def _hidden_layer(
        self, theta: np.ndarray, input: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute output values from one hidden layer.

        Args:
            theta (np.ndarray): Weights for this hidden layer, including bias weights.
            input (np.ndarray): Output values from the previous layer.

        Returns:
            preactivation (np.ndarray): Weighted sum of input values to this hidden layer.
            activation (np.ndarray): Output values from this hidden layer, including bias term.
        """
        preactivation = theta @ input
        activation = sigmoid(preactivation)
        activation = prepend_bias_term(activation)
        return preactivation, activation

    def _hidden_layers(
        self, input: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """Compute output values from all hidden layers.

        Args:
            input (np.ndarray): Output values from input layer.

        Returns:
            layers_preactivations (list): Weighted sums of input values to each hidden layer.
            layers_activations (list): Output values from each hidden layer, including bias term.
            final_hidden_activation (np.ndarray): Output values from final hidden layer, including bias term.
        """
        layers_preactivations = []
        layers_activations = []

        for theta in self.thetas[:-1]:
            preactivation, activation = self._hidden_layer(theta, input)
            input = activation

            layers_preactivations.append(preactivation)
            layers_activations.append(activation)

        final_hidden_activation = activation
        return layers_preactivations, layers_activations, final_hidden_activation

    def _output_layer(self, input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute output values from output layer.

        Args:
            input (np.ndarray): Output values from final hidden layer.

        Returns:
            preactivation (np.ndarray): Weighted sum of input values to output layer.
            activation (np.ndarray): Output values from output layer.
        """
        theta = self.thetas[-1]
        preactivation = theta @ input
        activation = sigmoid(preactivation)
        return preactivation, activation

    def _forward_propagate(
        self, x: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """Forward propagate one training instance.

        Args:
            x (np.ndarray): Input values to the network for one instance.

        Returns:
            layers_preactivations (list): Weighted sums of input values to hidden and output layers.
            layers_activations (list): Output values from input, hidden, and output layers.
            y_pred (np.ndarray): Output values from network.
        """
        input_layer_activation = self._input_layer(x)
        layers_activations = [input_layer_activation]
        layers_preactivations = []

        (
            hidden_layers_preactivations,
            hidden_layers_activations,
            final_hidden_activation,
        ) = self._hidden_layers(input_layer_activation)
        layers_preactivations.extend(hidden_layers_preactivations)
        layers_activations.extend(hidden_layers_activations)

        output_layer_preactivation, y_pred = self._output_layer(final_hidden_activation)
        layers_preactivations.append(output_layer_preactivation)
        layers_activations.append(y_pred)

        return layers_preactivations, layers_activations, y_pred

    def configure_for_fit(
        self, regularization_strength: float, step_size: float
    ) -> None:
        """Configure settings used during fitting.

        Args:
            regularization_strength (float): Strength of L2 regularization for non-bias weights.
            step_size (float): Step size used when updating weights.
        """
        self.regularization_strength = regularization_strength
        self.step_size = step_size

    def _update_weights(self, gradients: list[np.ndarray], step_size: float) -> None:
        """Update weights using gradients.

        Args:
            gradients (list): Gradients for weights in each network layer.
            step_size (float): Step size used when updating weights.
        """
        updated_thetas = []

        for theta, gradient in zip(self.thetas, gradients):
            updated_theta = theta - step_size * gradient
            updated_thetas.append(updated_theta)

        self.thetas = updated_thetas

    def _fit_label_mapping(self, y: np.ndarray) -> None:
        """Create mappings between original labels and neural network output indices.

        Args:
            y (np.ndarray): Original class labels.
        """
        labels = np.unique(y)
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.index_to_label = {index: label for index, label in enumerate(labels)}

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        """Encode original labels as neural network targets.

        Args:
            y (np.ndarray): Original class labels.

        Returns:
            y_encoded (np.ndarray): Binary class indices or one-hot multiclass targets.
        """
        label_indices = np.array([self.label_to_index[label] for label in y])

        if len(self.label_to_index) == 2:
            y_encoded = label_indices.reshape(-1, 1)
        else:
            y_encoded = np.eye(len(self.label_to_index))[label_indices]

        return y_encoded

    def _prepare_labels_for_fit(
        self, y_train: np.ndarray, y_test: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
        """Prepare labels for training while preserving labels for metrics.

        Args:
            y_train (np.ndarray): Training labels.
            y_test (np.ndarray): Test labels.

        Returns:
            y_train_encoded (np.ndarray): Training labels encoded for optimization.
            y_test_encoded (np.ndarray): Test labels encoded for cost computation.
            y_train_for_metrics (np.ndarray): Training labels used for metrics.
            y_test_for_metrics (np.ndarray): Test labels used for metrics.
        """
        y_train = np.array(y_train)
        y_test = None if y_test is None else np.array(y_test)

        if y_train.ndim == 1:
            self._fit_label_mapping(y_train)
            y_train_encoded = self._encode_labels(y_train)
            y_test_encoded = None if y_test is None else self._encode_labels(y_test)
            y_train_for_metrics = y_train
            y_test_for_metrics = y_test
        else:
            y_train_encoded = y_train
            y_test_encoded = y_test
            y_train_for_metrics = y_train
            y_test_for_metrics = y_test

        return y_train_encoded, y_test_encoded, y_train_for_metrics, y_test_for_metrics

    def _initialize_weights_for_fit(
        self, X_train: np.ndarray, y_train: np.ndarray, random_seed: int | None
    ) -> None:
        """Initialize weights when they were not supplied during construction.

        Args:
            X_train (np.ndarray): Training attributes.
            y_train (np.ndarray): Encoded training labels.
            random_seed (int): Random seed for reproducibility.
        """
        if self.thetas is not None:
            return

        if self.num_hidden_layers is None or self.num_neurons_per_hidden_layer is None:
            raise ValueError(
                "num_hidden_layers and num_neurons_per_hidden_layer are required "
                "when thetas are not supplied"
            )

        self.thetas = initialize_thetas(
            num_input_neurons=X_train.shape[1],
            num_hidden_layers=self.num_hidden_layers,
            num_neurons_per_hidden_layer=self.num_neurons_per_hidden_layer,
            num_output_neurons=y_train.shape[1],
            random_seed=random_seed,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        num_iterations: int,
        batch_size: int,
        random_seed: int,
        shuffle: bool,
        record_history: bool,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
    ) -> dict[str, list[Any]] | None:
        """Fit neural network.

        Args:
            X_train (list): Input values for training instances.
            y_train (list): True class labels for training instances.
            num_iterations (int): Number of times to process all training instances.
            batch_size (int): Number of training instances used for each weight update.
            random_seed (int): Random seed for reproducibility.
            shuffle (bool): Whether to shuffle training instances before creating batches.
            record_history (bool): Whether to record metrics after each weight update.
            X_test (list): Input values for test instances.
            y_test (list): True class labels for test instances.

        Returns:
            history (dict or None): Training and testing cost, accuracy, and F1 across weight updates.
                Returns None when record_history is False.
        """
        X_train = np.array(X_train)
        X_test = None if X_test is None else np.array(X_test)
        y_train, y_test, y_train_for_metrics, y_test_for_metrics = (
            self._prepare_labels_for_fit(y_train, y_test)
        )
        self._initialize_weights_for_fit(X_train, y_train, random_seed)

        history = None

        if record_history:
            history = {
                "iterations": [],
                "batches": [],
                "updates": [],
                "train_instances_seen": [],
                "train_cost": [],
                "test_cost": [],
                "train_accuracy": [],
                "test_accuracy": [],
                "train_f1": [],
                "test_f1": [],
            }

        random_generator = np.random.default_rng(random_seed)
        update_number = 0
        train_instances_seen = 0

        for iteration in range(num_iterations):
            batches = create_batches(
                X_train, y_train, batch_size, shuffle, random_generator
            )

            for batch_number, (X_batch, y_batch) in enumerate(batches, start=1):
                gradient_totals = [np.zeros_like(theta) for theta in self.thetas]

                for x_i, y_i in zip(X_batch, y_batch):
                    _, layers_activations, y_pred_i = self._forward_propagate(x_i)
                    deltas = compute_deltas(
                        self.thetas, y_pred_i, y_i, layers_activations
                    )
                    gradients = compute_gradients(layers_activations, deltas)

                    gradient_totals = accumulate_gradients(gradient_totals, gradients)

                final_gradients = regularize_and_average_gradients(
                    self.thetas,
                    gradient_totals,
                    self.regularization_strength,
                    len(X_batch),
                )
                self._update_weights(final_gradients, self.step_size)

                update_number += 1
                train_instances_seen += len(X_batch)

                if record_history:
                    train_cost = compute_average_regularized_cost_over_instances(
                        self, X_train, y_train, self.regularization_strength
                    )
                    train_accuracy, train_f1 = compute_metrics_over_instances(
                        self, X_train, y_train_for_metrics
                    )

                    history["iterations"].append(iteration + 1)
                    history["batches"].append(batch_number)
                    history["updates"].append(update_number)
                    history["train_instances_seen"].append(train_instances_seen)
                    history["train_cost"].append(train_cost)
                    history["train_accuracy"].append(train_accuracy)
                    history["train_f1"].append(train_f1)

                    if X_test is not None and y_test is not None:
                        test_cost = compute_average_regularized_cost_over_instances(
                            self, X_test, y_test, self.regularization_strength
                        )
                        test_accuracy, test_f1 = compute_metrics_over_instances(
                            self, X_test, y_test_for_metrics
                        )
                    else:
                        test_cost = None
                        test_accuracy = None
                        test_f1 = None

                    history["test_cost"].append(test_cost)
                    history["test_accuracy"].append(test_accuracy)
                    history["test_f1"].append(test_f1)

        return history

    def predict(self, X: np.ndarray) -> list[np.ndarray]:
        """Predict per-class probabilities.

        Args:
            X (list): Input values for instances.

        Returns:
            y_pred (list): Predicted per-class probabilities for instances.
        """
        y_pred = []

        for x_i in X:
            _, _, y_pred_i = self._forward_propagate(x_i)
            y_pred.append(y_pred_i)

        return y_pred

    def predict_index(self, X: np.ndarray) -> list[int]:
        """Predict class indices.

        Args:
            X (list): Input values for instances.

        Returns:
            predicted_indices (list): Predicted class indices for instances.
        """
        y_pred = self.predict(X)
        predicted_indices = []

        for y_pred_i in y_pred:
            if len(y_pred_i) == 1:
                predicted_index = int(y_pred_i[0] > 0.5)
            else:
                predicted_index = int(np.argmax(y_pred_i))

            predicted_indices.append(predicted_index)

        return predicted_indices

    def predict_label(self, X: np.ndarray) -> list[Any]:
        """Predict class labels.

        Args:
            X (list): Input values for instances.

        Returns:
            predicted_labels (list): Predicted class labels for instances.
        """
        if self.index_to_label is None:
            raise ValueError("predict_label requires fitting labels before prediction")

        predicted_indices = self.predict_index(X)
        predicted_labels = [
            self.index_to_label[predicted_index]
            for predicted_index in predicted_indices
        ]
        return predicted_labels
