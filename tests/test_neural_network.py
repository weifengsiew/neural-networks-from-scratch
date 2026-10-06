import numpy as np

from algorithms.neural_network import (
    NeuralNetwork,
    create_batches,
    initialize_thetas,
    sigmoid,
)


def test_sigmoid_returns_expected_values():
    np.testing.assert_allclose(
        sigmoid(np.array([-1.0, 0.0, 1.0])), [0.2689, 0.5, 0.7311], atol=1e-4
    )


def test_initialize_thetas_returns_expected_shapes():
    thetas = initialize_thetas(3, 2, 4, 2, random_seed=7)

    assert [theta.shape for theta in thetas] == [(4, 4), (4, 5), (2, 5)]


def test_create_batches_preserves_instances_without_shuffle():
    X = np.arange(12).reshape(6, 2)
    y = np.arange(6)

    batches = create_batches(
        X, y, batch_size=4, shuffle=False, random_generator=np.random.default_rng(7)
    )

    assert [len(batch_X) for batch_X, _ in batches] == [4, 2]
    np.testing.assert_array_equal(batches[0][0], X[:4])
    np.testing.assert_array_equal(batches[1][1], y[4:])


def test_network_fits_binary_labels_and_predicts_labels():
    X = np.array([[0.0], [1.0]])
    y = np.array(["negative", "positive"])
    network = NeuralNetwork(num_hidden_layers=1, num_neurons_per_hidden_layer=2)
    network.configure_for_fit(regularization_strength=0.0, step_size=0.5)

    history = network.fit(
        X,
        y,
        num_iterations=2,
        batch_size=2,
        random_seed=7,
        shuffle=False,
        record_history=True,
    )

    assert history is not None
    assert len(history["updates"]) == 2
    assert set(network.predict_label(X)) <= set(y)
