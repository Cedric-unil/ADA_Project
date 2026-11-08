# Generated with Claude Code (Anthropic): test the models on small synthetic tasks.
"""Tests for the models. They use small synthetic data, never the real data.

The order task has two classes made of the same values in opposite order: one rises, the other
falls. Their mean and standard deviation are identical, so only a model that reads the frames
in order can tell them apart.
"""

import numpy as np
import pytest

from ecoacoustics.features import window_statistics
from ecoacoustics.models import (
    average_by_cut,
    fit_predict_logistic,
    fit_predict_rate_tree,
    fit_predict_window_logistic,
    frame_scaler,
    train_lstm,
)

LENGTH = 30
SETTINGS = {
    "hidden_size": 8,
    "num_layers": 1,
    "dropout": 0.0,
    "batch_size": 16,
    "eval_batch_size": 64,
    "learning_rate": 0.01,
}


def _order_task(n_cuts: int, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    ramp = np.linspace(-1, 1, LENGTH)[:, None]
    labels = np.arange(n_cuts) % 2
    frames = [
        ((ramp if label == 0 else ramp[::-1]) + 0.05 * rng.normal(size=(LENGTH, 1))).astype(
            np.float32
        )
        for label in labels
    ]
    return frames, labels


def _lstm_accuracy(shuffle_order: bool, n_epochs: int = 30) -> tuple[float, np.ndarray]:
    train_frames, y_train = _order_task(40, seed=0)
    eval_frames, y_eval = _order_task(20, seed=1)
    history = train_lstm(
        train_frames, y_train, eval_frames, 2, SETTINGS, LENGTH, LENGTH, n_epochs, shuffle_order, 0
    )
    return float(np.mean(history[-1].argmax(axis=1) == y_eval)), history


def test_lstm_learns_order_and_returns_probabilities_after_every_epoch() -> None:
    accuracy, history = _lstm_accuracy(shuffle_order=False)

    assert history.shape == (30, 20, 2)
    np.testing.assert_allclose(history.sum(axis=2), 1.0, atol=1e-5)
    assert accuracy >= 0.9


def test_lstm_on_shuffled_frames_cannot_learn_order() -> None:
    accuracy, _ = _lstm_accuracy(shuffle_order=True)

    assert accuracy <= 0.75


def test_window_statistics_cannot_learn_order() -> None:
    train_frames, y_train = _order_task(40, seed=0)
    eval_frames, y_eval = _order_task(20, seed=1)

    probs = fit_predict_window_logistic(
        [window_statistics([f]) for f in train_frames],
        y_train,
        [window_statistics([f]) for f in eval_frames],
        2,
        1.0,
        2000,
    )

    assert np.mean(probs.argmax(axis=1) == y_eval) <= 0.75


def test_lstm_training_is_identical_with_the_same_seed() -> None:
    _, first = _lstm_accuracy(shuffle_order=True, n_epochs=3)
    _, second = _lstm_accuracy(shuffle_order=True, n_epochs=3)

    np.testing.assert_array_equal(first, second)


def test_logistic_regression_returns_a_column_for_classes_absent_from_training() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    y = (x[:, 0] > 0).astype(int)

    probs = fit_predict_logistic(x, y, x, n_classes=3, c=1.0, max_iter=2000)

    assert probs.shape == (40, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert (probs[:, 2] == 0).all()
    assert np.mean(probs.argmax(axis=1) == y) > 0.9


def test_rate_tree_uses_only_the_sample_rate() -> None:
    rates = np.array([600, 600, 48000, 48000, 96000, 96000])
    y = np.array([0, 0, 1, 1, 2, 2])

    probs = fit_predict_rate_tree(rates, y, np.array([600, 96000]), n_classes=4, seed=0)

    assert probs.argmax(axis=1).tolist() == [0, 2]
    assert probs.shape == (2, 4)


def test_frame_scaler_counts_every_cut_equally() -> None:
    short = np.zeros((1, 2), dtype=np.float32)
    long = np.full((100, 2), 10.0, dtype=np.float32)

    mean, std = frame_scaler([short, long])

    np.testing.assert_allclose(mean, 5.0)
    np.testing.assert_allclose(std, 5.0)


def test_average_by_cut_averages_each_cuts_windows() -> None:
    window_probs = np.array([[1.0, 0.0], [0.0, 1.0], [0.2, 0.8]])

    probs = average_by_cut(window_probs, np.array([0, 0, 1]), n_cuts=2)

    np.testing.assert_allclose(probs, [[0.5, 0.5], [0.2, 0.8]])
    assert probs.shape == (2, 2)
    with pytest.raises(IndexError):
        average_by_cut(window_probs, np.array([0, 0, 2]), n_cuts=2)
