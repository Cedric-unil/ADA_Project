# Generated with Claude Code (Anthropic): test the models on small synthetic tasks.
"""Tests for the models. They use small synthetic data, never the real data.

The order task has two classes made of the same values in opposite order: one rises, the other
falls. Their mean and standard deviation are identical, so only a model that reads the frames
in order can tell them apart.
"""

import os

import numpy as np
import pytest
import torch

from ecoacoustics import models
from ecoacoustics.features import window_statistics
from ecoacoustics.models import (
    CallLSTM,
    average_by_cut,
    fit_predict_logistic,
    fit_predict_rate_tree,
    fit_predict_window_logistic,
    frame_scaler,
    lstm_seeds,
    parallel_jobs,
    train_lstm,
    train_lstms,
)

LENGTH = 30
SETTINGS = {
    "hidden_size": 8,
    "num_layers": 1,
    "dropout": 0.0,
    "batch_size": 16,
    "eval_batch_size": 64,
    "learning_rate": 0.01,
    "num_threads": 2,
    "n_jobs": 2,
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


def test_ordered_and_shuffled_lstm_draw_the_same_windows_in_the_same_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(0)
    frames = [rng.normal(size=(30 + i, 2)).astype(np.float32) for i in range(12)]
    y_train = np.arange(12) % 3
    original_window, original_batch = models.random_window, models._batch

    def record(shuffle_order: bool) -> tuple[list[float], list[list[tuple[float, ...]]]]:
        starts: list[float] = []
        batches: list[list[tuple[float, ...]]] = []

        def window_spy(*args: object) -> np.ndarray:
            window = original_window(*args)
            starts.append(float(window[0, 0]))  # the first value, before any shuffling
            return window

        def batch_spy(batch: list[np.ndarray], *args: object) -> object:
            # Shuffling keeps the values of a window, so its sorted values identify it.
            batches.append([tuple(np.sort(window[:, 0]).tolist()) for window in batch])
            return original_batch(batch, *args)

        monkeypatch.setattr(models, "random_window", window_spy)
        monkeypatch.setattr(models, "_batch", batch_spy)
        train_lstm(frames, y_train, frames[:2], 3, SETTINGS, 10, 10, 3, shuffle_order, 42)
        return starts, batches

    ordered_starts, ordered_batches = record(shuffle_order=False)
    shuffled_starts, shuffled_batches = record(shuffle_order=True)

    assert ordered_starts == shuffled_starts
    assert ordered_batches == shuffled_batches


def test_lstm_runs_on_the_configured_number_of_threads() -> None:
    before = torch.get_num_threads()
    try:
        _lstm_accuracy(shuffle_order=False, n_epochs=1)
        assert torch.get_num_threads() == SETTINGS["num_threads"]
    finally:
        torch.set_num_threads(before)


def test_lstms_trained_at_the_same_time_give_the_same_results_as_one_by_one() -> None:
    train_frames, y_train = _order_task(20, seed=0)
    eval_frames, _ = _order_task(6, seed=1)
    calls = [
        {
            "train_frames": train_frames,
            "y_train": y_train,
            "eval_frames": eval_frames,
            "n_classes": 2,
            "settings": SETTINGS,
            "window": LENGTH,
            "hop": LENGTH,
            "n_epochs": 2,
            "shuffle_order": shuffle_order,
            "seed": seed,
        }
        for shuffle_order in (False, True)
        for seed in (0, 1)
    ]
    before = torch.get_num_threads()
    try:
        one_by_one = train_lstms(calls, {**SETTINGS, "n_jobs": 1})
        at_the_same_time = train_lstms(calls, {**SETTINGS, "n_jobs": 2})
    finally:
        torch.set_num_threads(before)

    assert len(at_the_same_time) == len(calls)
    for alone, together in zip(one_by_one, at_the_same_time, strict=True):
        np.testing.assert_array_equal(alone, together)


def test_parallel_jobs_never_ask_for_more_cpus_than_there_are() -> None:
    cpus = os.process_cpu_count() or 1

    assert parallel_jobs({"n_jobs": 1, "num_threads": 1}) == 1
    assert parallel_jobs({"n_jobs": 10 * cpus, "num_threads": 1}) == cpus
    assert parallel_jobs({"n_jobs": 4, "num_threads": 10 * cpus}) == 1


def test_lstm_predictions_do_not_depend_on_how_windows_are_batched() -> None:
    # Cuts of different lengths give windows of different lengths, which are sorted by length
    # and padded before they are predicted; the predictions must come back to the right cut.
    train_frames, y_train = _order_task(20, seed=0)
    rng = np.random.default_rng(2)
    eval_frames = [rng.normal(size=(n, 1)).astype(np.float32) for n in (30, 7, 19, 3, 30, 12)]

    def run(eval_batch_size: int) -> np.ndarray:
        settings = {**SETTINGS, "eval_batch_size": eval_batch_size}
        return train_lstm(train_frames, y_train, eval_frames, 2, settings, 10, 5, 2, False, 0)

    np.testing.assert_allclose(run(1), run(64), atol=1e-6)


def test_lstm_seeds_start_from_the_seed_in_the_config() -> None:
    assert lstm_seeds(42, 5) == [42, 43, 44, 45, 46]


def test_lstm_can_read_the_last_output_for_the_pilot() -> None:
    torch.manual_seed(0)
    last = CallLSTM(2, 3, 4, 1, 0.0, use_last_output=True).eval()
    average = CallLSTM(2, 3, 4, 1, 0.0).eval()
    average.load_state_dict(last.state_dict())
    short = torch.randn(5, 2)
    padded = torch.zeros(9, 2)
    padded[:5] = short
    lengths = torch.tensor([5, 9])
    batch = torch.stack([padded, torch.randn(9, 2)])

    with torch.no_grad():
        outputs, _ = last.lstm(short[None])
        from_last, from_average = last(batch, lengths), average(batch, lengths)
        expected_last = last.classifier(outputs[0, -1])
        expected_average = last.classifier(outputs[0].mean(dim=0))

    # The last real output of the short window, not the output over its padding.
    torch.testing.assert_close(from_last[0], expected_last)
    torch.testing.assert_close(from_average[0], expected_average)


def test_logistic_scaler_is_fitted_on_training_rows_only() -> None:
    # Required by CLAUDE.md. PCA joins this requirement when stage 04 exists.
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(60, 3))
    y_train = (x_train[:, 0] > 0).astype(int)
    x_eval = rng.normal(size=(10, 3))
    extreme = 1000 * x_eval

    alone = fit_predict_logistic(x_train, y_train, x_eval, 2, 1.0, 2000)
    beside_extreme_rows = fit_predict_logistic(
        x_train, y_train, np.vstack([x_eval, extreme]), 2, 1.0, 2000
    )

    # A scaler fitted on the evaluated rows too would be changed by the extreme rows.
    np.testing.assert_array_equal(alone, beside_extreme_rows[:10])


def test_window_logistic_scaler_is_fitted_on_training_rows_only() -> None:
    rng = np.random.default_rng(0)
    train_windows = [rng.normal(size=(3, 4)) for _ in range(30)]
    y_train = np.arange(30) % 2
    eval_windows = [rng.normal(size=(2, 4)) for _ in range(5)]
    extreme = [1000 * windows for windows in eval_windows]

    alone = fit_predict_window_logistic(train_windows, y_train, eval_windows, 2, 1.0, 2000)
    beside_extreme_rows = fit_predict_window_logistic(
        train_windows, y_train, eval_windows + extreme, 2, 1.0, 2000
    )

    np.testing.assert_array_equal(alone, beside_extreme_rows[:5])


def test_lstm_frame_scaler_is_fitted_on_training_frames_only() -> None:
    train_frames, y_train = _order_task(20, seed=0)
    eval_frames, _ = _order_task(6, seed=1)
    extreme = [1000 * frames for frames in eval_frames]

    def run(evaluated: list[np.ndarray]) -> np.ndarray:
        return train_lstm(
            train_frames, y_train, evaluated, 2, SETTINGS, LENGTH, LENGTH, 2, False, 0
        )

    alone = run(eval_frames)
    beside_extreme_frames = run(eval_frames + extreme)

    # Only close, not equal: a larger batch can round slightly differently.
    np.testing.assert_allclose(alone, beside_extreme_frames[:, :6], atol=1e-5)


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
