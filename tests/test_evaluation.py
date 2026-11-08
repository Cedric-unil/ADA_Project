# Generated with Claude Code (Anthropic): test the metrics, baselines and bootstrap.
"""Tests for the evaluation helpers, on small hand-made labels."""

import numpy as np
import pytest

from ecoacoustics.evaluation import (
    balanced_weights,
    bootstrap_gap,
    chance_scores,
    macro_f1,
    majority_predictions,
)


def test_macro_f1_averages_over_the_species_present() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 2])  # species 2 is absent, but the wrong guess still costs recall

    assert macro_f1(y_true, y_pred) == pytest.approx((1 + 2 / 3) / 2)


def test_balanced_weights_give_every_class_the_same_total() -> None:
    labels = np.array([0, 0, 0, 1, 2, 2])

    weights = balanced_weights(labels, n_classes=4)
    totals = np.bincount(labels, weights=weights[labels], minlength=4)

    np.testing.assert_allclose(totals[:3], totals[0])
    assert weights[3] == 0
    assert totals.sum() == pytest.approx(len(labels))


def test_chance_scores_are_close_to_one_over_the_number_of_classes() -> None:
    y_true = np.repeat(np.arange(4), 50)

    chance = chance_scores(y_true, np.arange(4), n_draws=200, rng=np.random.default_rng(0))

    assert chance["accuracy"] == pytest.approx(0.25, abs=0.02)
    assert chance["macro_f1"] == pytest.approx(0.25, abs=0.03)


def test_majority_predicts_the_most_common_training_class() -> None:
    assert majority_predictions(np.array([2, 2, 1]), n_eval=3).tolist() == [2, 2, 2]


def test_bootstrap_gap_is_zero_for_identical_models_and_positive_for_a_better_one() -> None:
    rng = np.random.default_rng(0)
    y_true = np.repeat(np.arange(3), 20)
    groups = np.repeat(np.arange(12), 5)
    noisy = np.where(rng.random(60) < 0.5, y_true, (y_true + 1) % 3)

    same = bootstrap_gap(y_true, noisy, noisy, groups, 200, 0.95, np.random.default_rng(1))
    better = bootstrap_gap(y_true, y_true, noisy, groups, 200, 0.95, np.random.default_rng(1))

    assert same == {"gap": 0.0, "ci_low": 0.0, "ci_high": 0.0, "share_a_better": 0.0}
    assert better["gap"] > 0
    assert better["ci_low"] > 0
    assert better["share_a_better"] == 1.0


def test_bootstrap_keeps_whole_groups_together() -> None:
    # With a single group, every bootstrap sample holds exactly the original cuts.
    y_true = np.array([0, 1, 0, 1])
    pred_a = np.array([0, 1, 1, 1])
    pred_b = np.array([0, 0, 0, 1])

    result = bootstrap_gap(
        y_true, pred_a, pred_b, np.zeros(4), 50, 0.9, np.random.default_rng(0)
    )

    assert result["ci_low"] == pytest.approx(result["gap"])
    assert result["ci_high"] == pytest.approx(result["gap"])
