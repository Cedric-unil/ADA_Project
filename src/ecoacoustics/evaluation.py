# Generated with Claude Code (Anthropic): metrics, baselines and the bootstrap comparison.
"""Metrics, reference baselines, class weights and the tape-level bootstrap.

Macro-F1 is the main metric. It is the plain average of the F1 score of each species present
among the evaluated cuts, so every species counts equally however many cuts it has. A
prediction of a species that is absent from the evaluated cuts still counts as an error, through
the recall of the true species.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average F1 over the species present in ``y_true``."""
    return float(
        f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0)
    )


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Macro-F1 and accuracy of one set of predictions."""
    return {"macro_f1": macro_f1(y_true, y_pred), "accuracy": float(accuracy_score(y_true, y_pred))}


def scores_across_runs(y_true: np.ndarray, runs: np.ndarray) -> dict[str, float]:
    """Mean and spread of macro-F1 and accuracy over several runs of one model.

    ``runs`` has one row of predictions per run, here one per seed. The spread is the standard
    deviation (with ``n - 1``) of each score and the lowest and highest macro-F1. A single run
    has no spread, so those entries are NaN.
    """
    each = [scores(y_true, run) for run in np.atleast_2d(runs)]
    f1 = np.array([score["macro_f1"] for score in each])
    accuracy = np.array([score["accuracy"] for score in each])
    several = len(each) > 1
    return {
        "macro_f1": float(f1.mean()),
        "macro_f1_sd": float(f1.std(ddof=1)) if several else float("nan"),
        "macro_f1_min": float(f1.min()) if several else float("nan"),
        "macro_f1_max": float(f1.max()) if several else float("nan"),
        "accuracy": float(accuracy.mean()),
        "accuracy_sd": float(accuracy.std(ddof=1)) if several else float("nan"),
    }


def gaps_by_run(y_true: np.ndarray, runs_a: np.ndarray, runs_b: np.ndarray) -> np.ndarray:
    """Macro-F1 of model A minus model B, run by run, so the gap is paired by seed.

    A model with a single run, such as a logistic regression, is compared with every run of the
    other.
    """
    f1_a = np.array([macro_f1(y_true, run) for run in np.atleast_2d(runs_a)])
    f1_b = np.array([macro_f1(y_true, run) for run in np.atleast_2d(runs_b)])
    return f1_a - f1_b


def gap_spread(y_true: np.ndarray, runs_a: np.ndarray, runs_b: np.ndarray) -> dict[str, float]:
    """Spread of the paired macro-F1 gap across runs: standard deviation, lowest and highest."""
    gaps = gaps_by_run(y_true, runs_a, runs_b)
    if len(gaps) == 1:
        return dict.fromkeys(("gap_sd", "gap_min", "gap_max"), float("nan"))
    return {
        "gap_sd": float(gaps.std(ddof=1)),
        "gap_min": float(gaps.min()),
        "gap_max": float(gaps.max()),
    }


def balanced_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """One weight per class, so that every class carries the same total weight in training.

    A class with ``n_k`` of ``n`` examples gets ``n / (n_classes * n_k)``, the rule scikit-learn
    calls "balanced". Classes without examples get weight 0.
    """
    counts = np.bincount(labels, minlength=n_classes)
    weights = np.zeros(n_classes)
    present = counts > 0
    weights[present] = len(labels) / (present.sum() * counts[present])
    return weights


def chance_scores(
    y_true: np.ndarray, classes: np.ndarray, n_draws: int, rng: np.random.Generator
) -> dict[str, float]:
    """Expected scores of guessing uniformly among ``classes``, averaged over ``n_draws``."""
    draws = [scores(y_true, rng.choice(classes, len(y_true))) for _ in range(n_draws)]
    return {key: float(np.mean([draw[key] for draw in draws])) for key in draws[0]}


def majority_predictions(train_labels: np.ndarray, n_eval: int) -> np.ndarray:
    """Predict the most common training species for every evaluated cut."""
    return np.full(n_eval, np.bincount(train_labels).argmax())


def _mean_macro_f1(y_true: np.ndarray, runs: np.ndarray) -> float:
    return float(np.mean([macro_f1(y_true, run) for run in runs]))


def bootstrap_gap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    n_bootstrap: int,
    ci_level: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Macro-F1 of model A minus model B, with a bootstrap interval that resamples whole tapes.

    Cuts from the same tape are not independent, so each bootstrap sample draws tapes with
    replacement and keeps all cuts of each drawn tape. ``pred_a`` and ``pred_b`` hold one row of
    predictions per run, or a single row, and a model's score is the mean over its runs. Returns
    the observed gap, the interval, and the share of bootstrap samples in which A beats B.
    """
    runs_a, runs_b = np.atleast_2d(pred_a), np.atleast_2d(pred_b)
    tape_ids = np.unique(groups)
    rows_of = {tape: np.flatnonzero(groups == tape) for tape in tape_ids}
    gaps = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        drawn = rng.choice(tape_ids, size=len(tape_ids), replace=True)
        rows = np.concatenate([rows_of[tape] for tape in drawn])
        gaps[b] = _mean_macro_f1(y_true[rows], runs_a[:, rows]) - _mean_macro_f1(
            y_true[rows], runs_b[:, rows]
        )
    tail = 100 * (1 - ci_level) / 2
    low, high = np.percentile(gaps, [tail, 100 - tail])
    return {
        "gap": _mean_macro_f1(y_true, runs_a) - _mean_macro_f1(y_true, runs_b),
        "ci_low": float(low),
        "ci_high": float(high),
        "share_a_better": float(np.mean(gaps > 0)),
    }
