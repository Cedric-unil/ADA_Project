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
    replacement and keeps all cuts of each drawn tape. Returns the observed gap, the interval,
    and the share of bootstrap samples in which A beats B.
    """
    tape_ids = np.unique(groups)
    rows_of = {tape: np.flatnonzero(groups == tape) for tape in tape_ids}
    gaps = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        drawn = rng.choice(tape_ids, size=len(tape_ids), replace=True)
        rows = np.concatenate([rows_of[tape] for tape in drawn])
        gaps[b] = macro_f1(y_true[rows], pred_a[rows]) - macro_f1(y_true[rows], pred_b[rows])
    tail = 100 * (1 - ci_level) / 2
    low, high = np.percentile(gaps, [tail, 100 - tail])
    return {
        "gap": macro_f1(y_true, pred_a) - macro_f1(y_true, pred_b),
        "ci_low": float(low),
        "ci_high": float(high),
        "share_a_better": float(np.mean(gaps > 0)),
    }
