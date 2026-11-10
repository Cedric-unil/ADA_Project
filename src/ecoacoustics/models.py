# Generated with Claude Code (Anthropic): define the baseline and sequence models.
"""The models compared in stage 05 and the code that trains them.

- Logistic regression on summary statistics, either of whole cuts or of windows.
- A reference model that sees only each cut's native sample rate.
- An LSTM that reads the frames of a window in order. Its control reads the same frames in a
  random order, which keeps what the frames contain but destroys their order.

Every ``fit_predict`` function trains a fresh model on the training cuts only and returns, for
each evaluated cut, one probability per class. Classes are numbered ``0 .. n_classes - 1``.
Training weights give every species the same total weight, which suits macro-F1.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ecoacoustics.evaluation import balanced_weights
from ecoacoustics.features import cut_windows, random_window

logger = logging.getLogger(__name__)


def _all_classes(probs: np.ndarray, seen: np.ndarray, n_classes: int) -> np.ndarray:
    """Put scikit-learn's probability columns under their class; unseen classes get 0."""
    full = np.zeros((len(probs), n_classes))
    full[:, seen] = probs
    return full


def average_by_cut(window_probs: np.ndarray, owner: np.ndarray, n_cuts: int) -> np.ndarray:
    """Average the probabilities of each cut's windows; ``owner`` gives each window's cut."""
    sums = np.zeros((n_cuts, window_probs.shape[1]))
    np.add.at(sums, owner, window_probs)
    return sums / np.bincount(owner, minlength=n_cuts)[:, None]


def fit_predict_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    n_classes: int,
    c: float,
    max_iter: int,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Standardise the inputs, fit a multinomial logistic regression, and predict ``x_eval``.

    The standardisation is fitted on the training rows only. ``c`` is the inverse strength of
    the L2 penalty: smaller values give a simpler model. Without ``sample_weight``, every
    species gets the same total weight.
    """
    if sample_weight is None:
        sample_weight = balanced_weights(y_train, n_classes)[y_train]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=max_iter))
    model.fit(
        x_train,
        y_train,
        standardscaler__sample_weight=sample_weight,
        logisticregression__sample_weight=sample_weight,
    )
    return _all_classes(model.predict_proba(x_eval), model.classes_, n_classes)


def fit_predict_window_logistic(
    train_windows: list[np.ndarray],
    y_train: np.ndarray,
    eval_windows: list[np.ndarray],
    n_classes: int,
    c: float,
    max_iter: int,
) -> np.ndarray:
    """Logistic regression on window statistics; a cut's prediction averages its windows.

    ``train_windows`` and ``eval_windows`` hold one array per cut, one row per window. Each
    training cut carries the same total weight however many windows it has, and every species
    the same total weight, so the 20-minute cuts do not dominate.
    """
    owner = np.concatenate([np.full(len(w), i) for i, w in enumerate(train_windows)])
    per_cut = balanced_weights(y_train, n_classes)[y_train] / np.bincount(owner)
    eval_owner = np.concatenate([np.full(len(w), i) for i, w in enumerate(eval_windows)])
    probs = fit_predict_logistic(
        np.vstack(train_windows),
        y_train[owner],
        np.vstack(eval_windows),
        n_classes,
        c,
        max_iter,
        sample_weight=per_cut[owner],
    )
    return average_by_cut(probs, eval_owner, len(eval_windows))


def fit_predict_rate_tree(
    rate_train: np.ndarray, y_train: np.ndarray, rate_eval: np.ndarray, n_classes: int, seed: int
) -> np.ndarray:
    """A decision tree that sees nothing but each cut's native sample rate (on a log scale).

    It shows how much the recording setup alone reveals about species.
    """
    tree = DecisionTreeClassifier(random_state=seed)
    tree.fit(
        np.log2(rate_train)[:, None],
        y_train,
        sample_weight=balanced_weights(y_train, n_classes)[y_train],
    )
    probs = tree.predict_proba(np.log2(rate_eval)[:, None])
    return _all_classes(probs, tree.classes_, n_classes)


class CallLSTM(nn.Module):
    """An LSTM reads a window frame by frame; a linear classifier reads the average of its outputs.

    The LSTM's output at each frame depends on every frame before it, so the average still
    depends on their order. Averaging over all frames, rather than keeping only the last
    output, makes training much faster and steadier (the pilot in stage 05b, on one validation
    fold). ``use_last_output`` gives the first design, which that pilot compares.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        use_last_output: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)
        self.use_last_output = use_last_output

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # The LSTM reads forwards, so its output at a frame depends only on that frame and the
        # ones before it: the zeros that pad a short window at its end cannot change any output
        # up to the window's real end. The mask leaves the padding out of the average, and the
        # last output is read at the real end. Packing the windows gives the same outputs, but
        # PyTorch then runs a much slower LSTM on the CPU (about 5 times slower on this data).
        outputs, _ = self.lstm(x)
        if self.use_last_output:
            last_output = outputs[torch.arange(len(lengths)), lengths - 1]
            return self.classifier(self.dropout(last_output))
        real = torch.arange(outputs.shape[1])[None, :] < lengths[:, None]
        average = (outputs * real.unsqueeze(-1)).sum(dim=1) / lengths[:, None]
        return self.classifier(self.dropout(average))


def lstm_seeds(seed: int, n_seeds: int) -> list[int]:
    """The seeds an LSTM is trained on: the seed from config.yaml, then the next integers."""
    return [seed + offset for offset in range(n_seeds)]


def frame_scaler(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of each frame column, counting every cut equally.

    Weighting cuts rather than frames stops the three 20-minute sperm whale cuts, a quarter of
    all frames, from setting the scale.
    """
    means = np.array([f.mean(axis=0, dtype=np.float64) for f in frames])
    squares = np.array([np.square(f, dtype=np.float64).mean(axis=0) for f in frames])
    mean = means.mean(axis=0)
    std = np.sqrt(squares.mean(axis=0) - mean**2)
    return mean.astype(np.float32), std.astype(np.float32)


def _batch(
    windows: list[np.ndarray], mean: np.ndarray, std: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardise the windows and pad them with zeros to the longest one."""
    tensors = [torch.from_numpy((w - mean) / std) for w in windows]
    return pad_sequence(tensors, batch_first=True), torch.tensor([len(w) for w in windows])


def _prepare_windows(
    windows: list[np.ndarray], scale: tuple[np.ndarray, np.ndarray], batch_size: int
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], np.ndarray]:
    """Standardise and pad the windows to predict once, in batches of windows of similar length.

    The same windows are predicted after every epoch, so they are prepared only once. Sorting
    them by length keeps the padding small. Returns the batches, and for each window in batch
    order its position in ``windows``.
    """
    order = np.argsort([len(w) for w in windows], kind="stable")
    batches = [
        _batch([windows[i] for i in order[start : start + batch_size]], *scale)
        for start in range(0, len(order), batch_size)
    ]
    return batches, order


def _predict_cuts(
    model: CallLSTM,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    order: np.ndarray,
    owner: np.ndarray,
    n_cuts: int,
) -> np.ndarray:
    """Probabilities for every window, averaged over the windows of each cut.

    ``batches`` and ``order`` come from :func:`_prepare_windows`; ``owner`` gives each window's
    cut, in the original order of the windows.
    """
    model.eval()
    with torch.no_grad():
        in_batch_order = np.vstack(
            [torch.softmax(model(x, lengths), dim=1).numpy() for x, lengths in batches]
        )
    probs = np.empty_like(in_batch_order)
    probs[order] = in_batch_order
    return average_by_cut(probs, owner, n_cuts)


def train_lstm(
    train_frames: list[np.ndarray],
    y_train: np.ndarray,
    eval_frames: list[np.ndarray],
    n_classes: int,
    settings: dict[str, Any],
    window: int,
    hop: int,
    n_epochs: int,
    shuffle_order: bool,
    seed: int,
) -> np.ndarray:
    """Train a fresh LSTM and return its probabilities for the evaluated cuts after each epoch.

    Each epoch shows the model one window, at a random position, of every training cut, so
    every cut counts once per epoch however long it is. A cut's prediction averages all its
    windows. With ``shuffle_order``, the frames inside every window are put in random order,
    in training and in evaluation. ``settings`` is the ``lstm`` section of config.yaml.

    With the same seed, the ordered and the shuffled model start from the same weights and see
    the same windows in the same batches. The number of CPU threads is fixed by
    ``settings["num_threads"]`` because it changes the results, through rounding.

    Returns an array of shape ``(n_epochs, len(eval_frames), n_classes)``.
    """
    rng = np.random.default_rng(seed)
    # Shuffling has its own generator: sharing one would change which windows and batches come next.
    shuffle_rng = np.random.default_rng([seed, 1])
    torch.manual_seed(seed)
    torch.set_num_threads(settings["num_threads"])
    scale = frame_scaler(train_frames)
    model = CallLSTM(
        train_frames[0].shape[1],
        n_classes,
        settings["hidden_size"],
        settings["num_layers"],
        settings["dropout"],
        settings.get("use_last_output", False),
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    class_weights = torch.tensor(balanced_weights(y_train, n_classes), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    labels = torch.from_numpy(y_train).long()

    eval_windows, eval_owner = [], []
    for i, frames in enumerate(eval_frames):
        windows = cut_windows(frames, window, hop)
        eval_windows += windows
        eval_owner += [i] * len(windows)
    if shuffle_order:
        eval_windows = [shuffle_rng.permutation(w) for w in eval_windows]
    eval_owner = np.array(eval_owner)
    eval_batches, eval_order = _prepare_windows(eval_windows, scale, settings["eval_batch_size"])

    history = np.empty((n_epochs, len(eval_frames), n_classes), dtype=np.float32)
    for epoch in range(n_epochs):
        model.train()
        crops = [random_window(frames, window, rng) for frames in train_frames]
        if shuffle_order:
            crops = [shuffle_rng.permutation(crop) for crop in crops]
        order = rng.permutation(len(crops))
        for start in range(0, len(order), settings["batch_size"]):
            batch = order[start : start + settings["batch_size"]]
            x, lengths = _batch([crops[i] for i in batch], *scale)
            loss = loss_fn(model(x, lengths), labels[torch.from_numpy(batch)])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        history[epoch] = _predict_cuts(
            model, eval_batches, eval_order, eval_owner, len(eval_frames)
        )
    return history


def parallel_jobs(settings: dict[str, Any]) -> int:
    """How many LSTMs to train at once: ``settings["n_jobs"]``, or fewer on a smaller computer.

    Each training uses ``settings["num_threads"]`` CPU threads, so no more are started than the
    computer has CPUs for. The number only changes the speed, never the results.
    """
    cpus = os.process_cpu_count() or 1
    return max(1, min(settings["n_jobs"], cpus // settings["num_threads"]))


def train_lstms(calls: list[dict[str, Any]], settings: dict[str, Any]) -> list[np.ndarray]:
    """Run :func:`train_lstm` once per entry of ``calls``, several at once, in separate processes.

    Each entry holds the arguments of one call to :func:`train_lstm`. The trainings do not
    depend on each other, and each one sets its own seed and number of threads, so the results
    are the same whether they run one by one or several at once. They come back in the order of
    ``calls``. ``settings`` is the ``lstm`` section of config.yaml, read for ``n_jobs`` and
    ``num_threads``.
    """
    n_jobs = min(parallel_jobs(settings), len(calls))
    logger.info("Training %d LSTMs, %d at a time", len(calls), n_jobs)
    if n_jobs <= 1:
        return [train_lstm(**call) for call in calls]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(train_lstm, **call) for call in calls]
        return [future.result() for future in futures]
