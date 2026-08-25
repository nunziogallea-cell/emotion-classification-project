
import numpy as np

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)


def sigmoid(x):
    """
    Convert logits into probabilities.
    """
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    """
    Compute multilabel classification metrics.

    A threshold of 0.5 is used for the baseline evaluation.
    """

    logits, labels = eval_pred

    probabilities = sigmoid(logits)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "micro_f1": f1_score(
            labels,
            predictions,
            average="micro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "micro_precision": precision_score(
            labels,
            predictions,
            average="micro",
            zero_division=0,
        ),
        "micro_recall": recall_score(
            labels,
            predictions,
            average="micro",
            zero_division=0,
        ),
    }
