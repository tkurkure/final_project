"""Metrics, bootstrap confidence intervals, and the paired bootstrap for transfer gaps."""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from . import config as C
from .fastmetrics import bootstrap_indices, metric_over_samples


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Comparable with the WCB paper's headline numbers (RoBERTa-base pooled = 0.714)."""
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Counts every label equally. Matters here because `irrelevant` is only ~3.4% of the
    data, so weighted F1 is dominated by the three common classes."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "weighted_f1": weighted_f1,
    "macro_f1": macro_f1,
    "accuracy": accuracy_score,
}


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict:
    """Full metric bundle for one evaluation set."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(C.STANCE_LABELS)))

    per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    support = np.bincount(y_true, minlength=len(labels))

    return {
        "n": int(len(y_true)),
        "weighted_f1": float(weighted_f1(y_true, y_pred)),
        "macro_f1": float(macro_f1(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": {C.ID2STANCE[i]: float(per_class[i]) for i in labels},
        "support": {C.ID2STANCE[i]: int(support[i]) for i in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def bootstrap_ci(y_true: Sequence[int], y_pred: Sequence[int],
                 metric: str = "weighted_f1",
                 n_boot: int = C.N_BOOTSTRAP,
                 ci: float = C.BOOTSTRAP_CI,
                 rng_seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap over evaluation *items*.

    Per-bank test sets hold only 128-176 sentences, so a point estimate of F1 carries
    real sampling noise; this quantifies it.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(rng_seed)
    idx = bootstrap_indices(len(y_true), n_boot, rng)

    stats = metric_over_samples(y_true, y_pred, idx, len(C.STANCE_LABELS), metric)

    lo, hi = np.percentile(stats, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"point": float(METRICS[metric](y_true, y_pred)), "lo": float(lo),
            "hi": float(hi), "se": float(stats.std(ddof=1))}


def paired_bootstrap_gap(y_true: Sequence[int],
                         y_pred_a: Sequence[int],
                         y_pred_b: Sequence[int],
                         metric: str = "weighted_f1",
                         n_boot: int = C.N_BOOTSTRAP,
                         ci: float = C.BOOTSTRAP_CI,
                         rng_seed: int = 0) -> Dict[str, float]:
    """CI on (metric(A) - metric(B)) when A and B scored the SAME items.

    Resampling items once and recomputing both metrics on that resample preserves the
    correlation between the two models' errors. That makes the interval both correct and
    tighter than differencing two independent CIs -- the two models agree on most
    sentences, and a paired design credits that agreement.

    Caller must have verified the two prediction vectors are aligned row-for-row
    (see splits.assert_same_evaluation_items).
    """
    y_true = np.asarray(y_true)
    a = np.asarray(y_pred_a)
    b = np.asarray(y_pred_b)
    assert len(y_true) == len(a) == len(b), "paired bootstrap needs aligned vectors"

    rng = np.random.default_rng(rng_seed)
    K = len(C.STANCE_LABELS)
    # ONE index matrix drives both models: that shared resample is exactly what makes
    # the comparison paired.
    idx = bootstrap_indices(len(y_true), n_boot, rng)
    diffs = (metric_over_samples(y_true, a, idx, K, metric)
             - metric_over_samples(y_true, b, idx, K, metric))

    lo, hi = np.percentile(diffs, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    fn = METRICS[metric]
    point = fn(y_true, a) - fn(y_true, b)
    return {
        "gap": float(point), "lo": float(lo), "hi": float(hi),
        "se": float(diffs.std(ddof=1)),
        # Fraction of resamples where the gap reverses sign -- a two-sided bootstrap
        # p-value for "is there any gap at all".
        "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
    }
