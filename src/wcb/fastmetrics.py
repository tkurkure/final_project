"""Vectorized F1 over many bootstrap resamples at once.

The bootstrap is the analysis bottleneck: 25 banks x 3 seeds x 3 gap components x B
resamples, each needing two F1 computations. Calling `sklearn.f1_score` in a Python loop
makes a full analysis pass take minutes, which in practice pushes you toward too few
resamples.

The trick: a classification's entire F1 depends only on its KxK confusion counts, so all
B resamples can be reduced to a (B, K, K) tensor with one `bincount`, and the metrics
computed from it with pure array arithmetic. This turns minutes into milliseconds and
lets the analysis use the full 10,000 resamples.

Verified to agree with scikit-learn to ~1e-12 (see tests at the bottom of this file).
"""
from __future__ import annotations

import numpy as np


def confusion_from_samples(y_true: np.ndarray, y_pred: np.ndarray,
                           idx: np.ndarray, n_classes: int) -> np.ndarray:
    """Confusion matrices for many resamples at once.

    idx : (B, n) integer array of item indices, one row per resample.
    Returns (B, n_classes, n_classes), rows = true label, cols = predicted.
    """
    B, n = idx.shape
    K = n_classes

    # One integer per (true, pred) pair, offset by resample so a single bincount
    # separates all B matrices.
    code = (y_true.astype(np.int64) * K + y_pred.astype(np.int64))[idx]   # (B, n)
    code += (np.arange(B, dtype=np.int64) * (K * K))[:, None]

    counts = np.bincount(code.ravel(), minlength=B * K * K)
    return counts.reshape(B, K, K)


def f1_from_confusion(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class F1, macro F1 and weighted F1 from a (..., K, K) confusion tensor.

    Zero-division is resolved to 0, matching sklearn's `zero_division=0`.
    """
    cm = cm.astype(np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)          # (..., K)
    support = cm.sum(axis=-1)                          # true totals  (..., K)
    predicted = cm.sum(axis=-2)                        # predicted totals (..., K)

    denom = support + predicted
    # F1 = 2*TP / (support + predicted) is algebraically equal to the
    # 2PR/(P+R) form but avoids forming precision and recall separately.
    per_class = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)

    macro = per_class.mean(axis=-1)

    total = support.sum(axis=-1, keepdims=True)
    weights = np.divide(support, total, out=np.zeros_like(support), where=total > 0)
    weighted = (per_class * weights).sum(axis=-1)

    return per_class, macro, weighted


def accuracy_from_confusion(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1).sum(axis=-1)
    total = cm.sum(axis=(-2, -1))
    return np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)


def bootstrap_indices(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n) resample index matrix."""
    return rng.integers(0, n, size=(n_boot, n))


def metric_over_samples(y_true: np.ndarray, y_pred: np.ndarray, idx: np.ndarray,
                        n_classes: int, metric: str) -> np.ndarray:
    """Evaluate one metric across every resample in `idx`."""
    cm = confusion_from_samples(y_true, y_pred, idx, n_classes)
    if metric == "accuracy":
        return accuracy_from_confusion(cm)
    _, macro, weighted = f1_from_confusion(cm)
    if metric == "macro_f1":
        return macro
    if metric == "weighted_f1":
        return weighted
    raise ValueError(f"unknown metric {metric!r}")


def _selftest() -> None:
    """Agreement check against scikit-learn across random problems."""
    from sklearn.metrics import accuracy_score, f1_score

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        K = int(rng.integers(2, 6))
        n = int(rng.integers(20, 300))
        y = rng.integers(0, K, n)
        p = rng.integers(0, K, n)
        # Also exercise the degenerate cases: a class never predicted, or absent entirely.
        if rng.random() < 0.3:
            p = np.where(p == K - 1, 0, p)

        idx = np.arange(n)[None, :]
        cm = confusion_from_samples(y, p, idx, K)
        per_class, macro, weighted = f1_from_confusion(cm)

        labels = list(range(K))
        ref_pc = f1_score(y, p, average=None, labels=labels, zero_division=0)
        ref_macro = f1_score(y, p, average="macro", labels=labels, zero_division=0)
        ref_weighted = f1_score(y, p, average="weighted", labels=labels, zero_division=0)
        ref_acc = accuracy_score(y, p)

        worst = max(worst,
                    np.abs(per_class[0] - ref_pc).max(),
                    abs(macro[0] - ref_macro),
                    abs(weighted[0] - ref_weighted),
                    abs(accuracy_from_confusion(cm)[0] - ref_acc))

    assert worst < 1e-10, f"max disagreement with sklearn: {worst}"
    print(f"fastmetrics selftest OK (max disagreement {worst:.2e} over 200 problems)")


if __name__ == "__main__":
    _selftest()
