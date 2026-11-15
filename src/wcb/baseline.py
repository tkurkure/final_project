"""TF-IDF + logistic regression baseline.

Cheap enough to run across all 25 LOBO folds x 3 split seeds in minutes, so it produces
a complete transfer-gap picture before any GPU time is spent -- and stands as the
fallback result if GPU access falls through.

Uses the *aggressive* preprocessing path (casefolding, punctuation stripping). The
RoBERTa path deliberately does not do this: a pretrained transformer expects natural
casing and punctuation, which carry signal its tokenizer was trained on.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config as C
from .splits import Fold

_PUNCT = re.compile(r"[^a-z0-9\s%]+")
_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")


def normalize_aggressive(text: str) -> str:
    """Casefold, map numerals to a placeholder, drop punctuation, collapse whitespace.

    Numbers are collapsed to <num> because the specific value ("2.5 percent") is noise
    for a bag-of-words model while the *presence* of a figure is signal.
    """
    text = text.lower()
    text = _NUM.sub(" <num> ", text)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def build_pipeline(C_reg: float = 1.0, max_features: Optional[int] = 50_000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=max_features,
            preprocessor=normalize_aggressive,
        )),
        ("clf", LogisticRegression(
            C=C_reg, max_iter=2000, solver="lbfgs",
        )),
    ])


def tune_C(fold: Fold, grid: Sequence[float] = (0.1, 0.5, 1.0, 5.0, 10.0),
           metric: str = "weighted_f1") -> float:
    """Select regularization strength on the fold's validation split.

    Tuned once on the pooled fold and then frozen, mirroring how the RoBERTa
    hyperparameters are handled -- tuning per fold would let each LOBO model peek at a
    different model-selection budget and make the 25 folds incomparable.
    """
    from .evaluate import METRICS
    fn = METRICS[metric]
    best_c, best_score = grid[0], -np.inf
    for c in grid:
        pipe = build_pipeline(C_reg=c)
        pipe.fit(fold.train[C.TEXT_COLUMN], fold.train["stance_id"])
        score = fn(fold.val["stance_id"].values, pipe.predict(fold.val[C.TEXT_COLUMN]))
        if score > best_score:
            best_c, best_score = c, score
    return best_c


def run_fold(fold: Fold, eval_df: pd.DataFrame, C_reg: float = 1.0) -> Dict:
    """Fit on the fold's training rows and predict the evaluation rows.

    Returns predictions aligned row-for-row with `eval_df` so they can be paired against
    any other condition's predictions on the same frame.
    """
    pipe = build_pipeline(C_reg=C_reg)
    pipe.fit(fold.train[C.TEXT_COLUMN], fold.train["stance_id"])

    probs = pipe.predict_proba(eval_df[C.TEXT_COLUMN])
    preds = probs.argmax(axis=1)

    # predict_proba columns follow clf.classes_, which may omit a class absent from
    # training; remap onto the canonical 4-way ordering so saved probabilities are always
    # interpretable by index.
    full = np.zeros((len(eval_df), len(C.STANCE_LABELS)))
    for col, cls in enumerate(pipe.named_steps["clf"].classes_):
        full[:, int(cls)] = probs[:, col]
    preds = np.array([pipe.named_steps["clf"].classes_[i] for i in probs.argmax(axis=1)], dtype=int)

    return {
        "sentence_id": eval_df["sentence_id"].tolist(),
        "y_true": eval_df["stance_id"].tolist(),
        "y_pred": preds.tolist(),
        "probs": full.tolist(),
        "n_train": len(fold.train),
    }
