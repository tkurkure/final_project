"""Bank-level language features used to ask *what predicts* the transfer gap.

These six features are fixed in advance, before any gap is computed. With only n = 25
banks the analysis is badly underpowered, so the honest protection against fishing is to
commit to a small feature set up front and report it whole -- including the features that
turn out flat -- rather than searching for whichever correlation happens to land.

Every feature is computed to mirror the LOBO setup it is meant to explain: the bank's own
statistics come from its TRAINING rows, and the reference distribution comes from the
*other 24* banks' training rows. Test rows are never touched, so no feature can peek at
the labels the gap is measured on.
"""
from __future__ import annotations

import collections
import logging
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from . import config as C
from .data import filter_banks

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z]+")

FEATURE_NAMES = [
    "stance_js_divergence",
    "mean_sentence_length",
    "hedging_rate",
    "forward_looking_rate",
    "vocabulary_novelty",
    "embedding_distance",
]


def _label_distribution(df: pd.DataFrame, column: str, labels: List[str]) -> np.ndarray:
    counts = df[column].value_counts()
    vec = np.array([counts.get(lab, 0) for lab in labels], dtype=float)
    return vec / vec.sum() if vec.sum() else vec


def stance_js_divergence(bank_df: pd.DataFrame, others_df: pd.DataFrame) -> float:
    """Jensen-Shannon distance between this bank's stance mix and the other 24 banks'.

    Expected to dominate: the proposal already notes 9% of China's sentences are hawkish
    against 45% of Turkey's, so a model fit to the pooled prior is badly calibrated for
    the extremes.
    """
    p = _label_distribution(bank_df, C.STANCE_COLUMN, C.STANCE_LABELS)
    q = _label_distribution(others_df, C.STANCE_COLUMN, C.STANCE_LABELS)
    return float(jensenshannon(p, q, base=2))


def mean_sentence_length(bank_df: pd.DataFrame) -> float:
    """Mean whitespace-token count (proposal: 17 at the SNB to 34 at the Bank of Korea)."""
    return float(bank_df[C.TEXT_COLUMN].str.split().str.len().mean())


def hedging_rate(bank_df: pd.DataFrame) -> float:
    """Share of sentences annotated `uncertain`.

    Uses the dataset's own certainty labels rather than a hand-built hedge word list --
    the annotation already encodes what counts as hedged in this domain, and it costs
    nothing. This is one of the two places the auxiliary labels earn their keep given
    multi-task learning is out of scope.
    """
    return float((bank_df[C.CERTAIN_COLUMN] == "uncertain").mean())


def forward_looking_rate(bank_df: pd.DataFrame) -> float:
    """Share of sentences annotated `forward looking`."""
    return float((bank_df[C.TIME_COLUMN] == "forward looking").mean())


def _top_vocab(texts: Iterable[str], k: int = 1000) -> set:
    counts = collections.Counter()
    for t in texts:
        counts.update(_WORD.findall(t.lower()))
    return {w for w, _ in counts.most_common(k)}


def vocabulary_novelty(bank_df: pd.DataFrame, others_df: pd.DataFrame, k: int = 1000) -> float:
    """1 - Jaccard overlap between this bank's top-k unigrams and the other banks' top-k.

    Higher means the bank's everyday vocabulary is less represented in what the LOBO model
    actually trained on.
    """
    a = _top_vocab(bank_df[C.TEXT_COLUMN], k)
    b = _top_vocab(others_df[C.TEXT_COLUMN], k)
    if not a or not b:
        return float("nan")
    return 1.0 - len(a & b) / len(a | b)


def embedding_distance(bank_df: pd.DataFrame, others_df: pd.DataFrame,
                       model_name: str = C.MODEL_NAME,
                       batch_size: int = 64,
                       max_samples: int = 700,
                       device: Optional[str] = None) -> float:
    """Cosine distance between mean sentence embeddings, using an *off-the-shelf* encoder.

    Deliberately not the fine-tuned model: this measures how far apart the banks' language
    sits in a representation that knows nothing about our task or our labels, so it stays
    an independent predictor of the gap rather than a restatement of it.

    Imported lazily so the other five features run without torch installed.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    def centroid(texts: List[str]) -> np.ndarray:
        if len(texts) > max_samples:
            rng = np.random.default_rng(0)
            texts = [texts[i] for i in rng.choice(len(texts), max_samples, replace=False)]
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                          max_length=C.DEFAULT_HPARAMS["max_length"], return_tensors="pt").to(device)
                out = model(**enc).last_hidden_state
                # Mean-pool over real tokens only; padding would drag every centroid
                # toward the pad embedding and compress the distances we care about.
                mask = enc["attention_mask"].unsqueeze(-1).float()
                vecs.append(((out * mask).sum(1) / mask.sum(1)).cpu().numpy())
        return np.concatenate(vecs).mean(axis=0)

    a = centroid(bank_df[C.TEXT_COLUMN].tolist())
    b = centroid(others_df[C.TEXT_COLUMN].tolist())
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return 1.0 - cos


def compute_bank_features(frames: Dict[str, pd.DataFrame],
                          banks: Optional[List[str]] = None,
                          include_embeddings: bool = True) -> pd.DataFrame:
    """Feature table with one row per bank, computed from training rows only."""
    banks = banks or C.BANKS
    train = frames["train"]
    rows = []

    for bank in banks:
        bank_df = filter_banks(train, bank)
        others_df = filter_banks(train, bank, exclude=True)

        row = {
            "bank": bank,
            "stance_js_divergence": stance_js_divergence(bank_df, others_df),
            "mean_sentence_length": mean_sentence_length(bank_df),
            "hedging_rate": hedging_rate(bank_df),
            "forward_looking_rate": forward_looking_rate(bank_df),
            "vocabulary_novelty": vocabulary_novelty(bank_df, others_df),
        }
        if include_embeddings:
            try:
                row["embedding_distance"] = embedding_distance(bank_df, others_df)
            except ImportError:
                logger.warning("torch not available -- skipping embedding_distance")
                row["embedding_distance"] = float("nan")
        rows.append(row)
        logger.info("features: %s", bank)

    return pd.DataFrame(rows).set_index("bank")
