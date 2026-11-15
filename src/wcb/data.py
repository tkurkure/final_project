"""Loading, cleaning and integrity-checking of the WCB annotated dataset.

The single entry point is `load_annotated(seed)`, which returns train/val/test frames
that have already been deduplicated and given stable sentence ids. Everything downstream
(baseline, training, features, analysis) consumes those frames, so cleaning happens in
exactly one place.
"""
from __future__ import annotations

import logging
from typing import Dict

import pandas as pd
from datasets import load_dataset

from . import config as C

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")

# Deduplication policy, in precedence order:
#   test  -- NEVER modified, not even to drop a duplicate occurring within it. These rows
#            are the evaluation items the pooled and LOBO models are compared on, and they
#            are what the published WCB numbers were computed over. Shrinking test would
#            break both comparability and the paired bootstrap.
#   val   -- rows whose text also appears in test are dropped (leakage into model selection).
#   train -- rows whose text appears in test or val are dropped, plus within-train repeats.
def _normalize_for_dedup(text: str) -> str:
    """Casefold and collapse whitespace. Used ONLY to detect duplicates, never to alter
    the text that reaches a model."""
    return " ".join(text.lower().split())


def load_raw(seed: int = C.PRIMARY_SEED) -> Dict[str, pd.DataFrame]:
    """Load one split-seed config of the annotated dataset as pandas frames, untouched."""
    ds = load_dataset(C.ANNOTATED_DATASET, name=str(seed))
    return {s: ds[s].to_pandas() for s in SPLITS}


def load_annotated(seed: int = C.PRIMARY_SEED, verify: bool = True) -> Dict[str, pd.DataFrame]:
    """Load, verify and deduplicate one split-seed config.

    Returns frames with added columns:
      sentence_id  stable id, unique across all three splits
      split        which split the row came from
      stance_id    integer label matching config.STANCE_LABELS ordering
    """
    frames = load_raw(seed)

    for split, df in frames.items():
        df["split"] = split
        # __index_level_0__ is the row's index in the authors' original dataframe, so it
        # is stable across splits and across the three seed configs -- exactly what we
        # need to assert that two models scored the same items.
        df["sentence_id"] = df[C.INDEX_COLUMN].astype(int)
        df["stance_id"] = df[C.STANCE_COLUMN].map(C.STANCE2ID)

    if verify:
        verify_integrity(frames)

    return deduplicate(frames)


def verify_integrity(frames: Dict[str, pd.DataFrame]) -> None:
    """Gate 0 as assertions. Fails loudly rather than silently producing wrong numbers."""
    combined = pd.concat(frames.values(), ignore_index=True)

    # Label vocabulary: an unmapped stance would become NaN and silently poison training.
    unknown = set(combined[C.STANCE_COLUMN]) - set(C.STANCE_LABELS)
    assert not unknown, f"unexpected stance labels: {unknown}"
    assert combined["stance_id"].notna().all(), "stance_id has NaNs -- label map is stale"

    # Bank vocabulary: catches the README-slug-vs-column-value trap.
    unknown_banks = set(combined[C.BANK_COLUMN]) - set(C.BANKS)
    assert not unknown_banks, f"unexpected bank_name values: {unknown_banks}"
    assert len(set(combined[C.BANK_COLUMN])) == C.N_BANKS

    # sentence_id must identify a row uniquely, or the identical-test-set assertion in
    # the LOBO runner is meaningless.
    assert combined["sentence_id"].is_unique, "sentence_id is not unique across splits"

    assert len(combined) == 25_000, f"expected 25,000 sentences, got {len(combined)}"

    # Every bank contributes exactly 1,000 sentences. NOTE: the split is global 70/15/15,
    # NOT stratified by bank, so per-bank TEST counts legitimately vary (128-176). Only
    # the per-bank total is guaranteed.
    per_bank = combined.groupby(C.BANK_COLUMN).size()
    assert (per_bank == 1000).all(), f"banks without 1000 sentences:\n{per_bank[per_bank != 1000]}"


def deduplicate(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Remove leakage between splits without ever touching the test split.

    Duplicates are matched on casefolded, whitespace-collapsed text; the original text is
    what reaches the model.
    """
    out = {s: frames[s].copy() for s in SPLITS}
    for df in out.values():
        df["_norm"] = df[C.TEXT_COLUMN].map(_normalize_for_dedup)

    test_texts = set(out["test"]["_norm"])

    # val: drop anything that also appears in test.
    n_val_before = len(out["val"])
    out["val"] = out["val"][~out["val"]["_norm"].isin(test_texts)]
    val_texts = set(out["val"]["_norm"])

    # train: drop anything appearing in test or val, then collapse within-train repeats.
    n_train_before = len(out["train"])
    out["train"] = out["train"][~out["train"]["_norm"].isin(test_texts | val_texts)]
    out["train"] = out["train"].drop_duplicates(subset="_norm", keep="first")

    out = {s: df.drop(columns="_norm").reset_index(drop=True) for s, df in out.items()}

    logger.info(
        "dedup: train %d->%d (-%d), val %d->%d (-%d), test %d (untouched)",
        n_train_before, len(out["train"]), n_train_before - len(out["train"]),
        n_val_before, len(out["val"]), n_val_before - len(out["val"]),
        len(out["test"]),
    )

    assert len(out["test"]) == len(frames["test"]), "test split must never shrink"
    assert not (set(out["train"][C.TEXT_COLUMN].map(_normalize_for_dedup)) & test_texts)
    assert not (set(out["val"][C.TEXT_COLUMN].map(_normalize_for_dedup)) & test_texts)
    return out


def filter_banks(df: pd.DataFrame, banks, exclude: bool = False) -> pd.DataFrame:
    """Keep (or drop) rows belonging to `banks`."""
    if isinstance(banks, str):
        banks = [banks]
    mask = df[C.BANK_COLUMN].isin(banks)
    return df[~mask if exclude else mask].reset_index(drop=True)


def bank_test_sizes(frames: Dict[str, pd.DataFrame]) -> pd.Series:
    """Per-bank test-set size. Varies by bank because the split is not stratified."""
    return frames["test"].groupby(C.BANK_COLUMN).size().sort_values()


# ---------------------------------------------------------------------------------
# Czech National Bank hold-out
# ---------------------------------------------------------------------------------
# The 26th bank: 500 annotated sentences, never part of the 25-bank training data. The
# WCB paper reports a mean weighted F1 of 0.800 here for a model trained on all 25, which
# makes it an independent check that our pipeline is wired up correctly -- and the one
# published generalization result we can reproduce directly.
CZECH_DATASET = "gtfintechlab/czech_national_bank"
CZECH_BANK_NAME = "czech_national_bank"


def load_czech(seed: int = C.PRIMARY_SEED) -> pd.DataFrame:
    """All 500 Czech sentences as one frame.

    The splits are concatenated deliberately: no model here ever trains on Czech data,
    so the whole set is held out and the paper's figure is computed over all 500.
    """
    ds = load_dataset(CZECH_DATASET, name=str(seed))
    df = pd.concat([ds[s].to_pandas() for s in ds], ignore_index=True)

    df["sentence_id"] = df[C.INDEX_COLUMN].astype(int)
    df["stance_id"] = df[C.STANCE_COLUMN].map(C.STANCE2ID)
    df[C.BANK_COLUMN] = CZECH_BANK_NAME
    df["split"] = "czech_holdout"

    unknown = set(df[C.STANCE_COLUMN]) - set(C.STANCE_LABELS)
    assert not unknown, f"unexpected Czech stance labels: {unknown}"
    assert df["stance_id"].notna().all()
    assert len(df) == 500, f"expected 500 Czech sentences, got {len(df)}"
    return df
