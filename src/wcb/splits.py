"""Builders for the three experimental conditions.

    pooled        train on all 25 banks            -> in-distribution ceiling
    lobo(bank)    train on the other 24 banks      -> performance on an unseen bank
    size_control  train on a random subset the     -> separates "bank absent" from
                  size of a LOBO training set         "4% less training data"

All three are evaluated on `evaluation_set(frames, bank)`, which returns the *same* rows
regardless of condition. That identity is what makes the paired bootstrap valid, and it
is asserted rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import config as C
from .data import filter_banks


@dataclass
class Fold:
    """One trainable unit: training rows, model-selection rows, and a name for results."""
    name: str
    condition: str                 # "pooled" | "lobo" | "size_control"
    train: pd.DataFrame
    val: pd.DataFrame
    held_out_bank: Optional[str] = None

    def __repr__(self) -> str:
        return (f"Fold({self.name}, {self.condition}, train={len(self.train)}, "
                f"val={len(self.val)}, held_out={self.held_out_bank})")


def evaluation_set(frames: Dict[str, pd.DataFrame], bank: str) -> pd.DataFrame:
    """The canonical test rows for one bank.

    Every condition is scored on exactly this frame. Sizes vary by bank (128-176) because
    the dataset's 70/15/15 split is global rather than stratified by bank.
    """
    assert bank in C.BANKS, f"unknown bank {bank!r}"
    rows = filter_banks(frames["test"], bank)
    assert len(rows) > 0, f"no test rows for {bank}"
    return rows


def pooled_fold(frames: Dict[str, pd.DataFrame]) -> Fold:
    """Train on everything. This is the WCB paper's 'general setup' and our ceiling."""
    return Fold(name="pooled", condition="pooled",
                train=frames["train"], val=frames["val"])


def lobo_fold(frames: Dict[str, pd.DataFrame], bank: str) -> Fold:
    """Train on the other 24 banks, with the held-out bank removed from val as well.

    Removing it from val matters: selecting a checkpoint on validation data that contains
    the target bank leaks that bank back into the model, which is exactly what the
    experiment claims not to do.
    """
    assert bank in C.BANKS, f"unknown bank {bank!r}"
    train = filter_banks(frames["train"], bank, exclude=True)
    val = filter_banks(frames["val"], bank, exclude=True)

    assert bank not in set(train[C.BANK_COLUMN]), f"{bank} leaked into LOBO train"
    assert bank not in set(val[C.BANK_COLUMN]), f"{bank} leaked into LOBO val"
    assert len(set(train[C.BANK_COLUMN])) == C.N_BANKS - 1

    return Fold(name=f"lobo_{bank}", condition="lobo",
                train=train, val=val, held_out_bank=bank)


def mean_lobo_train_size(frames: Dict[str, pd.DataFrame]) -> int:
    """Average training-set size across the 25 LOBO folds.

    Used to size the control so it is matched to a typical LOBO fold. Per-bank LOBO train
    sizes vary by only a few dozen rows (<0.2% of the total), so one control model is a
    fair comparison for all 25 banks.
    """
    per_bank = frames["train"].groupby(C.BANK_COLUMN).size()
    return int(round(len(frames["train"]) - per_bank.mean()))


def size_control_fold(frames: Dict[str, pd.DataFrame],
                      n_train: Optional[int] = None,
                      rng_seed: int = 0) -> Fold:
    """Train on a random subset of the pooled data matched in size to a LOBO fold.

    The comparison this enables:
        pooled  vs  size_control  -> cost of having less data
        size_control vs lobo      -> cost of that data being one specific bank
    Without it, the transfer gap confounds the two.
    """
    if n_train is None:
        n_train = mean_lobo_train_size(frames)
    assert n_train <= len(frames["train"])

    rng = np.random.default_rng(rng_seed)
    keep = rng.choice(len(frames["train"]), size=n_train, replace=False)
    train = frames["train"].iloc[np.sort(keep)].reset_index(drop=True)

    # A random subset must still contain all 25 banks -- otherwise it is accidentally a
    # LOBO fold and the control is not a control.
    assert len(set(train[C.BANK_COLUMN])) == C.N_BANKS

    return Fold(name=f"size_control_seed{rng_seed}", condition="size_control",
                train=train, val=frames["val"])


def assert_same_evaluation_items(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Guard the paired bootstrap: both conditions must have scored identical rows."""
    ids_a, ids_b = set(a["sentence_id"]), set(b["sentence_id"])
    assert ids_a == ids_b, (
        f"evaluation sets differ: {len(ids_a - ids_b)} only in A, "
        f"{len(ids_b - ids_a)} only in B -- paired bootstrap invalid")
