"""Transfer-gap tables, the gap decomposition, and feature correlations."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as C
from . import results_io
from .evaluate import METRICS, paired_bootstrap_gap

logger = logging.getLogger(__name__)

# The decomposition the size control buys us:
#   gap_total = pooled - lobo            everything the held-out bank's absence costs
#   gap_data  = pooled - size_control    the part explained purely by ~4% less training data
#   gap_bank  = size_control - lobo      the part attributable to *which* data is missing
# Only gap_bank supports the claim the project is actually making.
GAP_COMPONENTS = {
    "gap_total": ("pooled", "lobo"),
    "gap_data": ("pooled", "size_control"),
    "gap_bank": ("size_control", "lobo"),
}


def _aligned(model: str, seed: int, cond_a: str, cond_b: str, bank: str):
    """Load two conditions' predictions for one bank, aligned row-for-row by sentence_id.

    Returns None if either run is missing, so partially-complete sweeps still tabulate.
    """
    ra = results_io.load_result(model, seed, cond_a, bank)
    rb = results_io.load_result(model, seed, cond_b, bank)
    if ra is None or rb is None:
        return None

    da = pd.DataFrame({"sid": ra["sentence_id"], "y": ra["y_true"], "p": ra["y_pred"]})
    db = pd.DataFrame({"sid": rb["sentence_id"], "y": rb["y_true"], "p": rb["y_pred"]})

    assert set(da.sid) == set(db.sid), (
        f"{bank}: {cond_a} and {cond_b} scored different items -- paired bootstrap invalid")

    merged = da.merge(db, on="sid", suffixes=("_a", "_b"))
    assert (merged.y_a == merged.y_b).all(), f"{bank}: gold labels disagree across runs"
    return merged


def transfer_gap_table(model: str,
                       seeds: Optional[List[int]] = None,
                       banks: Optional[List[str]] = None,
                       metric: str = "weighted_f1",
                       n_boot: int = 2000) -> pd.DataFrame:
    """One row per (seed, bank): each condition's score, plus the three gap components
    with paired-bootstrap confidence intervals."""
    seeds = seeds or C.SPLIT_SEEDS
    banks = banks or C.BANKS
    fn = METRICS[metric]
    rows = []

    for seed in seeds:
        for bank in banks:
            row = {"seed": seed, "bank": bank}

            for cond in ("pooled", "lobo", "size_control"):
                r = results_io.load_result(model, seed, cond, bank)
                if r is not None:
                    row[f"{cond}_{metric}"] = fn(np.array(r["y_true"]), np.array(r["y_pred"]))
                    row["n_test"] = r["metrics"]["n"]

            for gap_name, (ca, cb) in GAP_COMPONENTS.items():
                m = _aligned(model, seed, ca, cb, bank)
                if m is None:
                    continue
                g = paired_bootstrap_gap(m.y_a.values, m.p_a.values, m.p_b.values,
                                         metric=metric, n_boot=n_boot)
                row[gap_name] = g["gap"]
                row[f"{gap_name}_lo"] = g["lo"]
                row[f"{gap_name}_hi"] = g["hi"]
                row[f"{gap_name}_p"] = g["p_two_sided"]

            if len(row) > 2:
                rows.append(row)

    return pd.DataFrame(rows)


def average_over_seeds(gap_table: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per bank: mean across split seeds, with the spread.

    The three dataset configs are three different partitions of the same 25,000
    sentences, so their spread is a genuine estimate of split sensitivity -- which
    matters when a per-bank test set holds only 128-176 items.
    """
    value_cols = [c for c in gap_table.columns
                  if c not in ("seed", "bank") and not c.endswith(("_lo", "_hi", "_p"))]
    agg = gap_table.groupby("bank")[value_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" if b != "mean" else a for a, b in agg.columns]
    sort_col = "gap_total" if "gap_total" in agg.columns else agg.columns[0]
    return agg.sort_values(sort_col, ascending=False)


def correlate_features_with_gap(gap_summary: pd.DataFrame,
                                feature_table: pd.DataFrame,
                                gap_column: str = "gap_total",
                                n_boot: int = 10_000,
                                rng_seed: int = 0) -> pd.DataFrame:
    """Spearman correlation between each pre-registered feature and the transfer gap.

    Reported with a bootstrap CI over banks and WITHOUT multiple-comparison ceremony,
    because with n = 25 this is descriptive: the CIs will be wide, and they are meant to
    be read as "how little we can tell from 25 points", not as significance tests. The
    full fixed feature set is always reported, including flat features.
    """
    df = gap_summary.join(feature_table, how="inner")
    y = df[gap_column].values
    rng = np.random.default_rng(rng_seed)
    n = len(df)
    rows = []

    for feat in feature_table.columns:
        x = df[feat].values
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 5:
            continue
        rho, p = spearmanr(x[ok], y[ok])

        boot = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            xb, yb = x[idx], y[idx]
            m = ~(np.isnan(xb) | np.isnan(yb))
            boot[i] = (spearmanr(xb[m], yb[m])[0]
                       if m.sum() >= 5 and len(set(xb[m])) > 1 else np.nan)

        boot = boot[~np.isnan(boot)]
        rows.append({
            "feature": feat, "spearman_rho": rho, "p_value": p, "n_banks": int(ok.sum()),
            "ci_lo": float(np.percentile(boot, 2.5)) if len(boot) else np.nan,
            "ci_hi": float(np.percentile(boot, 97.5)) if len(boot) else np.nan,
        })

    return (pd.DataFrame(rows)
            .assign(abs_rho=lambda d: d.spearman_rho.abs())
            .sort_values("abs_rho", ascending=False)
            .drop(columns="abs_rho")
            .reset_index(drop=True))


def error_breakdown_by_class(model: str, seed: int,
                             banks: Optional[List[str]] = None) -> pd.DataFrame:
    """Per-class F1 under each condition, to answer whether the drop lands on one label."""
    banks = banks or C.BANKS
    rows = []
    for bank in banks:
        for cond in ("pooled", "lobo", "size_control"):
            r = results_io.load_result(model, seed, cond, bank)
            if r is None:
                continue
            for label, f1 in r["metrics"]["per_class_f1"].items():
                rows.append({"bank": bank, "condition": cond, "label": label,
                             "f1": f1, "support": r["metrics"]["support"][label]})
    return pd.DataFrame(rows)


def sample_lobo_errors(model: str, seed: int, frames: Dict[str, pd.DataFrame],
                       n: int = 50, rng_seed: int = 0) -> pd.DataFrame:
    """Stratified sample of LOBO errors for hand-coding.

    Prefers errors the pooled model got RIGHT: those are the ones actually caused by the
    bank being unseen, rather than sentences that are simply hard for every model.
    """
    rows = []
    for bank in C.BANKS:
        lobo = results_io.load_result(model, seed, "lobo", bank)
        pooled = results_io.load_result(model, seed, "pooled", bank)
        if lobo is None:
            continue
        df = pd.DataFrame({"sentence_id": lobo["sentence_id"],
                           "y_true": lobo["y_true"], "lobo_pred": lobo["y_pred"]})
        if pooled is not None:
            pm = pd.DataFrame({"sentence_id": pooled["sentence_id"],
                               "pooled_pred": pooled["y_pred"]})
            df = df.merge(pm, on="sentence_id")
        else:
            df["pooled_pred"] = np.nan
        df["bank"] = bank
        rows.append(df[df.y_true != df.lobo_pred])

    errors = pd.concat(rows, ignore_index=True)
    errors["pooled_correct"] = errors.y_true == errors.pooled_pred

    text = pd.concat(frames.values(), ignore_index=True)[["sentence_id", C.TEXT_COLUMN]]
    errors = errors.merge(text, on="sentence_id", how="left")

    per_bank = max(1, n // C.N_BANKS)
    errors["_order"] = np.where(errors.pooled_correct, 0, 1)
    sample = (errors.sort_values("_order")
                    .groupby("bank", group_keys=False)
                    .head(per_bank)
                    .sample(frac=1.0, random_state=rng_seed)
                    .head(n))

    sample["true_label"] = sample.y_true.map(C.ID2STANCE)
    sample["lobo_label"] = sample.lobo_pred.map(C.ID2STANCE)
    sample["pooled_label"] = sample.pooled_pred.map(
        lambda i: C.ID2STANCE.get(i) if pd.notna(i) else None)

    # Blank column for the hand-coded taxonomy: new vocabulary / bank-specific
    # convention / genuinely ambiguous.
    sample["error_category"] = ""
    return sample.drop(columns="_order").reset_index(drop=True)
