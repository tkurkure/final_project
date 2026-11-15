#!/usr/bin/env python
"""Is the bank-specific gap just a shifted stance mix?

    python scripts/prior_shift.py                     # baseline and roberta
    python scripts/prior_shift.py --models roberta

A classifier trained on the other 24 banks has their stance mix baked into its output
probabilities. If the held-out bank is unusually dovish or hawkish, the model is
miscalibrated for it even if it reads every sentence well. Stance-mix divergence is the
strongest predictor of the gap, so this asks how much of the gap that miscalibration
alone accounts for -- versus something about the bank's language.

The correction is the standard label-shift adjustment (Saerens et al., 2002): multiply
each predicted distribution by p_target(y) / p_train(y) and renormalise. Two targets:

  oracle  the held-out bank's own TRAINING-row stance mix. Never its test labels, so the
          scored items stay untouched; it answers "what if we knew this bank's usual mix?"
  em      estimated from the unlabelled test sentences alone, by EM. Uses no labels at
          all, so it is what someone deploying on a genuinely new bank could actually do.

Every condition is corrected against its OWN training prior -- pooled and size control
against the full training mix, LOBO against the other 24 banks' -- so the corrected gap
still compares like with like. Nothing is retrained: this reads the saved probabilities.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from wcb import analysis, config as C, data, results_io
from wcb.data import filter_banks
from wcb.evaluate import weighted_f1
from wcb.logging_utils import setup_logging

# Shares the report's palette and rcParams, so this figure sits beside fig1-4 unchanged.
from make_report import BLUE, ORANGE, INK_2, INK_MUTED, pretty

log = logging.getLogger("prior_shift")

CONDITIONS = ("pooled", "size_control", "lobo")
VARIANTS = ("raw", "oracle", "em")


def stance_prior(df: pd.DataFrame) -> np.ndarray:
    """Add-one smoothed stance mix. Smoothing matters: a bank with no `irrelevant` training
    rows would otherwise get a zero prior and that class could never be predicted."""
    counts = np.bincount(df["stance_id"].astype(int), minlength=len(C.STANCE_LABELS))
    return (counts + 1) / (counts.sum() + len(counts))


def reweight(probs: np.ndarray, train_prior: np.ndarray, target_prior: np.ndarray) -> np.ndarray:
    adj = probs * (target_prior / train_prior)
    return adj / adj.sum(axis=1, keepdims=True)


def em_prior(probs: np.ndarray, train_prior: np.ndarray,
             n_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """Saerens-Latinne-Decaestecker EM: alternate re-weighting the posteriors and
    re-estimating the prior from their mean, until the prior stops moving."""
    prior = train_prior.copy()
    for _ in range(n_iter):
        new = reweight(probs, train_prior, prior).mean(axis=0)
        if np.abs(new - prior).max() < tol:
            return new
        prior = new
    return prior


def score_model(model: str, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        frames = data.load_annotated(seed)
        train = frames["train"]
        full_prior = stance_prior(train)

        for bank in C.BANKS:
            bank_prior = stance_prior(filter_banks(train, bank))
            train_priors = {"pooled": full_prior, "size_control": full_prior,
                            "lobo": stance_prior(filter_banks(train, bank, exclude=True))}
            row = {"seed": seed, "bank": bank}

            for cond in CONDITIONS:
                r = results_io.load_result(model, seed, cond, bank)
                if r is None:
                    continue
                y = np.array(r["y_true"])
                probs = np.array(r["probs"])
                src = train_priors[cond]
                preds = {
                    "raw": probs.argmax(axis=1),
                    "oracle": reweight(probs, src, bank_prior).argmax(axis=1),
                    "em": reweight(probs, src, em_prior(probs, src)).argmax(axis=1),
                }
                # The saved y_pred is the argmax of the saved probs; if these disagree the
                # probabilities are not what the headline numbers were computed from.
                assert (preds["raw"] == np.array(r["y_pred"])).all(), (model, seed, cond, bank)
                for v, p in preds.items():
                    row[f"{cond}_{v}"] = weighted_f1(y, p)

            for v in VARIANTS:
                row[f"gap_total_{v}"] = row[f"pooled_{v}"] - row[f"lobo_{v}"]
                row[f"gap_bank_{v}"] = row[f"size_control_{v}"] - row[f"lobo_{v}"]
            rows.append(row)
    return pd.DataFrame(rows)


def fig_prior_shift(summary: pd.DataFrame, out: Path, model: str) -> None:
    """Per-bank gap before and after telling the LOBO model the bank's stance mix."""
    d = summary.sort_values("gap_bank_raw")
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(7.5, 8))
    ax.hlines(y, d.gap_bank_raw, d.gap_bank_oracle, color=INK_MUTED, linewidth=1.1, zorder=1)
    ax.scatter(d.gap_bank_raw, y, s=36, color=BLUE, zorder=2, label="as trained")
    ax.scatter(d.gap_bank_oracle, y, s=36, color=ORANGE, zorder=2,
               label="after correcting for the bank's stance mix")

    ax.set_yticks(y, [pretty(b) for b in d.index])
    ax.axvline(0, color=INK_2, linewidth=0.9)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("bank-specific gap  (size-control − LOBO weighted F1)")
    ax.legend(loc="lower right", fontsize=8)

    raw, cor = d.gap_bank_raw.mean(), d.gap_bank_oracle.mean()
    ax.set_title(
        f"How much of the gap is just the wrong stance mix?   [{model}]\n"
        f"mean gap {raw:+.3f} → {cor:+.3f}  ({1 - cor / raw:.0%} closed)",
        loc="left", pad=12, fontsize=10)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["baseline", "roberta"])
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SPLIT_SEEDS)
    ap.add_argument("--n-boot", type=int, default=C.N_BOOTSTRAP)
    args = ap.parse_args()
    setup_logging()

    feat = pd.read_csv(C.RESULTS_DIR / "bank_features.csv", index_col=0).dropna(axis=1, how="all")

    for model in args.models:
        by_seed = score_model(model, args.seeds)
        by_seed.to_csv(C.RESULTS_DIR / f"{model}_prior_shift_by_seed.csv", index=False)
        summary = by_seed.drop(columns="seed").groupby("bank").mean()
        summary.to_csv(C.RESULTS_DIR / f"{model}_prior_shift_summary.csv")

        corr = pd.concat([
            analysis.correlate_features_with_gap(summary, feat, f"gap_bank_{v}", n_boot=args.n_boot)
            .assign(gap=f"gap_bank_{v}") for v in VARIANTS])
        corr.to_csv(C.RESULTS_DIR / f"{model}_prior_shift_correlations.csv", index=False)

        fig_prior_shift(summary, C.FIGURES_DIR / f"{model}_fig5_prior_shift.png", model)

        print(f"\n=== {model}: label-shift correction (mean over {len(summary)} banks) ===")
        print(f"{'':<14}{'raw':>9}{'oracle':>9}{'em':>9}")
        for cond in CONDITIONS:
            print(f"{cond + ' F1':<14}" + "".join(f"{summary[f'{cond}_{v}'].mean():>9.4f}" for v in VARIANTS))
        for g in ("gap_total", "gap_bank"):
            print(f"{g:<14}" + "".join(f"{summary[f'{g}_{v}'].mean():>+9.4f}" for v in VARIANTS))
        for v in ("oracle", "em"):
            raw, cor = summary.gap_bank_raw.mean(), summary[f"gap_bank_{v}"].mean()
            p = wilcoxon(summary[f"gap_bank_{v}"]).pvalue
            print(f"  {v}: closes {1 - cor / raw:.0%} of gap_bank; "
                  f"{(summary[f'gap_bank_{v}'] > 0).sum()}/25 banks still positive "
                  f"(Wilcoxon vs 0: p={p:.2g})")
        print("\nper-bank gap_bank, largest first:")
        print(summary.sort_values("gap_bank_raw", ascending=False)
              [[f"gap_bank_{v}" for v in VARIANTS]].head(8).round(3).to_string())
        print("\nfeature correlations (Spearman) before and after correction:")
        print(corr.pivot(index="feature", columns="gap", values="spearman_rho").round(3).to_string())


if __name__ == "__main__":
    main()
