#!/usr/bin/env python
"""Regenerate every table and figure from `results/` alone.

    python scripts/make_report.py                 # baseline
    python scripts/make_report.py --model roberta

Nothing here retrains anything: it reads the saved per-fold predictions, so the whole
report rebuilds in seconds and the figures can never drift from the numbers.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wcb import analysis, config as C, data, features, results_io
from wcb.logging_utils import setup_logging

log = logging.getLogger("make_report")

# --- palette ------------------------------------------------------------------------
# From the validated reference palette. Figures are light-mode only: they are destined
# for a printed writeup, not a theme-aware web page.
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8a85"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    # Recessive frame: keep the data the most prominent thing on the canvas.
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "figure.dpi": 150,
})


def pretty(bank: str) -> str:
    return C.BANK_DISPLAY_NAMES.get(bank, bank)


def fig_transfer_gap(summary: pd.DataFrame, out: Path, model: str) -> None:
    """Per-bank transfer gap. Signed quantity with a meaningful zero -> diverging blue/red."""
    d = summary.sort_values("gap_total")
    colors = [BLUE if v >= 0 else RED for v in d.gap_total]

    fig, ax = plt.subplots(figsize=(7.5, 8))
    y = np.arange(len(d))
    ax.barh(y, d.gap_total, color=colors, height=0.68)

    # Split error bars are the spread across the three dataset partitions, not a CI.
    if "gap_total_std" in d:
        ax.errorbar(d.gap_total, y, xerr=d.gap_total_std.fillna(0), fmt="none",
                    ecolor=INK_MUTED, elinewidth=0.9, capsize=2.5)

    ax.set_yticks(y, [pretty(b) for b in d.index])
    ax.axvline(0, color=INK_2, linewidth=0.9)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("transfer gap  (pooled weighted F1 − leave-one-bank-out weighted F1)")
    ax.set_title(f"How much worse is a bank the model never saw?   [{model}]", loc="left", pad=12)

    # Direct-label every bar: 25 values is few enough to read, and it removes any need
    # to measure against the axis. Labels clear the whisker rather than overstriking it.
    err = d.gap_total_std.fillna(0) if "gap_total_std" in d else pd.Series(0.0, index=d.index)
    span = (d.gap_total + err).max() - min(0, (d.gap_total - err).min())
    pad = 0.018 * span
    for yi, v, e in zip(y, d.gap_total, err):
        tip = v + e if v >= 0 else v - e
        ax.text(tip + (pad if v >= 0 else -pad), yi, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=INK_2)
    ax.set_xlim(min(0, (d.gap_total - err).min()) - 0.14 * span,
                (d.gap_total + err).max() + 0.14 * span)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


def fig_gap_decomposition(summary: pd.DataFrame, out: Path, model: str) -> None:
    """Does the gap come from less data, or from *which* data is missing?"""
    d = summary.sort_values("gap_bank")
    y = np.arange(len(d))
    h = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    # Grouped, not stacked: gap_data goes slightly negative for some banks, and a stack
    # with negative segments is unreadable.
    ax.barh(y + h / 2, d.gap_bank, height=h, color=BLUE, label="missing bank (size-control − LOBO)")
    ax.barh(y - h / 2, d.gap_data, height=h, color=ORANGE, label="less data (pooled − size-control)")

    ax.set_yticks(y, [pretty(b) for b in d.index])
    ax.axvline(0, color=INK_2, linewidth=0.9)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("contribution to the transfer gap (weighted F1)")
    ax.legend(loc="lower right", fontsize=8)

    share = d.gap_bank.mean() / d.gap_total.mean()
    ax.set_title(
        f"Almost none of the gap is a training-set-size effect   [{model}]\n"
        f"{share:.0%} of the mean gap is attributable to which bank is absent",
        loc="left", pad=12, fontsize=10)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


def fig_feature_scatter(summary: pd.DataFrame, feat: pd.DataFrame, corr: pd.DataFrame,
                        out: Path, model: str) -> None:
    """Each pre-registered feature against the bank-specific gap, on RANK axes.

    Rank-rank is the faithful picture of the statistic actually being reported: Spearman
    is a rank correlation, so a rank scatter shows exactly what it measures. It also
    fixes a real legibility problem -- one bank's gap is ~6x the next largest, and on raw
    axes it squashes the other 24 banks into an unreadable band along the bottom.
    """
    feats = [f for f in features.FEATURE_NAMES if f in feat.columns
             and feat[f].notna().any()]
    df = summary.join(feat, how="inner")
    rho = dict(zip(corr.feature, corr.spearman_rho))
    top = df.gap_bank.idxmax()

    gap_rank = df.gap_bank.rank()
    n = len(df)

    ncol = 3
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3.3 * nrow), squeeze=False)

    for ax, f in zip(axes.ravel(), feats):
        xr = df[f].rank()
        ax.scatter(xr, gap_rank, s=34, color=BLUE, alpha=0.85,
                   edgecolor=SURFACE, linewidth=1.2)   # surface ring on overlapping marks

        # Highlight the extreme bank with a second encoding (ring), not colour alone.
        ax.scatter([xr[top]], [gap_rank[top]], s=96, facecolor="none",
                   edgecolor=INK_2, linewidth=1.1)
        # The top-gap bank always sits on the top row, so the label goes below it, and on
        # whichever side has room -- toward the panel centre, never across its edge.
        left_half = xr[top] <= n / 2
        ax.annotate(pretty(top), (xr[top], gap_rank[top]),
                    textcoords="offset points", xytext=(9 if left_half else -9, -11),
                    fontsize=7.5, color=INK_MUTED, ha="left" if left_half else "right",
                    va="top")

        ax.grid(True)
        ax.set_axisbelow(True)
        # Slack at the edges so the highlight ring on a rank-1 or rank-n point is not clipped.
        ax.set_xlim(0.0, n + 1.0)
        ax.set_ylim(0.0, n + 1.0)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{f.replace('_', ' ')}  (rank)")
        ax.set_ylabel("bank-specific gap (rank)")
        r = rho.get(f, np.nan)
        ax.set_title(f"\u03c1 = {r:+.2f}", loc="left", fontsize=10)

    for ax in axes.ravel()[len(feats):]:
        ax.set_visible(False)

    fig.suptitle(
        f"What predicts the bank-specific gap?  n = 25 banks \u2014 descriptive, not inferential   [{model}]",
        x=0.012, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


def fig_per_class(model: str, seed: int, out: Path) -> None:
    """Where does the drop land? Per-class F1 under each condition."""
    d = analysis.error_breakdown_by_class(model, seed)
    if d.empty:
        return
    piv = d.groupby(["label", "condition"]).f1.mean().unstack()
    order = [l for l in C.STANCE_LABELS if l in piv.index]
    piv = piv.loc[order]
    conds = [c for c in ("pooled", "size_control", "lobo") if c in piv.columns]
    colors = {"pooled": BLUE, "size_control": ORANGE, "lobo": AQUA}

    x = np.arange(len(piv))
    w = 0.8 / len(conds)
    fig, ax = plt.subplots(figsize=(7.5, 4))

    for i, c in enumerate(conds):
        off = (i - (len(conds) - 1) / 2) * w
        bars = ax.bar(x + off, piv[c], width=w * 0.92, color=colors[c], label=c)
        # Relief rule: aqua sits below 3:1 on this surface, so values are labeled directly.
        for b, v in zip(bars, piv[c]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", fontsize=7, color=INK_2)

    ax.set_xticks(x, order)
    ax.set_ylabel("mean per-class F1 across 25 banks")
    ax.set_ylim(0, min(1.0, piv.values.max() * 1.18))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(ncol=len(conds), fontsize=8, loc="upper right")
    ax.set_title(f"Which label does the drop land on?   [{model}, seed {seed}]", loc="left", pad=10)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="baseline")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--n-boot", type=int, default=C.N_BOOTSTRAP)
    args = ap.parse_args()

    setup_logging()
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model = args.model

    available = results_io.load_all(model)
    if available.empty:
        log.error("no results found under %s -- run the sweep first", C.RESULTS_DIR / model)
        sys.exit(1)
    seeds = args.seeds or sorted(available.seed.unique())
    log.info("model=%s  seeds=%s  folds=%d", model, seeds, len(available))

    # --- tables ----------------------------------------------------------------------
    gaps = analysis.transfer_gap_table(model, seeds=seeds, n_boot=args.n_boot)
    gaps.to_csv(C.RESULTS_DIR / f"{model}_gap_by_seed.csv", index=False)
    summary = analysis.average_over_seeds(gaps)
    summary.to_csv(C.RESULTS_DIR / f"{model}_gap_summary.csv")

    frames = data.load_annotated(C.PRIMARY_SEED)
    feat_path = C.RESULTS_DIR / "bank_features.csv"
    feat = pd.read_csv(feat_path, index_col=0) if feat_path.exists() else None
    # A cache written before a feature existed would otherwise drop it from the report
    # without a word -- and the pre-registration promises the full set.
    if feat is None or set(features.FEATURE_NAMES) - set(feat.columns):
        feat = features.compute_bank_features(frames)
        feat.to_csv(feat_path)
    feat = feat.dropna(axis=1, how="all")

    # gap_bank, not gap_total: the question is what makes a bank hard to transfer to, and
    # gap_total still carries the (small, noisy) less-data component.
    corr = analysis.correlate_features_with_gap(summary, feat, "gap_bank", n_boot=args.n_boot)
    corr.to_csv(C.RESULTS_DIR / f"{model}_feature_correlations.csv", index=False)

    errors = analysis.sample_lobo_errors(model, seeds[0], frames, n=50)
    errors.to_csv(C.RESULTS_DIR / f"{model}_error_sample.csv", index=False)

    # --- figures ---------------------------------------------------------------------
    fig_transfer_gap(summary, C.FIGURES_DIR / f"{model}_fig1_transfer_gap.png", model)
    fig_gap_decomposition(summary, C.FIGURES_DIR / f"{model}_fig2_gap_decomposition.png", model)
    fig_feature_scatter(summary, feat, corr, C.FIGURES_DIR / f"{model}_fig3_features.png", model)
    fig_per_class(model, seeds[0], C.FIGURES_DIR / f"{model}_fig4_per_class_f1.png")

    # --- console summary --------------------------------------------------------------
    print(f"\n=== {model}: headline numbers ===")
    print(f"  pooled weighted F1 (mean over banks) : {summary.pooled_weighted_f1.mean():.4f}")
    print(f"  LOBO   weighted F1 (mean over banks) : {summary.lobo_weighted_f1.mean():.4f}")
    print(f"  transfer gap                         : {summary.gap_total.mean():+.4f}")
    print(f"    of which, less training data       : {summary.gap_data.mean():+.4f}")
    print(f"    of which, the bank being absent    : {summary.gap_bank.mean():+.4f}")
    print(f"  banks with a positive gap            : {(summary.gap_total > 0).sum()}/{len(summary)}")
    print(f"\n  largest gap: {pretty(summary.gap_total.idxmax())} "
          f"({summary.gap_total.max():+.4f})")
    print("\n=== feature correlations with gap_bank (n=25) ===")
    print(corr.round(4).to_string(index=False))
    print(f"\nfigures -> {C.FIGURES_DIR}\ntables  -> {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
