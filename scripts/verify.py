#!/usr/bin/env python
"""Run every correctness check the project depends on.

    python scripts/verify.py                  # structural checks (no GPU needed)
    python scripts/verify.py --model roberta  # adds the two reproduction checkpoints

Checks that need a trained model are reported as SKIP until that model exists, so this
is runnable at any point and the output doubles as a status board.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from wcb import analysis, config as C, data, evaluate as E, results_io, splits
from wcb.logging_utils import setup_logging

log = logging.getLogger("verify")

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"
results: list[tuple[str, str, str]] = []


def check(name: str):
    """Decorator turning a function into a reported check.

    A check returns (status, detail); an assertion failure inside it becomes a FAIL with
    the assertion message, so checks read as plain assertions.
    """
    def wrap(fn):
        def run(*a, **kw):
            try:
                status, detail = fn(*a, **kw)
            except AssertionError as exc:
                status, detail = FAIL, str(exc) or "assertion failed"
            except Exception as exc:                      # noqa: BLE001
                status, detail = FAIL, f"{type(exc).__name__}: {exc}"
            results.append((name, status, detail))
            log.info("[%s] %-38s %s", status, name, detail)
            return status
        return run
    return wrap


# --- structural -----------------------------------------------------------------------
@check("gate0: data integrity")
def check_integrity(frames):
    combined = __import__("pandas").concat(frames.values())
    assert len(combined) == 25_000 - 33, f"unexpected total after dedup: {len(combined)}"
    assert combined["sentence_id"].is_unique
    return PASS, f"{len(combined)} rows, ids unique, labels and banks recognised"


@check("gate0: test split untouched")
def check_test_untouched(frames):
    assert len(frames["test"]) == 3750, f"test split is {len(frames['test'])}, expected 3750"
    sizes = frames["test"].groupby(C.BANK_COLUMN).size()
    assert len(sizes) == C.N_BANKS
    return PASS, f"3750 rows, {sizes.min()}-{sizes.max()} per bank (split is not stratified)"


@check("no leakage into LOBO folds")
def check_leakage(frames):
    for bank in C.BANKS:
        fold = splits.lobo_fold(frames, bank)
        assert bank not in set(fold.train[C.BANK_COLUMN]), f"{bank} in LOBO train"
        assert bank not in set(fold.val[C.BANK_COLUMN]), f"{bank} in LOBO val"
    return PASS, f"all {C.N_BANKS} folds exclude the held-out bank from train AND val"


@check("size control contains every bank")
def check_size_control(frames):
    fold = splits.size_control_fold(frames)
    assert len(set(fold.train[C.BANK_COLUMN])) == C.N_BANKS
    lobo_sizes = [len(splits.lobo_fold(frames, b).train) for b in C.BANKS]
    drift = max(abs(len(fold.train) - s) for s in lobo_sizes) / len(frames["train"])
    assert drift < 0.01, f"control size differs from LOBO folds by {drift:.1%}"
    return PASS, f"n={len(fold.train)}, within {drift:.2%} of every LOBO fold"


@check("fast metrics match scikit-learn")
def check_fastmetrics():
    from wcb.fastmetrics import _selftest
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        _selftest()
    return PASS, "agreement < 1e-10 across 200 random problems"


@check("paired bootstrap tighter than unpaired")
def check_paired():
    rng = np.random.default_rng(0)
    n = 150
    y = rng.integers(0, 4, n)
    a = np.where(rng.random(n) < 0.75, y, rng.integers(0, 4, n))
    b = np.where(rng.random(n) < 0.85, a, rng.integers(0, 4, n))
    g = E.paired_bootstrap_gap(y, a, b, n_boot=4000)
    ca, cb = E.bootstrap_ci(y, a, n_boot=4000), E.bootstrap_ci(y, b, n_boot=4000)
    indep = (ca["se"] ** 2 + cb["se"] ** 2) ** 0.5
    assert g["se"] < indep, "paired interval is not tighter -- pairing is not working"
    return PASS, f"paired se {g['se']:.4f} vs unpaired {indep:.4f} ({indep/g['se']:.1f}x)"


# --- result-dependent -------------------------------------------------------------------
@check("identical evaluation items across conditions")
def check_same_items(model, seed):
    n = 0
    for bank in C.BANKS:
        for a, b in analysis.GAP_COMPONENTS.values():
            if analysis._aligned(model, seed, a, b, bank) is not None:
                n += 1
    assert n > 0, "no comparable results found"
    return PASS, f"{n} condition pairs verified on identical sentence ids"


@check("transfer gap is positive for most banks")
def check_direction(model):
    gaps = analysis.transfer_gap_table(model, n_boot=200)
    if gaps.empty or "gap_total" not in gaps:
        return SKIP, "no gap results yet"
    s = analysis.average_over_seeds(gaps)
    pos = (s.gap_total > 0).sum()
    assert pos >= 0.6 * len(s), f"only {pos}/{len(s)} banks have a positive gap"
    return PASS, (f"{pos}/{len(s)} banks positive; mean {s.gap_total.mean():+.4f} "
                  f"({s.gap_bank.mean()/s.gap_total.mean():.0%} from bank identity)")


@check("reproduces WCB pooled weighted F1 ~0.714")
def check_repro(model):
    runs = results_io.load_all(model)
    pooled = runs[runs.condition == "pooled"]
    if pooled.empty:
        return SKIP, "no pooled results yet"
    mean_f1 = pooled.groupby("seed").weighted_f1.mean().mean()
    if model != "roberta":
        return SKIP, f"{model} pooled weighted F1 = {mean_f1:.4f} (target applies to roberta)"
    assert abs(mean_f1 - 0.714) < 0.05, f"got {mean_f1:.4f}, expected ~0.714"
    return PASS, f"pooled weighted F1 = {mean_f1:.4f} vs published 0.714"


@check("Czech hold-out (26th bank, never trained on)")
def check_czech(model, checkpoint):
    """Informational, not a gate.

    The pooled weighted-F1 target of 0.714 is well specified -- same model size, same
    setup, the paper's Table 1 -- so that one is a real check. The Czech figure of 0.800
    comes from an appendix that does not state which model produced it, and the paper's
    best PLM is RoBERTa-LARGE (0.740 general vs 0.714 for base). Failing a base-sized
    model against a number that may describe a larger one would be a miscalibrated gate,
    so this reports the value and flags it rather than failing the suite.
    """
    if model != "roberta" or not checkpoint or not Path(checkpoint).exists():
        return SKIP, "needs a fine-tuned pooled checkpoint"

    from wcb.demo import RobertaBackend
    czech = data.load_czech()
    probs = RobertaBackend(checkpoint).predict_proba(czech[C.TEXT_COLUMN].tolist())
    preds = probs.argmax(1)
    m = E.compute_metrics(czech["stance_id"].values, preds)

    # Per-class detail, because the interesting failure mode is over-predicting the rare
    # `irrelevant` class: it is 3.4% of training data but only 1.6% of the Czech set, and
    # checkpoints here are selected on validation MACRO F1, which rewards that class.
    pred_mix = {C.ID2STANCE[i]: int((preds == i).sum()) for i in range(len(C.STANCE_LABELS))}
    true_mix = {C.ID2STANCE[i]: int((czech["stance_id"].values == i).sum())
                for i in range(len(C.STANCE_LABELS))}
    detail = (f"weighted F1 {m['weighted_f1']:.4f}, macro {m['macro_f1']:.4f} "
              f"(appendix reports 0.800, model size unstated); "
              f"predicted {pred_mix} vs true {true_mix}")

    return (PASS if m["weighted_f1"] >= 0.70 else WARN), detail


@check("report regenerates from results alone")
def check_report(model):
    import subprocess
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "scripts/make_report.py"),
                        "--model", model, "--n-boot", "500"],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, f"make_report failed: {r.stderr[-300:]}"
    figs = list((root / "figures").glob(f"{model}_fig*.png"))
    assert len(figs) >= 3, f"only {len(figs)} figures produced"
    return PASS, f"{len(figs)} figures regenerated"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="baseline")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--skip-report", action="store_true")
    args = ap.parse_args()

    setup_logging()
    log.info("verifying model=%s\n", args.model)

    frames = data.load_annotated(C.PRIMARY_SEED)

    check_integrity(frames)
    check_test_untouched(frames)
    check_leakage(frames)
    check_size_control(frames)
    check_fastmetrics()
    check_paired()
    check_same_items(args.model, C.PRIMARY_SEED)
    check_direction(args.model)
    check_repro(args.model)
    check_czech(args.model, args.checkpoint)
    if not args.skip_report:
        check_report(args.model)

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    log.info("\n%d passed, %d warned, %d skipped, %d FAILED",
             len(results) - n_fail - n_skip - n_warn, n_warn, n_skip, n_fail)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
