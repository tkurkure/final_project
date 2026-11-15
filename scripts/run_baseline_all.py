#!/usr/bin/env python
"""Run the TF-IDF baseline across every condition, bank and split seed.

    python scripts/run_baseline_all.py                 # all 3 seeds, all 25 banks
    python scripts/run_baseline_all.py --seeds 5768    # one seed
    python scripts/run_baseline_all.py --force         # ignore completed folds

CPU-only and resumable: a fold whose results file already exists is skipped, so an
interrupted sweep continues where it stopped.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wcb import baseline, config as C, data, evaluate as E, results_io, splits
from wcb.logging_utils import setup_logging

MODEL = "baseline"
log = logging.getLogger("run_baseline")


def score_and_save(fold, frames, banks, seed, force: bool, C_reg: float) -> int:
    """Fit `fold` once, then score it on each requested bank's evaluation set."""
    todo = [b for b in banks if force or not results_io.is_done(MODEL, seed, fold.condition, b)]
    if not todo:
        log.info("  %-14s all %d banks already done", fold.condition, len(banks))
        return 0

    t0 = time.time()
    for bank in todo:
        eval_df = splits.evaluation_set(frames, bank)
        out = baseline.run_fold(fold, eval_df, C_reg=C_reg)
        out["metrics"] = E.compute_metrics(out["y_true"], out["y_pred"])
        out["C_reg"] = C_reg
        results_io.save_result(MODEL, seed, fold.condition, bank, out)
    log.info("  %-14s scored %d bank(s) in %.1fs", fold.condition, len(todo), time.time() - t0)
    return len(todo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SPLIT_SEEDS)
    ap.add_argument("--banks", type=str, nargs="+", default=None,
                    help="bank names, or 'focus' for the 5-bank subset")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--C", dest="C_reg", type=float, default=None,
                    help="skip tuning and use this regularization strength")
    args = ap.parse_args()

    setup_logging()

    banks = C.BANKS if args.banks is None else (
        C.FOCUS_BANKS if args.banks == ["focus"] else args.banks)
    for b in banks:
        assert b in C.BANKS, f"unknown bank {b!r}"

    # Tune the regularization strength ONCE on the primary seed's pooled fold, then freeze
    # it everywhere. Tuning per fold would give each LOBO model a different model-selection
    # budget and make the 25 folds incomparable.
    C_reg = args.C_reg
    if C_reg is None:
        log.info("tuning C on pooled fold of seed %d ...", C.PRIMARY_SEED)
        frames = data.load_annotated(C.PRIMARY_SEED)
        C_reg = baseline.tune_C(splits.pooled_fold(frames))
        log.info("selected C=%g (frozen for all folds)\n", C_reg)

    total = 0
    for seed in args.seeds:
        log.info("=== seed %d ===", seed)
        frames = data.load_annotated(seed)

        total += score_and_save(splits.pooled_fold(frames), frames, banks, seed, args.force, C_reg)
        total += score_and_save(splits.size_control_fold(frames), frames, banks, seed, args.force, C_reg)

        # Each LOBO fold is a separate model, scored only on its own held-out bank.
        t0, n = time.time(), 0
        for bank in banks:
            if not args.force and results_io.is_done(MODEL, seed, "lobo", bank):
                continue
            fold = splits.lobo_fold(frames, bank)
            eval_df = splits.evaluation_set(frames, bank)
            out = baseline.run_fold(fold, eval_df, C_reg=C_reg)
            out["metrics"] = E.compute_metrics(out["y_true"], out["y_pred"])
            out["C_reg"] = C_reg
            results_io.save_result(MODEL, seed, "lobo", bank, out)
            n += 1
        log.info("  %-14s trained %d fold(s) in %.1fs", "lobo", n, time.time() - t0)
        total += n

    log.info("\ndone: %d fold-evaluations written under %s", total, C.RESULTS_DIR / MODEL)


if __name__ == "__main__":
    main()
