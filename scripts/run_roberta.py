#!/usr/bin/env python
"""Fine-tune RoBERTa for one or more conditions. This is the GPU workhorse.

    # 0. before committing to the sweep, time a single fold on this machine
    python scripts/run_roberta.py --condition lobo --banks ecb --time-only

    # 1. choose hyperparameters ONCE on the pooled fold
    python scripts/run_roberta.py --hparam-search

    # 2. the Tier-1 sweep: pooled + size control + all 25 LOBO folds
    python scripts/run_roberta.py --condition pooled size_control lobo

    # 3. Tier 2, if a second session is available
    python scripts/run_roberta.py --condition lobo --banks focus --seeds 78516 944601

    # 4. verify against the two published checkpoints
    python scripts/verify.py --model roberta \
        --checkpoint results/roberta/seed5768/pooled_model

Resumable by design: a fold whose results file already exists is skipped, so an 8-hour
JupyterHub session that times out mid-sweep costs one fold, not the run. Checkpoints are
discarded once a fold's per-sentence predictions are saved -- those predictions are all
the downstream analysis needs, and keeping 25 fine-tuned models would mean ~12 GB.

This single script replaces the separately-planned run_pooled / run_lobo /
run_size_control, which would have been near-identical copies.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wcb import config as C, data, evaluate as E, results_io, splits
from wcb.logging_utils import setup_logging
from wcb.train import TrainConfig, get_device, train_fold

MODEL = "roberta"
log = logging.getLogger("run_roberta")

HPARAM_FILE = C.RESULTS_DIR / MODEL / "selected_hparams.json"


def load_selected_hparams() -> dict:
    """Hyperparameters chosen once on the pooled fold and frozen for every other fold.

    Searching per fold would cost 25x and give each LOBO model a different
    model-selection budget, making the folds incomparable.
    """
    if HPARAM_FILE.exists():
        chosen = json.loads(HPARAM_FILE.read_text())["best"]
        log.info("using frozen hyperparameters from %s: %s", HPARAM_FILE.name, chosen)
        return chosen
    log.warning("no %s found -- falling back to config defaults", HPARAM_FILE.name)
    return {k: C.DEFAULT_HPARAMS[k] for k in ("batch_size", "learning_rate")}


def score_banks(trained, frames, banks, seed, condition, extra=None) -> None:
    """Score a trained model on each bank's canonical evaluation set and save results."""
    for bank in banks:
        eval_df = splits.evaluation_set(frames, bank)
        out = trained.predict(eval_df)
        payload = {
            "sentence_id": eval_df["sentence_id"].tolist(),
            "y_true": eval_df["stance_id"].tolist(),
            "y_pred": out["preds"].tolist(),
            "probs": out["probs"].tolist(),
            "metrics": E.compute_metrics(eval_df["stance_id"].values, out["preds"]),
            "hparams": trained.cfg.to_dict(),
            "val_macro_f1": trained.best_val_macro_f1,
            "history": trained.history,
            **(extra or {}),
        }
        results_io.save_result(MODEL, seed, condition, bank, payload)


def free(trained) -> None:
    """Release the fine-tuned weights before the next fold starts."""
    import gc
    import torch
    del trained
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_hparam_search(seeds, device) -> None:
    """Grid from the WCB paper, evaluated on the pooled fold's validation split."""
    frames = data.load_annotated(C.PRIMARY_SEED)
    fold = splits.pooled_fold(frames)
    results = []

    for bs in C.HPARAM_GRID["batch_size"]:
        for lr in C.HPARAM_GRID["learning_rate"]:
            log.info("grid: batch_size=%d lr=%g", bs, lr)
            cfg = TrainConfig(batch_size=bs, learning_rate=lr, device=device)
            t0 = time.time()
            trained = train_fold(fold, cfg)
            results.append({"batch_size": bs, "learning_rate": lr,
                            "val_macro_f1": trained.best_val_macro_f1,
                            "minutes": (time.time() - t0) / 60})
            log.info("  -> val_macro_f1=%.4f (%.1f min)",
                     trained.best_val_macro_f1, (time.time() - t0) / 60)
            free(trained)

    best = max(results, key=lambda r: r["val_macro_f1"])
    HPARAM_FILE.parent.mkdir(parents=True, exist_ok=True)
    HPARAM_FILE.write_text(json.dumps(
        {"best": {"batch_size": best["batch_size"], "learning_rate": best["learning_rate"]},
         "all": results}, indent=2))
    log.info("\nselected batch_size=%d lr=%g (val_macro_f1=%.4f) -> %s",
             best["batch_size"], best["learning_rate"], best["val_macro_f1"], HPARAM_FILE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", nargs="+", default=["pooled", "size_control", "lobo"],
                    choices=["pooled", "size_control", "lobo"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[C.PRIMARY_SEED])
    ap.add_argument("--banks", nargs="+", default=None,
                    help="bank names, or 'focus' for the 5-bank subset")
    ap.add_argument("--epochs", type=int, default=C.DEFAULT_HPARAMS["num_epochs"])
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--hparam-search", action="store_true")
    ap.add_argument("--time-only", action="store_true",
                    help="train the first requested fold, report timing, save nothing")
    ap.add_argument("--limit-train", type=int, default=None,
                    help="subsample training rows (smoke tests only)")
    ap.add_argument("--no-save-pooled", action="store_true",
                    help="skip saving the pooled checkpoint (it is kept by default)")
    args = ap.parse_args()

    setup_logging()
    device = args.device or get_device()
    log.info("device: %s", device)

    if args.hparam_search:
        run_hparam_search(args.seeds, device)
        return

    banks = C.BANKS if args.banks is None else (
        C.FOCUS_BANKS if args.banks == ["focus"] else args.banks)
    for b in banks:
        assert b in C.BANKS, f"unknown bank {b!r}"

    hp = load_selected_hparams()
    base_cfg = dict(num_epochs=args.epochs, device=device,
                    batch_size=hp["batch_size"], learning_rate=hp["learning_rate"])

    sweep_start = time.time()
    n_trained = 0

    for seed in args.seeds:
        log.info("\n=== seed %d ===", seed)
        frames = data.load_annotated(seed)

        for condition in args.condition:
            # pooled and size_control are ONE model each, scored on every bank;
            # lobo is one model per bank.
            fold_banks = [None] if condition in ("pooled", "size_control") else banks

            for held_out in fold_banks:
                targets = banks if held_out is None else [held_out]
                if not args.force and all(
                        results_io.is_done(MODEL, seed, condition, b) for b in targets):
                    log.info("%s%s: already done, skipping", condition,
                             f" ({held_out})" if held_out else "")
                    continue

                if condition == "pooled":
                    fold = splits.pooled_fold(frames)
                elif condition == "size_control":
                    fold = splits.size_control_fold(frames)
                else:
                    fold = splits.lobo_fold(frames, held_out)

                if args.limit_train:
                    fold.train = fold.train.head(args.limit_train)

                log.info("\n-- %s  train=%d  val=%d  held_out=%s",
                         fold.name, len(fold.train), len(fold.val), fold.held_out_bank)
                t0 = time.time()
                trained = train_fold(fold, TrainConfig(**base_cfg))
                minutes = (time.time() - t0) / 60
                log.info("   trained in %.1f min (val_macro_f1=%.4f)",
                         minutes, trained.best_val_macro_f1)
                n_trained += 1

                if args.time_only:
                    log.info("\n--time-only: one fold took %.1f min. "
                             "25 LOBO folds ~= %.1f h; 8-hour session fits %d folds.",
                             minutes, minutes * 25 / 60, int(8 * 60 / minutes))
                    return

                score_banks(trained, frames, targets, seed, condition,
                            extra={"n_train": len(fold.train), "minutes": minutes})

                # The 25 LOBO checkpoints are discarded -- their saved per-sentence
                # predictions are all the analysis needs, and keeping them would mean
                # ~12 GB. The POOLED checkpoint is kept: it is what the demo serves and
                # what the Czech hold-out check scores.
                if condition == "pooled" and not args.no_save_pooled:
                    dest = C.RESULTS_DIR / MODEL / f"seed{seed}" / "pooled_model"
                    dest.mkdir(parents=True, exist_ok=True)
                    trained.save(dest)
                    log.info("   saved pooled checkpoint -> %s", dest)

                free(trained)

    log.info("\ndone: %d fold(s) trained in %.1f min -> %s",
             n_trained, (time.time() - sweep_start) / 60, C.RESULTS_DIR / MODEL)


if __name__ == "__main__":
    main()
