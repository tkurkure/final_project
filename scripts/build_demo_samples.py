#!/usr/bin/env python
"""Pick the per-bank sample sentences the demo's "Explore the 25 banks" tab shows.

    python scripts/build_demo_samples.py

For each bank, takes TEST sentences from the demo checkpoint's split (seed 5768) -- so the
demo model never trained on them -- and stores three readings of each: the human label,
the pooled model's prediction (trained WITH the bank) and the LOBO model's (trained
WITHOUT it). All three already sit in the saved per-fold results; nothing is re-run.

The sample is deliberately not random: it leads with sentences the pooled model gets
right and the LOBO model gets wrong, because those are what the gap looks like. The demo
says so on screen, and pairs every sample with the bank's full-test-set gap so the
picked examples can't stand in for the overall number.

Output: results/demo_bank_samples.json, small enough to ship with the demo on its own.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from wcb import config as C, data, results_io

SEED = C.PRIMARY_SEED     # the split the demo checkpoint (pooled_model) was trained on
PER_BANK = 6
GAP_EXAMPLES = 3          # at most this many "LOBO wrong, pooled right" sentences
MAX_WORDS = 45            # long sentences make an unreadable table


def main() -> None:
    test = data.load_annotated(SEED)["test"].set_index("sentence_id")
    summary = pd.read_csv(C.RESULTS_DIR / "roberta_gap_summary.csv", index_col=0)
    rng = np.random.default_rng(0)
    banks = []

    for bank in C.BANKS:
        pooled = results_io.load_result("roberta", SEED, "pooled", bank)
        lobo = results_io.load_result("roberta", SEED, "lobo", bank)
        assert pooled["sentence_id"] == lobo["sentence_id"], bank

        rows = pd.DataFrame({"sentence_id": pooled["sentence_id"], "human": pooled["y_true"],
                             "with_bank": pooled["y_pred"], "without_bank": lobo["y_pred"]})
        rows = rows.join(test[[C.TEXT_COLUMN, C.YEAR_COLUMN]], on="sentence_id")
        rows = rows[rows[C.TEXT_COLUMN].str.split().str.len() <= MAX_WORDS]

        gap = rows[(rows.with_bank == rows.human) & (rows.without_bank != rows.human)]
        rest = rows.drop(gap.index)
        take_gap = gap.sample(min(GAP_EXAMPLES, len(gap)), random_state=rng.integers(1 << 31))
        take_rest = rest.sample(min(PER_BANK - len(take_gap), len(rest)),
                                random_state=rng.integers(1 << 31))
        picked = pd.concat([take_gap, take_rest]).sort_values(C.YEAR_COLUMN)

        banks.append({
            "bank": bank,
            "name": C.BANK_DISPLAY_NAMES[bank],
            # Report numbers: pooled minus LOBO weighted F1, mean over 3 splits, in points.
            "drop_points": round(100 * summary.loc[bank, "gap_total"], 1),
            "with_bank_f1": round(100 * summary.loc[bank, "pooled_weighted_f1"], 1),
            "without_bank_f1": round(100 * summary.loc[bank, "lobo_weighted_f1"], 1),
            "sentences": [{
                "year": int(r[C.YEAR_COLUMN]) if pd.notna(r[C.YEAR_COLUMN]) else None,
                "text": r[C.TEXT_COLUMN],
                "human": C.ID2STANCE[r.human],
                "with_bank": C.ID2STANCE[r.with_bank],
                "without_bank": C.ID2STANCE[r.without_bank],
            } for _, r in picked.iterrows()],
        })

    banks.sort(key=lambda b: b["drop_points"], reverse=True)
    out = C.RESULTS_DIR / "demo_bank_samples.json"
    out.write_text(json.dumps({"seed": SEED, "banks": banks}, indent=1, ensure_ascii=False))

    print(f"wrote {out}")
    for b in banks:
        s = b["sentences"]
        n_gap = sum(x["with_bank"] == x["human"] != x["without_bank"] for x in s)
        print(f"  {b['name']:<40} drop {b['drop_points']:>5.1f}  "
              f"{len(s)} sentences, {n_gap} showing the gap")


if __name__ == "__main__":
    main()
