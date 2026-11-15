"""Reading and writing run results.

One JSON per (model, seed, condition, bank), holding both the metric bundle and the
per-sentence predictions. Storing predictions is what lets error analysis and the paired
bootstrap run later without retraining anything -- and it means fine-tuned LOBO
checkpoints can be deleted the moment a fold finishes, instead of keeping ~12 GB of
weights around.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import config as C

# The only directory names that hold fold results. Globbing is restricted to these
# because a saved HuggingFace checkpoint lives alongside them (results/<model>/seed<N>/
# pooled_model/) and its own config.json would otherwise be read as a fold result.
CONDITIONS = ("pooled", "lobo", "size_control")


def result_path(model: str, seed: int, condition: str, bank: str) -> Path:
    """results/<model>/seed<seed>/<condition>/<bank>.json"""
    return C.RESULTS_DIR / model / f"seed{seed}" / condition / f"{bank}.json"


def save_result(model: str, seed: int, condition: str, bank: str, payload: Dict) -> Path:
    path = result_path(model, seed, condition, bank)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "seed": seed, "condition": condition, "bank": bank, **payload}
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_result(model: str, seed: int, condition: str, bank: str) -> Optional[Dict]:
    path = result_path(model, seed, condition, bank)
    return json.loads(path.read_text()) if path.exists() else None


def is_done(model: str, seed: int, condition: str, bank: str) -> bool:
    """Resumability check: a completed fold is simply one whose file already exists."""
    return result_path(model, seed, condition, bank).exists()


def load_all(model: str) -> pd.DataFrame:
    """Flatten every saved result for a model into one tidy frame of metrics."""
    rows: List[Dict] = []
    root = C.RESULTS_DIR / model
    paths = sorted(p for cond in CONDITIONS for p in root.glob(f"seed*/{cond}/*.json"))
    for path in paths:
        r = json.loads(path.read_text())
        if "model" not in r:          # not a fold result -- skip defensively
            continue
        m = r.get("metrics", {})
        rows.append({
            "model": r["model"], "seed": r["seed"], "condition": r["condition"],
            "bank": r["bank"], "n_train": r.get("n_train"),
            "n_test": m.get("n"),
            "weighted_f1": m.get("weighted_f1"), "macro_f1": m.get("macro_f1"),
            "accuracy": m.get("accuracy"),
            **{f"f1_{k}": v for k, v in (m.get("per_class_f1") or {}).items()},
        })
    return pd.DataFrame(rows)


def load_predictions(model: str, seed: int, condition: str, bank: str) -> pd.DataFrame:
    """Per-sentence predictions for one fold, as a frame aligned by sentence_id."""
    r = load_result(model, seed, condition, bank)
    if r is None:
        raise FileNotFoundError(result_path(model, seed, condition, bank))
    return pd.DataFrame({
        "sentence_id": r["sentence_id"],
        "y_true": r["y_true"],
        "y_pred": r["y_pred"],
    })
