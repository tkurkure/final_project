"""Central configuration: dataset ids, bank slugs, label maps, paths, hyperparameters.

Every other module imports constants from here so that a single edit (e.g. switching
split seed) propagates consistently across baseline, training, analysis and demo.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Results and figures can be redirected with environment variables. This exists for
# ephemeral compute: on Colab the project directory is wiped when the session ends, so
# pointing WCB_RESULTS_DIR at mounted Google Drive is what makes the sweep genuinely
# resumable across disconnects rather than merely restartable.
RESULTS_DIR = Path(os.environ.get("WCB_RESULTS_DIR") or PROJECT_ROOT / "results")
FIGURES_DIR = Path(os.environ.get("WCB_FIGURES_DIR") or PROJECT_ROOT / "figures")

BASELINE_RESULTS = RESULTS_DIR / "baseline"
POOLED_RESULTS = RESULTS_DIR / "pooled"
LOBO_RESULTS = RESULTS_DIR / "lobo"
SIZE_CONTROL_RESULTS = RESULTS_DIR / "size_control"

# --------------------------------------------------------------------------------------
# HuggingFace dataset ids
# --------------------------------------------------------------------------------------
ANNOTATED_DATASET = "gtfintechlab/all_annotated_sentences_25000"
FULL_CORPUS_DATASET = "gtfintechlab/WCB_380k_sentences"
PER_BANK_DATASET_TEMPLATE = "gtfintechlab/{bank}"

# The three dataset configs are three *different 70/15/15 partitions* of the same 25,000
# sentences -- not three training seeds. Repeating across them is the correct way to put
# error bars on a per-bank F1 estimated from only 150 test items.
SPLIT_SEEDS = [5768, 78516, 944601]
PRIMARY_SEED = 5768

# --------------------------------------------------------------------------------------
# Banks
# --------------------------------------------------------------------------------------
# IMPORTANT: these are the values that actually appear in the `bank_name` column of the
# aggregated dataset, verified by inspection -- they are NOT the same as the per-bank HF
# dataset slugs in the WorldCentralBanks README (which uses `federal_reserve_system`,
# `european_central_bank`, `reserve_bank_of_india`, `central_bank_of_brazil`, ...).
# Always filter with these; use BANK_DISPLAY_NAMES for anything user-facing.
BANKS = [
    "bank_negara_malaysia",
    "bank_of_canada",
    "bank_of_colombia",
    "bank_of_england",
    "bank_of_israel",
    "bank_of_japan",
    "bank_of_korea",
    "bank_of_mexico",
    "bank_of_phillipines",
    "bank_of_thailand",
    "brazil_copom",
    "central_bank_of_chile",
    "central_bank_of_egypt",
    "central_bank_of_russia",
    "central_bank_republic_of_turkey",
    "central_bank_taiwan",
    "central_reserve_bank_peru",
    "ecb",
    "fomc",
    "monetary_authority_singapore",
    "national_bank_poland",
    "peoples_bank_of_china",
    "rbi",
    "reserve_bank_australia",
    "swiss_national_bank",
]
N_BANKS = len(BANKS)

BANK_DISPLAY_NAMES = {
    "bank_negara_malaysia": "Bank Negara Malaysia",
    "bank_of_canada": "Bank of Canada",
    "bank_of_colombia": "Bank of the Republic (Colombia)",
    "bank_of_england": "Bank of England",
    "bank_of_israel": "Bank of Israel",
    "bank_of_japan": "Bank of Japan",
    "bank_of_korea": "Bank of Korea",
    "bank_of_mexico": "Bank of Mexico",
    "bank_of_phillipines": "Central Bank of the Philippines",
    "bank_of_thailand": "Bank of Thailand",
    "brazil_copom": "Central Bank of Brazil (COPOM)",
    "central_bank_of_chile": "Central Bank of Chile",
    "central_bank_of_egypt": "Central Bank of Egypt",
    "central_bank_of_russia": "Central Bank of the Russian Federation",
    "central_bank_republic_of_turkey": "Central Bank of Turkey",
    "central_bank_taiwan": "Central Bank of China (Taiwan)",
    "central_reserve_bank_peru": "Central Reserve Bank of Peru",
    "ecb": "European Central Bank",
    "fomc": "Federal Reserve System (FOMC)",
    "monetary_authority_singapore": "Monetary Authority of Singapore",
    "national_bank_poland": "National Bank of Poland",
    "peoples_bank_of_china": "People's Bank of China",
    "rbi": "Reserve Bank of India",
    "reserve_bank_australia": "Reserve Bank of Australia",
    "swiss_national_bank": "Swiss National Bank",
}

# Tier-2 subset: the five banks named in the proposal as the diverse fallback set. Used
# for multi-seed LOBO runs when GPU budget does not allow 25 banks x 3 seeds.
FOCUS_BANKS = [
    "central_bank_republic_of_turkey",
    "peoples_bank_of_china",
    "brazil_copom",
    "ecb",
    "national_bank_poland",
]

# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------
# Canonical ordering. Index order is fixed so saved logits/confusion matrices from any
# run remain interpretable without needing the label map alongside them.
# NOTE: values are lowercase in the dataset, and time labels have NO hyphen
# ("not forward looking", not "Not forward-looking"). Verified by inspection.
STANCE_LABELS = ["hawkish", "dovish", "neutral", "irrelevant"]
TIME_LABELS = ["forward looking", "not forward looking"]
CERTAIN_LABELS = ["certain", "uncertain"]

STANCE2ID = {label: i for i, label in enumerate(STANCE_LABELS)}
ID2STANCE = {i: label for label, i in STANCE2ID.items()}

# Document-level stance score is (#hawkish - #dovish) / #non-irrelevant, so hawkish and
# dovish carry opposite sign and neutral contributes only to the denominator.
STANCE_SCORE_WEIGHTS = {"hawkish": 1.0, "dovish": -1.0, "neutral": 0.0, "irrelevant": None}

TEXT_COLUMN = "sentences"
STANCE_COLUMN = "stance_label"
TIME_COLUMN = "time_label"
CERTAIN_COLUMN = "certain_label"
BANK_COLUMN = "bank_name"
YEAR_COLUMN = "year"
INDEX_COLUMN = "__index_level_0__"

# --------------------------------------------------------------------------------------
# Model / training hyperparameters
# --------------------------------------------------------------------------------------
MODEL_NAME = "roberta-base"

# Grid from the WCB paper (Section 4). Searched ONCE on the pooled setup; the winner is
# then frozen across all 25 LOBO folds -- grid-searching per fold costs 25x for no gain.
HPARAM_GRID = {"batch_size": [32, 16], "learning_rate": [1e-5, 1e-6]}

# Frozen defaults. Gate 0 measured token lengths over all 25k sentences: p50=31, p95=65,
# p99=90, max=305. max_length=128 covers 99.78% of sentences, so truncation is a
# non-issue and the compute budget holds.
DEFAULT_HPARAMS = {
    "batch_size": 32,
    "learning_rate": 1e-5,
    "num_epochs": 4,
    "max_length": 128,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "seed": 42,
}

# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
N_BOOTSTRAP = 10_000
BOOTSTRAP_CI = 0.95
