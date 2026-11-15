"""Hugging Face Spaces entry point for the demo (see scripts/deploy_space.sh).

The Space mirrors the repo layout -- src/wcb/ and results/demo_bank_samples.json -- plus
the pooled RoBERTa checkpoint in model/, so wcb.demo runs unchanged.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import nltk

from wcb import demo
from wcb.logging_utils import setup_logging

# Calibrated locally on the seed-5768 validation split for this exact checkpoint
# (results/demo_threshold_cache.json). Recomputing it on the Space's CPU would add
# several minutes to every cold start and download the dataset.
THRESHOLD = 0.5414676666259766

setup_logging()
nltk.download("punkt_tab", quiet=True)   # better sentence splitting than the regex fallback
backend = demo.RobertaBackend(str(ROOT / "model"), device="cpu")
demo.build_app(backend, THRESHOLD).launch()
