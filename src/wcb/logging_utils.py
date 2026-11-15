"""Shared logging setup so run output stays readable.

The HF stack logs every HTTP request at INFO under the vendored logger name `httpx2`,
which otherwise buries the lines we actually care about.
"""
from __future__ import annotations

import logging

NOISY_LOGGERS = (
    "httpx", "httpx2", "httpcore", "urllib3", "filelock",
    "fsspec", "huggingface_hub", "datasets",
)


def setup_logging(level: int = logging.INFO, fmt: str = "%(message)s") -> None:
    logging.basicConfig(level=level, format=fmt)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
