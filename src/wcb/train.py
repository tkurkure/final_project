"""Fine-tuning RoBERTa for stance classification, one fold at a time.

Deliberately a plain PyTorch loop rather than HF `Trainer`. The training here is simple
(single-task classification, fixed hyperparameters), while `Trainer`'s argument names
have churned across transformers versions -- and this code has to run unchanged on a
laptop, on Colab and on Dartmouth JupyterHub, whatever each has installed.

Checkpoint selection uses validation MACRO F1, not weighted F1: `irrelevant` is only
~3.4% of the data, so weighted F1 barely moves when a model abandons that class
entirely, and we would silently select a model that never predicts it.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import config as C
from .splits import Fold

logger = logging.getLogger(__name__)


def get_device(prefer: Optional[str] = None) -> str:
    import torch
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    model_name: str = C.MODEL_NAME
    batch_size: int = C.DEFAULT_HPARAMS["batch_size"]
    learning_rate: float = C.DEFAULT_HPARAMS["learning_rate"]
    num_epochs: int = C.DEFAULT_HPARAMS["num_epochs"]
    max_length: int = C.DEFAULT_HPARAMS["max_length"]
    weight_decay: float = C.DEFAULT_HPARAMS["weight_decay"]
    warmup_ratio: float = C.DEFAULT_HPARAMS["warmup_ratio"]
    seed: int = C.DEFAULT_HPARAMS["seed"]
    eval_batch_size: int = 128
    device: Optional[str] = None

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


class _TextDataset:
    """Minimal dataset; tokenization happens per batch in the collator."""

    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        item = {"text": self.texts[i]}
        if self.labels is not None:
            item["label"] = self.labels[i]
        return item


def _make_collator(tokenizer, max_length: int):
    import torch

    def collate(batch):
        enc = tokenizer([b["text"] for b in batch], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        if "label" in batch[0]:
            enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return enc

    return collate


def _loader(dataset, tokenizer, cfg: TrainConfig, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=_make_collator(tokenizer, cfg.max_length))


@dataclass
class TrainedModel:
    """A fine-tuned model plus the tokenizer and metadata needed to use or save it."""
    model: object
    tokenizer: object
    cfg: TrainConfig
    device: str
    history: list = field(default_factory=list)
    best_val_macro_f1: float = float("nan")

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Class probabilities and argmax predictions for a frame of sentences."""
        import torch
        self.model.eval()
        ds = _TextDataset(df[C.TEXT_COLUMN])
        dl = _loader(ds, self.tokenizer, self.cfg, self.cfg.eval_batch_size, shuffle=False)

        probs = []
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())

        probs = np.concatenate(probs)
        return {"probs": probs, "preds": probs.argmax(axis=1)}

    def save(self, path) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def train_fold(fold: Fold, cfg: Optional[TrainConfig] = None) -> TrainedModel:
    """Fine-tune one fold, selecting the best epoch on validation macro F1."""
    import torch
    from torch.optim import AdamW
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    from .evaluate import macro_f1

    cfg = cfg or TrainConfig()
    device = cfg.device or get_device()
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=len(C.STANCE_LABELS)).to(device)

    train_dl = _loader(_TextDataset(fold.train[C.TEXT_COLUMN], fold.train["stance_id"]),
                       tokenizer, cfg, cfg.batch_size, shuffle=True)
    val_dl = _loader(_TextDataset(fold.val[C.TEXT_COLUMN], fold.val["stance_id"]),
                     tokenizer, cfg, cfg.eval_batch_size, shuffle=False)

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(d in n for d in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(d in n for d in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(params, lr=cfg.learning_rate)
    total_steps = len(train_dl) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps)

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_state, best_f1, history = None, -np.inf, []

    for epoch in range(cfg.num_epochs):
        model.train()
        t0, total_loss = time.time(), 0.0

        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = model(**batch).loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

        # --- validation -------------------------------------------------------------
        model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for batch in val_dl:
                labels = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                preds.append(model(**batch).logits.float().argmax(-1).cpu().numpy())
                golds.append(labels.numpy())

        val_f1 = macro_f1(np.concatenate(golds), np.concatenate(preds))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_dl),
                        "val_macro_f1": float(val_f1), "seconds": time.time() - t0})
        logger.info("    epoch %d/%d  loss=%.4f  val_macro_f1=%.4f  (%.0fs)",
                    epoch + 1, cfg.num_epochs, total_loss / len(train_dl), val_f1,
                    time.time() - t0)

        if val_f1 > best_f1:
            best_f1 = val_f1
            # Kept in memory rather than written to disk: with 25 LOBO folds, round-tripping
            # ~500 MB of weights per epoch would dominate the runtime.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainedModel(model=model, tokenizer=tokenizer, cfg=cfg, device=device,
                        history=history, best_val_macro_f1=float(best_f1))
