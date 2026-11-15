#!/usr/bin/env python
"""Convert the demo checkpoint into the ONNX bundle the browser demo (web/) loads.

    pip install -e ".[web]"
    python scripts/export_web_model.py --checkpoint results/roberta/seed5768/pooled_model

Writes build/web_model/ in the layout transformers.js expects (config.json, tokenizer
files, onnx/model.onnx), ready to upload as a Hugging Face model repo.

The weights stay full precision (fp32, ~500 MB) so the browser gives the same labels as
the PyTorch model the report is based on. Compression was measured and rejected: on the
3,750 seed-5768 test sentences, int8 changed 290 labels (7.7%) and one bank's F1 by 5.6
points; fp16 changed 4. The script re-checks agreement on a slice of the test set.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from wcb import config as C, data

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json"]
CHECK_SENTENCES = 500


def export(checkpoint: Path, out: Path) -> Path:
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).eval()
    path = out / "onnx" / "model.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)

    enc = tok(["a sample sentence", "another"], padding=True, return_tensors="pt")
    torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), str(path),
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                                    "attention_mask": {0: "batch", 1: "seq"},
                                    "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)

    # Real label names, so the bundle is self-describing on the Hub.
    cfg = json.loads((checkpoint / "config.json").read_text())
    cfg["id2label"] = {str(i): s for i, s in C.ID2STANCE.items()}
    cfg["label2id"] = {s: i for i, s in C.ID2STANCE.items()}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for name in TOKENIZER_FILES:
        shutil.copy(checkpoint / name, out / name)

    check(model, tok, path)
    return path


def check(model, tok, path: Path) -> None:
    """Label agreement between the ONNX export and the PyTorch model on test sentences."""
    texts = data.load_annotated(C.PRIMARY_SEED)["test"][C.TEXT_COLUMN].tolist()[:CHECK_SENTENCES]
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    same = 0
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], padding=True, truncation=True,
                  max_length=C.DEFAULT_HPARAMS["max_length"], return_tensors="np")
        feeds = {k: enc[k].astype(np.int64) for k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            ref = model(**{k: torch.tensor(v) for k, v in feeds.items()}).logits.numpy()
        same += int((session.run(None, feeds)[0].argmax(1) == ref.argmax(1)).sum())
    print(f"ONNX matches PyTorch on {same}/{len(texts)} test sentences")
    assert same == len(texts), "ONNX export disagrees with PyTorch -- do not ship it"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=C.PROJECT_ROOT / "build" / "web_model")
    args = ap.parse_args()
    path = export(args.checkpoint, args.out)
    print(f"wrote {args.out} ({path.stat().st_size / 1e6:.0f} MB model)")


if __name__ == "__main__":
    main()
