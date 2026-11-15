"""Gradio demo with two tabs.

  Explore the 25 banks   real test sentences from each bank, read three ways: the human
                         label, the model trained WITH the bank, and the model that never
                         saw it. Precomputed by scripts/build_demo_samples.py.
  Try your own statement paste or upload (.txt / .pdf) a statement and get a
                         sentence-by-sentence stance read from the live model.

Run with:
    python -m wcb.demo                      # TF-IDF backend, trains in ~10s
    python -m wcb.demo --backend roberta --checkpoint results/roberta/pooled_model

Design commitments, both of which come straight from the project's stated risk -- that
someone trusts a confident-looking score for a bank the model reads badly:

  * The score is never shown alone. Every document score is displayed next to the
    sentences it was computed from, so a reader can always check the arithmetic.
  * A bank outside the 25 training banks triggers a prominent warning, because that is
    exactly the leave-one-bank-out condition this project measures, and we know what it
    costs on average -- and that it costs far more for some banks.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config as C
from .logging_utils import setup_logging

logger = logging.getLogger(__name__)

OTHER_BANK = "— other / not in the dataset —"

# Diverging pair for the two opposed policy stances, plus recessive neutrals. Text and
# rule colors in the result panel use Gradio's theme variables instead, so the panel
# stays legible when the browser is in dark mode.
STANCE_COLORS = {
    "hawkish": "#e34948",
    "dovish": "#2a78d6",
    "neutral": "#6b6a66",
    "irrelevant": "#a8a7a2",
}

# One plain-English line per label, shown in the legend so no reader needs the jargon.
STANCE_MEANINGS = {
    "hawkish": "leans toward higher interest rates (fighting inflation)",
    "dovish": "leans toward lower interest rates (supporting growth and jobs)",
    "neutral": "about policy or the economy, but doesn't lean either way",
    "irrelevant": "not about monetary policy (e.g. scheduling)",
}

# Fraction of the least-confident validation sentences to flag. Calibrated against the
# validation split rather than guessed, and the measured accuracy either side of the
# threshold is logged so the number means something.
FLAG_FRACTION = 0.15

# Scoring the full validation split takes ~2 minutes for RoBERTa, and the answer only
# changes when the checkpoint does, so it is cached per checkpoint file.
THRESHOLD_CACHE = C.RESULTS_DIR / "demo_threshold_cache.json"

SAMPLES_PATH = C.RESULTS_DIR / "demo_bank_samples.json"

# A pasted annual report can run to thousands of sentences; past this the reading gets
# slow on CPU and the page unreadable, so only the opening is analyzed (and the page says so).
MAX_SENTENCES = 150

# Why a bank is hard, for the banks where the error analysis found a clear reason.
BANK_NOTES = {
    "bank_negara_malaysia": (
        "Why it's hard: Malaysia's annotators usually labeled upbeat growth sentences "
        "(\"growth is expected to remain strong\") as <b>dovish</b>. Many other banks "
        "label similar sentences hawkish or neutral, so a model that never saw Malaysia "
        "tends to disagree with its labels."),
    "monetary_authority_singapore": (
        "Why it's hard: Singapore sets policy through its exchange rate (the S$NEER "
        "policy band) instead of an interest rate. Letting the currency rise is a "
        "<b>tightening</b>, but that vocabulary barely appears at any other bank, so a model "
        "that never saw Singapore misreads it. It also labels growth news mostly neutral."),
}


# ---------------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------------
_ABBREV = r"(?<!\bU\.S)(?<!\be\.g)(?<!\bi\.e)(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<!\bNo)(?<!\bvs)"
_FALLBACK_SPLIT = re.compile(rf"{_ABBREV}(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text: str) -> List[str]:
    """Split a pasted document into sentences.

    Prefers NLTK's punkt, which handles the decimals ("2.5 percent") and abbreviations
    ("the U.S. economy") that are everywhere in this domain; falls back to a guarded
    regex when the punkt data is not downloadable (e.g. an offline cluster node).
    """
    text = text.strip()
    if not text:
        return []
    try:
        from nltk.tokenize import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        logger.warning("punkt unavailable -- using the regex sentence splitter")
        return [s.strip() for s in _FALLBACK_SPLIT.split(text) if s.strip()]


# ---------------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------------
@dataclass
class Prediction:
    sentence: str
    label: str
    confidence: float
    low_confidence: bool


class Backend:
    """Anything that can turn sentences into stance probabilities.

    `unseen_drop` and `worst_drop` are this model's measured leave-one-bank-out cost
    (pooled minus LOBO weighted F1, in points, averaged over three splits): the mean
    over 25 banks, and the hardest bank. The unseen-bank warning quotes them.
    """
    name = "base"
    display_name = "base"
    unseen_drop = 0
    worst_drop = 0

    def predict_proba(self, sentences: List[str]) -> np.ndarray:
        raise NotImplementedError


class BaselineBackend(Backend):
    """TF-IDF + logistic regression, fit on the pooled training split at startup.

    Fast enough (~10s) that the demo is runnable before any GPU work exists, and it is
    the documented fallback if GPU access falls through.
    """
    name = "tfidf"
    display_name = "simple word-count model (TF-IDF)"
    unseen_drop = 5     # mean gap_total 0.053
    worst_drop = 30     # People's Bank of China, 0.295

    def __init__(self, frames):
        from .baseline import build_pipeline
        from .splits import pooled_fold

        fold = pooled_fold(frames)
        logger.info("fitting TF-IDF baseline on %d sentences ...", len(fold.train))
        self.pipe = build_pipeline()
        self.pipe.fit(fold.train[C.TEXT_COLUMN], fold.train["stance_id"])
        self.classes = self.pipe.named_steps["clf"].classes_

    def predict_proba(self, sentences: List[str]) -> np.ndarray:
        raw = self.pipe.predict_proba(sentences)
        full = np.zeros((len(sentences), len(C.STANCE_LABELS)))
        for col, cls in enumerate(self.classes):
            full[:, int(cls)] = raw[:, col]
        return full


class RobertaBackend(Backend):
    """A fine-tuned RoBERTa checkpoint saved by scripts/run_roberta.py."""
    name = "roberta"
    display_name = "fine-tuned RoBERTa language model"
    unseen_drop = 3     # mean gap_total 0.027
    worst_drop = 15     # Bank Negara Malaysia, 0.155

    def __init__(self, checkpoint: str, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .train import get_device

        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.model.to(self.device).eval()
        self.torch = torch

    def predict_proba(self, sentences: List[str]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(sentences), 32):
                enc = self.tokenizer(sentences[i:i + 32], padding=True, truncation=True,
                                     max_length=C.DEFAULT_HPARAMS["max_length"],
                                     return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.concatenate(out)


# ---------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------
def calibrate_threshold(backend: Backend, frames, fraction: float = FLAG_FRACTION) -> float:
    """Pick the confidence cutoff from the validation split, and verify it separates.

    Flags the least-confident `fraction` of validation sentences, then logs accuracy
    above and below the cutoff. If those two numbers are close, the flag is decoration
    and the log says so.
    """
    val = frames["val"]
    probs = backend.predict_proba(val[C.TEXT_COLUMN].tolist())
    conf = probs.max(axis=1)
    threshold = float(np.quantile(conf, fraction))

    correct = probs.argmax(axis=1) == val["stance_id"].values
    below, above = conf < threshold, conf >= threshold
    acc_below = correct[below].mean() if below.any() else float("nan")
    acc_above = correct[above].mean() if above.any() else float("nan")
    logger.info("confidence threshold %.3f (flags %.0f%% of val): "
                "accuracy %.1f%% below vs %.1f%% above",
                threshold, 100 * fraction, 100 * acc_below, 100 * acc_above)
    return threshold


def cached_threshold(backend: Backend, load_frames, checkpoint: str, seed: int) -> float:
    """calibrate_threshold, remembered per (checkpoint, split seed).

    Keyed on the weights file's size and mtime too, so retraining into the same
    directory invalidates the entry instead of silently reusing a stale cutoff.
    `load_frames` is only called on a cache miss, so a cached launch never downloads
    the dataset at all.
    """
    weights = next(Path(checkpoint).glob("*.safetensors"), None) or Path(checkpoint)
    stat = weights.stat()
    key = f"{Path(checkpoint).resolve()}|seed={seed}|{stat.st_size}|{stat.st_mtime_ns}"

    try:
        cache = json.loads(THRESHOLD_CACHE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    if key in cache:
        logger.info("using cached confidence threshold %.3f", cache[key])
        return cache[key]

    logger.info("first launch with this model: scoring the validation set to set the "
                "low-confidence cutoff (~2 min, cached afterwards) ...")
    threshold = calibrate_threshold(backend, load_frames())
    cache[key] = threshold
    THRESHOLD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_CACHE.write_text(json.dumps(cache, indent=2))
    return threshold


def document_score(preds: List[Prediction]) -> Tuple[float, int, int, int]:
    """(hawkish - dovish) / (sentences excluding irrelevant), in [-1, +1]."""
    hawk = sum(p.label == "hawkish" for p in preds)
    dove = sum(p.label == "dovish" for p in preds)
    relevant = sum(p.label != "irrelevant" for p in preds)
    score = (hawk - dove) / relevant if relevant else 0.0
    return score, hawk, dove, relevant


def analyze(text: str, backend: Backend, threshold: float) -> List[Prediction]:
    sentences = split_sentences(text)[:MAX_SENTENCES]
    if not sentences:
        return []
    probs = backend.predict_proba(sentences)
    out = []
    for sent, row in zip(sentences, probs):
        i = int(row.argmax())
        conf = float(row[i])
        out.append(Prediction(sent, C.ID2STANCE[i], conf, conf < threshold))
    return out


# ---------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------
def _score_bar(score: float, color: str) -> str:
    """A -1..+1 track with a marker, so the number has a visible place on the scale."""
    pos = (score + 1) / 2 * 100
    return (
        "<div style='max-width:420px;margin:6px 0 4px 0'>"
        "<div style='position:relative;height:8px;border-radius:4px;"
        f"background:linear-gradient(90deg,{STANCE_COLORS['dovish']}33,#e4e3df,"
        f"{STANCE_COLORS['hawkish']}33)'>"
        "<div style='position:absolute;left:50%;top:-3px;width:1px;height:14px;"
        "background:#8a8a85'></div>"
        f"<div style='position:absolute;left:calc({pos:.1f}% - 7px);top:-3px;width:14px;"
        f"height:14px;border-radius:50%;background:{color};border:2px solid #fff;"
        "box-shadow:0 0 0 1px #8a8a85'></div></div>"
        "<div style='display:flex;justify-content:space-between;font-size:0.78em;"
        "color:var(--body-text-color-subdued,#52514e);margin-top:4px'>"
        "<span>&minus;1 fully dovish</span><span>0 balanced</span>"
        "<span>+1 fully hawkish</span></div></div>")


def render(preds: List[Prediction], bank: str, backend: Backend, n_total: int = 0) -> str:
    if not preds:
        return ("<p style='color:var(--body-text-color-subdued,#52514e)'>Paste or upload a "
                "statement, then click Analyze.</p>")

    score, hawk, dove, relevant = document_score(preds)
    unseen = bank == OTHER_BANK

    n_aside = len(preds) - relevant
    aside = ("No sentences were" if n_aside == 0
             else "1 sentence was" if n_aside == 1 else f"{n_aside} sentences were")
    direction = ("leans hawkish" if score > 0.05
                 else "leans dovish" if score < -0.05 else "balanced")
    color = (STANCE_COLORS["hawkish"] if score > 0.05
             else STANCE_COLORS["dovish"] if score < -0.05 else STANCE_COLORS["neutral"])

    parts = [
        "<div style=\"font-family:ui-sans-serif,system-ui,sans-serif;color:var(--body-text-color,#0b0b0b)\">"
    ]

    if unseen:
        parts.append(
            "<div style=\"border-left:4px solid #eb6834;background:rgba(235,104,52,0.14);"
            "padding:10px 14px;margin-bottom:14px;border-radius:4px\">"
            "<strong>Heads up: the model never saw statements from this bank.</strong><br>"
            "<span style='color:var(--body-text-color-subdued,#52514e);font-size:0.92em'>"
            f"In our tests, this model scored about {backend.unseen_drop} points lower "
            "(on a 0&ndash;100 accuracy scale) on banks it wasn't trained on, and up to "
            f"{backend.worst_drop} points lower for banks with unusual wording. "
            "Treat this reading as a rough guide.</span></div>")
    elif bank:
        parts.append(
            "<div style='color:var(--body-text-color-subdued,#52514e);font-size:0.88em;margin-bottom:10px'>"
            f"The model was trained on statements from {html.escape(bank)}, so this is "
            "the case it handles best.</div>")

    if n_total > len(preds):
        parts.append(
            "<div style='color:var(--body-text-color-subdued,#52514e);font-size:0.88em;"
            f"margin-bottom:10px'>This text has {n_total} sentences; only the first "
            f"{len(preds)} were analyzed. Paste a shorter excerpt to read a later part.</div>")

    parts.append(
        f"<div style=\"display:flex;align-items:baseline;gap:14px\">"
        f"<span style='font-size:2.1em;font-weight:600;color:{color}'>{score:+.2f}</span>"
        f"<span style='color:var(--body-text-color-subdued,#52514e)'>overall stance &middot; "
        f"<strong>{direction}</strong></span></div>")
    parts.append(_score_bar(score, color))
    parts.append(
        f"<div style='color:var(--body-text-color-subdued,#52514e);font-size:0.9em;margin:8px 0 14px 0'>"
        f"How it's computed: ({hawk} hawkish &minus; {dove} dovish) &divide; {relevant} "
        f"sentences about policy = {score:+.2f}. "
        f"{aside} set aside as not about policy. "
        f"Model: {backend.display_name}.</div>")

    # Legend -- identity is never carried by color alone; each sentence is also
    # labeled in text below, and each label is defined in plain words here.
    parts.append("<div style='margin-bottom:12px;font-size:0.85em;line-height:1.7'>")
    for label, col in STANCE_COLORS.items():
        parts.append(
            f"<div><span style='display:inline-block;width:10px;height:10px;"
            f"background:{col};border-radius:2px;margin-right:6px'></span>"
            f"<strong>{label}</strong> <span style='color:var(--body-text-color-subdued,#52514e)'>&mdash; "
            f"{STANCE_MEANINGS[label]}</span></div>")
    parts.append("</div>")

    parts.append("<div style='border-top:1px solid var(--border-color-primary,#e4e3df);padding-top:10px'>")
    for i, p in enumerate(preds, 1):
        col = STANCE_COLORS[p.label]
        flag = ("<span style='color:#eb6834;font-weight:600' "
                "title='The model is less sure about this sentence than about 85% of "
                "the sentences it was tested on'>"
                " &#9888; unsure &mdash; worth a second look</span>"
                if p.low_confidence else "")
        parts.append(
            f"<div style='margin:0 0 9px 0;padding-left:10px;border-left:3px solid {col}'>"
            f"<div style='font-size:0.78em;color:var(--body-text-color-subdued,#52514e);margin-bottom:2px'>"
            f"{i}. <strong style='color:{col}'>{p.label}</strong> "
            f"&middot; model is {p.confidence:.0%} sure{flag}</div>"
            f"<div>{html.escape(p.sentence)}</div></div>")
    parts.append("</div></div>")
    return "".join(parts)


# ---------------------------------------------------------------------------------
# "Explore the 25 banks" tab
# ---------------------------------------------------------------------------------
MUTED = "var(--body-text-color-subdued,#52514e)"
TEXT = "var(--body-text-color,#0b0b0b)"
RULE = "var(--border-color-primary,#e4e3df)"


def load_bank_samples() -> List[dict]:
    """The precomputed per-bank samples, hardest bank first; [] if not built yet."""
    try:
        return json.loads(SAMPLES_PATH.read_text())["banks"]
    except FileNotFoundError:
        logger.warning("%s missing -- run scripts/build_demo_samples.py; "
                       "the Explore tab will be empty", SAMPLES_PATH)
        return []


def bank_choice_label(entry: dict) -> str:
    drop = entry["drop_points"]
    return (f"{entry['name']}  ·  {drop:.1f}-point drop" if drop >= 0.5
            else f"{entry['name']}  ·  no real drop")


def _pill(label: str) -> str:
    return (f"<span style='display:inline-block;padding:1px 8px;border-radius:10px;"
            f"background:{STANCE_COLORS[label]};color:#fff;font-size:0.85em;"
            f"white-space:nowrap'>{label}</span>")


def _model_cell(pred: str, human: str) -> str:
    """The model's label, with a right/wrong mark in words as well as symbols."""
    ok = pred == human
    mark = ("<span style='color:#1baf7a;font-weight:600' title='matches the human label'>"
            "&#10003;</span>" if ok else
            "<span style='color:#e34948;font-weight:600' title='disagrees with the human "
            "label'>&#10007;</span>")
    return f"{_pill(pred)} {mark}"


def _score_row(label: str, value: float, color: str) -> str:
    return (
        "<div style='display:flex;align-items:center;gap:10px;margin:4px 0'>"
        f"<span style='width:190px;font-size:0.9em;color:{MUTED}'>{label}</span>"
        f"<div style='flex:1;max-width:320px;height:12px;background:{RULE};border-radius:6px'>"
        f"<div style='width:{value:.0f}%;height:12px;background:{color};border-radius:6px'>"
        f"</div></div><span style='font-weight:600;width:40px'>{value:.0f}</span></div>")


def render_bank(entry: Optional[dict]) -> str:
    if not entry:
        return (f"<p style='color:{MUTED}'>No sample sentences found. Run "
                "<code>scripts/build_demo_samples.py</code> first.</p>")

    drop = entry["drop_points"]
    headline = (f"Leaving {html.escape(entry['name'])} out of training costs the model "
                f"<b>{drop:.1f} points</b>." if drop >= 0.5 else
                f"Leaving {html.escape(entry['name'])} out of training makes "
                "<b>no real difference</b>: the model reads it just as well without it.")
    note = BANK_NOTES.get(entry["bank"], "")

    parts = [f"<div style='font-family:ui-sans-serif,system-ui,sans-serif;color:{TEXT}'>",
             f"<div style='font-size:1.15em;margin-bottom:8px'>{headline}</div>",
             _score_row("Model trained with this bank", entry["with_bank_f1"], "#2a78d6"),
             _score_row("Model that never saw it", entry["without_bank_f1"], "#eb6834"),
             f"<div style='font-size:0.8em;color:{MUTED};margin:4px 0 12px 0'>Accuracy "
             "score (weighted F1, out of 100) on all of this bank's test sentences, "
             "averaged over three test splits.</div>"]
    if note:
        parts.append(f"<div style='border-left:4px solid #eb6834;background:rgba(235,104,52,"
                     f"0.14);padding:8px 12px;border-radius:4px;margin-bottom:12px'>{note}</div>")

    head = ("<th style='text-align:left;padding:6px 8px;border-bottom:1px solid "
            f"{RULE};font-weight:600'>")
    cell = f"<td style='padding:8px;border-bottom:1px solid {RULE};vertical-align:top'>"
    parts.append("<div style='overflow-x:auto'><table style='width:100%;border-collapse:"
                 "collapse;font-size:0.92em'><tr>"
                 f"{head}Year</th>{head}Sentence</th>{head}Human label</th>"
                 f"{head}Trained with bank</th>{head}Never saw bank</th></tr>")
    for s in entry["sentences"]:
        parts.append(
            f"<tr>{cell}{s['year'] or ''}</td>{cell}{html.escape(s['text'])}</td>"
            f"{cell}{_pill(s['human'])}</td>"
            f"{cell}{_model_cell(s['with_bank'], s['human'])}</td>"
            f"{cell}{_model_cell(s['without_bank'], s['human'])}</td></tr>")
    parts.append("</table></div></div>")
    return "".join(parts)


# ---------------------------------------------------------------------------------
# "Try your own statement" tab
# ---------------------------------------------------------------------------------
EXAMPLE = (
    "The Committee judges that inflation remains elevated and that risks to price "
    "stability are tilted to the upside. Accordingly, the Committee decided to raise "
    "the target range for the federal funds rate by 25 basis points. Economic activity "
    "has continued to expand at a moderate pace, although growth in business investment "
    "has slowed. The Committee will continue to monitor incoming data and stands ready "
    "to adjust the stance of policy as appropriate. The next meeting is scheduled for "
    "March 15."
)


def extract_text(path: str) -> str:
    """Text from an uploaded .txt or .pdf, with layout line breaks flattened.

    PDF extraction is approximate: page headers, footers and footnotes come through as
    ordinary text, which is why the result lands in the editable box rather than going
    straight to the model.
    """
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        raw = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        raw = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)        # re-join words hyphenated at line ends
    else:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    return re.sub(r"\s+", " ", raw).strip()


def build_app(backend: Backend, threshold: float):
    import gradio as gr

    samples = load_bank_samples()
    by_label = {bank_choice_label(e): e for e in samples}
    bank_choices = [C.BANK_DISPLAY_NAMES[b] for b in C.BANKS] + [OTHER_BANK]

    def show_bank(label):
        return render_bank(by_label.get(label))

    def run(text, bank):
        return render(analyze(text, backend, threshold), bank, backend,
                      n_total=len(split_sentences(text)))

    def load_file(path):
        if not path:
            return gr.update()
        try:
            text = extract_text(path)
        except Exception as exc:                     # a corrupt or encrypted PDF
            logger.warning("could not read %s: %s", path, exc)
            raise gr.Error("Couldn't read that file. Try a different PDF, or copy and "
                           "paste the text instead.")
        if not text:
            raise gr.Error("No text found in that file. If it's a scanned PDF, it has no "
                           "text layer; copy and paste the statement instead.")
        gr.Info("Text extracted. Check it in the box (PDF headers and footers can sneak "
                "in), then click Analyze.")
        return text

    with gr.Blocks(title="Central Bank Stance Reader") as app:
        gr.Markdown(
            "# Central Bank Stance Reader\n"
            "Central banks raise interest rates to cool inflation and cut them to support "
            "the economy. This model reads central bank statements one sentence at a time "
            "and labels each one **hawkish** (leans toward higher rates), **dovish** "
            "(leans toward lower rates), **neutral**, or **irrelevant** (not about policy).")

        with gr.Tab("Explore the 25 banks"):
            gr.Markdown(
                "**What happens when the model reads a bank it has never seen?** We "
                "trained it 25 times, each time leaving one bank out, then tested it on "
                "that bank. Pick a bank to compare the model trained with it against the "
                "one that never saw it. Banks are sorted by how much the score drops.")
            bank_pick = gr.Dropdown(list(by_label), label="Central bank",
                                    value=next(iter(by_label), None))
            bank_out = gr.HTML()
            bank_pick.change(show_bank, bank_pick, bank_out)

        with gr.Tab("Try your own statement"):
            gr.Markdown(
                "Paste a statement or upload one as a **.txt** or **.pdf** file, pick the "
                "bank that wrote it, then click **Analyze**. You get a label for every "
                "sentence and an overall score from **&minus;1 (fully dovish)** to "
                "**+1 (fully hawkish)**.")
            with gr.Row():
                with gr.Column(scale=3):
                    text = gr.Textbox(label="Statement", lines=10, value=EXAMPLE)
                    upload = gr.File(label="Or upload a statement (.txt or .pdf)",
                                     file_types=[".txt", ".pdf"], type="filepath")
                with gr.Column(scale=2):
                    bank = gr.Dropdown(bank_choices,
                                       value=C.BANK_DISPLAY_NAMES["fomc"],
                                       label="Which central bank wrote it?",
                                       info="Pick the last option if the bank isn't listed.")
                    go = gr.Button("Analyze", variant="primary")
            out = gr.HTML()
            upload.upload(load_file, upload, text)
            go.click(run, [text, bank], out)

        app.load(show_bank, bank_pick, bank_out)
        app.load(run, [text, bank], out)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["baseline", "roberta"], default="baseline")
    ap.add_argument("--checkpoint", default=None, help="path to a fine-tuned RoBERTa dir")
    ap.add_argument("--seed", type=int, default=C.PRIMARY_SEED)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    setup_logging()
    from .data import load_annotated

    if args.backend == "roberta":
        if not args.checkpoint:
            ap.error("--backend roberta requires --checkpoint")
        backend = RobertaBackend(args.checkpoint)
        threshold = cached_threshold(backend, lambda: load_annotated(args.seed),
                                     args.checkpoint, args.seed)
    else:
        # The baseline is refit at every launch, so there is no stable model to key a
        # cache on -- and scoring val with it takes under a second anyway.
        frames = load_annotated(args.seed)
        backend = BaselineBackend(frames)
        threshold = calibrate_threshold(backend, frames)

    build_app(backend, threshold).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
