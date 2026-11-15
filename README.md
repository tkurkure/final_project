# Cross-Bank Monetary Policy Stance Detection

Sentence-level stance classification (hawkish / dovish / neutral /
irrelevant) of central-bank communications, built on the World Central Banks (WCB)
dataset of Shah et al. (2025).

**Live demo:** https://huggingface.co/spaces/f006tsj/central-bank-stance-reader

## Research questions

1. **How much worse does a model do on a bank it wasn't trained on?**
2. **What about a bank's language predicts that drop?**

The WCB paper benchmarks a pooled model and per-bank models, but never does
leave-one-bank-out; its only generalization test was a single unseen 26th bank. The
systematic 25-fold LOBO sweep here is the contribution.

## Experimental design

Three conditions, all scored on the **same** test sentences for a given bank:

| Condition | Trained on | Answers |
|---|---|---|
| `pooled` | all 25 banks (17,468 sentences) | in-distribution ceiling |
| `lobo` | the other 24 banks (~16,770) | performance on an unseen bank |
| `size_control` | a *random* subset of equal size | is the gap just "less data"? |

The gap decomposes as:

```
gap_total = pooled - lobo            # total cost of the bank being absent
gap_data  = pooled - size_control    # cost of ~4% less training data
gap_bank  = size_control - lobo      # cost of it being THAT bank
```

The size control is what makes the headline claim defensible: without it, `gap_total`
confounds "this bank is distinctive" with "this model saw 4% fewer sentences".

Gaps use a **paired bootstrap** (both conditions score identical items, so resampling
items once and differencing preserves their error correlation — empirically ~2x tighter
than differencing independent intervals).

## Layout

```
src/wcb/
  config.py       bank names, label maps, paths, hyperparameters
  data.py         loading, dedup, integrity assertions
  splits.py       the three conditions + leakage assertions
  baseline.py     TF-IDF + logistic regression
  train.py        RoBERTa fine-tuning (plain PyTorch loop)
  evaluate.py     metrics, bootstrap, paired bootstrap
  features.py     six pre-registered bank-level features
  analysis.py     gap tables, decomposition, correlations
  results_io.py   per-fold JSON with per-sentence predictions
scripts/
  run_baseline_all.py   CPU sweep: 25 banks x 3 seeds x 3 conditions
  run_roberta.py        GPU workhorse; resumable
  make_report.py        every table and figure from results/ alone
  prior_shift.py        how much of the gap is just a shifted stance mix
  build_demo_samples.py per-bank sample sentences for the demo's Explore tab
  deploy_space.sh       publish the Gradio demo to a Hugging Face Space (needs PRO)
  export_web_model.py   ONNX export of the demo model for the browser version
  deploy_web.sh         publish the browser version (free static Space + model repo)
space/                  the Gradio Space's app.py, README and CPU requirements
web/                    the browser version: index.html, worker.js, Space README
notebooks/              thin drivers that import from src/wcb
results/                one JSON per fold (committed)
```

## Running

```bash
# local (macOS / CPU)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[demo]"

# on a GPU cluster that already has torch -- do not reinstall torch, it would
# replace the CUDA-matched build with a generic one
pip install -e ".[demo]"

# CPU: complete transfer-gap picture, ~8 minutes
.venv/bin/python scripts/run_baseline_all.py

# GPU: time one fold before committing to the sweep
.venv/bin/python scripts/run_roberta.py --condition lobo --banks ecb --time-only

# GPU: choose hyperparameters once, then the Tier-1 sweep
.venv/bin/python scripts/run_roberta.py --hparam-search
.venv/bin/python scripts/run_roberta.py --condition pooled size_control lobo

# check everything, and rebuild all tables and figures
.venv/bin/python scripts/verify.py --model roberta \
    --checkpoint results/roberta/seed5768/pooled_model
.venv/bin/python scripts/make_report.py --model roberta

# the demo
.venv/bin/python -m wcb.demo                      # TF-IDF backend, runs today
.venv/bin/python -m wcb.demo --backend roberta \
    --checkpoint results/roberta/seed5768/pooled_model

# the live browser demo: export the model, then publish (needs `hf auth login`)
uv pip install --python .venv/bin/python -e ".[web]"
.venv/bin/python scripts/export_web_model.py \
    --checkpoint results/roberta/seed5768/pooled_model
scripts/deploy_web.sh f006tsj
```

Every run is **resumable** — a fold whose results file exists is skipped — so an
8-hour JupyterHub session that times out costs one fold, not the sweep.

## Data notes

These cost real debugging time, so they are recorded here:

- `bank_name` values are **not** the README slugs: it is `fomc`, `ecb`, `rbi`,
  `brazil_copom`, `bank_of_phillipines` (sic) — not `federal_reserve_system` etc.
- Label values are **lowercase**, and time labels have no hyphen:
  `"not forward looking"`.
- The 70/15/15 split is **global, not stratified by bank**. Each bank has exactly 1,000
  sentences, but per-bank test counts range 128 (ECB) to 176 (Russia). The design only
  needs both conditions to score identical items, which still holds.
- The dataset's three configs (`5768`, `78516`, `944601`) are three different
  *partitions*, not three training seeds — so averaging over them estimates split
  sensitivity, which matters when a per-bank test set has ~150 items.
- Token lengths: p50=31, p99=90, max=305. `max_length=128` covers 99.78%.
- Cross-split duplicates exist (4 train/test, 6 train/val, 1 val/test). They are removed
  from **train and val only** — the test split is never modified, or the published
  numbers stop being comparable.

## Reproduction checkpoints

From the WCB paper, used to validate the pipeline before spending GPU time:

| Check | Target |
|---|---|
| RoBERTa-base, pooled, mean weighted F1 | 0.714 |
| Czech National Bank hold-out | 0.800 |
