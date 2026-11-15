---
title: Central Bank Stance Reader
emoji: 🏦
colorFrom: blue
colorTo: red
sdk: static
app_file: index.html
pinned: false
short_description: How well does a stance model read a bank it never saw?
models:
  - f006tsj/central-bank-stance-roberta
# Cross-origin isolation lets the in-browser model use several CPU cores.
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# Central Bank Stance Reader

CS 72 final project demo. A fine-tuned RoBERTa model labels each sentence of a central
bank statement as hawkish, dovish, neutral, or irrelevant. It runs entirely in the
visitor's browser (transformers.js), so no text leaves the page.

- **Explore the 25 banks:** real test sentences from each bank, comparing the model
  trained with that bank against one trained without it (leave-one-bank-out).
- **Try your own statement:** paste text or upload a .txt or .pdf.

Data: the World Central Banks (WCB) annotated dataset, Shah et al. (2025),
`gtfintechlab/all_annotated_sentences_25000`.
