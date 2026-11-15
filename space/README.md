---
title: Central Bank Stance Reader
emoji: 🏦
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: How well does a stance model read a bank it never saw?
---

# Central Bank Stance Reader

CS 72 final project demo. A fine-tuned RoBERTa model labels each sentence of a central
bank statement as hawkish, dovish, neutral, or irrelevant.

- **Explore the 25 banks:** real test sentences from each bank, comparing the model
  trained with that bank against one trained without it (leave-one-bank-out).
- **Try your own statement:** paste text or upload a .txt or .pdf.

Data: the World Central Banks (WCB) annotated dataset, Shah et al. (2025),
`gtfintechlab/all_annotated_sentences_25000`.
