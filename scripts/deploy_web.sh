#!/usr/bin/env bash
# Publish the browser demo: the model to a Hugging Face model repo, the page to a free
# static Space. Needs `hf auth login` (a write token) and scripts/export_web_model.py first.
#
#   scripts/deploy_web.sh f006tsj
#
# Safe to re-run: each upload only sends files that changed. If you change the user or
# model name, update MODEL_ID in web/index.html and `models:` in web/README.md to match.
set -euo pipefail

USER="${1:?usage: $0 <hf-username>}"
MODEL="$USER/central-bank-stance-roberta"
SPACE="$USER/central-bank-stance-reader"
HF="${HF:-.venv/bin/hf}"

cd "$(dirname "$0")/.."
test -f build/web_model/onnx/model.onnx || { echo "run scripts/export_web_model.py first"; exit 1; }

"$HF" repos create "$MODEL" --repo-type model --public --exist-ok
"$HF" upload "$MODEL" build/web_model . --commit-message "fp32 ONNX export for transformers.js"

"$HF" repos create "$SPACE" --repo-type space --space-sdk static --public --exist-ok
"$HF" upload "$SPACE" web . --repo-type space --commit-message "Browser demo"
"$HF" upload "$SPACE" results/demo_bank_samples.json demo_bank_samples.json \
    --repo-type space --commit-message "Per-bank sample sentences"

echo "Live at https://huggingface.co/spaces/$SPACE"
