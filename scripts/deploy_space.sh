#!/usr/bin/env bash
# Publish the demo to a Hugging Face Space. Needs `hf auth login` (a write token) first.
#
#   scripts/deploy_space.sh <username>/central-bank-stance-reader
#
# Safe to re-run: each upload only sends files that changed.
set -euo pipefail

SPACE="${1:?usage: $0 <username>/<space-name>}"
CHECKPOINT="${CHECKPOINT:-cs72_final_project-2/results/roberta/seed5768/pooled_model}"
HF="${HF:-.venv/bin/hf}"

cd "$(dirname "$0")/.."

"$HF" repos create "$SPACE" --repo-type space --space-sdk gradio --public --exist-ok
"$HF" upload "$SPACE" space . --repo-type space --commit-message "App and Space config"
"$HF" upload "$SPACE" src/wcb src/wcb --repo-type space --exclude "__pycache__/*" \
    --commit-message "wcb package"
"$HF" upload "$SPACE" results/demo_bank_samples.json results/demo_bank_samples.json \
    --repo-type space --commit-message "Per-bank sample sentences"
"$HF" upload "$SPACE" "$CHECKPOINT" model --repo-type space --commit-message "RoBERTa checkpoint"

echo "Building at https://huggingface.co/spaces/$SPACE"
