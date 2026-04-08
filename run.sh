#!/bin/bash
# Experiment entrypoint — called by CER inside the Singularity container.
#
# Environment variables set by CER:
#   CER_COMMIT      - full git commit hash
#   WANDB_PROJECT   - W&B project name (from cer.yaml)
#   WANDB_RUN_NAME  - cer-<commit_short>
#   WANDB_TAGS      - commit hash (for querying via `cer results`)

set -euo pipefail

# Install extra dependencies (Bug fix: was missing)
if [ -f requirements.txt ]; then
    pip install --quiet -r requirements.txt
fi

python train.py
