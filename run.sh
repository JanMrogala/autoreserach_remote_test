#!/bin/bash
set -euo pipefail

pip install -q -r requirements.txt 2>/dev/null

python train.py "$@"
