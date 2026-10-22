#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH=/workspace
: "${HIP_VISIBLE_DEVICES:?HIP_VISIBLE_DEVICES must select physical GPU0}"
python test_stage_a.py
