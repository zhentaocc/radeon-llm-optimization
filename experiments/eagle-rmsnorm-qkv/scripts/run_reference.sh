#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH=/workspace
export VLLM_ROCM_USE_FUSED_RMSNORM_QKV=0
python test_reference.py
