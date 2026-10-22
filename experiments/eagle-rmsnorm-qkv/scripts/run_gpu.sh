#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export PYTHONPATH=/workspace
: "${HIP_VISIBLE_DEVICES:?HIP_VISIBLE_DEVICES must select the approved physical GPU}"
export VLLM_ROCM_USE_FUSED_RMSNORM_QKV=1
export VLLM_ROCM_USE_SKINNY_GEMM=1
python test_gpu.py
