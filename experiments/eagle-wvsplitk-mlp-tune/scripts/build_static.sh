#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export MAX_JOBS="${MAX_JOBS:-16}"
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx1100
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.29.0

if [[ ! -d /workspace/.build-tools/cmake ]]; then
  python -m pip install --disable-pip-version-check --no-cache-dir \
    --target /workspace/.build-tools cmake ninja
fi
export PATH="/workspace/.build-tools/bin:$PATH"
export PYTHONPATH="/workspace/.build-tools:${PYTHONPATH:-}"

python setup.py build_ext --inplace
