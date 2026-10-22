#!/usr/bin/env bash
set -euo pipefail

# Non-submittable AgentX-derived optimization smoke. The caller supplies all
# machine-local paths; this repository intentionally contains no host layout.
port="${1:?usage: $0 PORT LABEL CONCURRENCY [DURATION_SECONDS]}"
label="${2:?usage: $0 PORT LABEL CONCURRENCY [DURATION_SECONDS]}"
concurrency="${3:?usage: $0 PORT LABEL CONCURRENCY [DURATION_SECONDS]}"
duration="${4:-60}"

: "${AGENTX_WORK_ROOT:?set AGENTX_WORK_ROOT}"
: "${AGENTX_VENV:?set AGENTX_VENV}"
: "${AGENTX_TOKENIZER:?set AGENTX_TOKENIZER}"
: "${AGENTX_DERIVED_DATASET:?set AGENTX_DERIVED_DATASET}"

out="${AGENTX_WORK_ROOT}/results/${label}-c${concurrency}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HOME="${HF_HOME:-${AGENTX_WORK_ROOT}/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${AGENTX_WORK_ROOT}/cache}"
export TMPDIR="${TMPDIR:-${AGENTX_WORK_ROOT}/tmp}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${AGENTX_WORK_ROOT}/pycache}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT="${AIPERF_DATASET_CONFIGURATION_TIMEOUT:-1800}"
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT:-1800}"
export AIPERF_DATASET_MMAP_BASE_PATH="${AIPERF_DATASET_MMAP_BASE_PATH:-${AGENTX_WORK_ROOT}/mmap}"
export AIPERF_DATASET_MMAP_CACHE_DIR="${AIPERF_DATASET_MMAP_CACHE_DIR:-${XDG_CACHE_HOME}/aiperf/dataset_mmap}"

mkdir -p "$out" "$AIPERF_DATASET_MMAP_BASE_PATH" "$AIPERF_DATASET_MMAP_CACHE_DIR" "$PYTHONPYCACHEPREFIX"

exec "${AGENTX_VENV}/bin/agentx" profile \
  --scenario inferencex-agentx-mvp \
  --unsafe-override \
  --url "http://127.0.0.1:${port}" \
  --endpoint-type chat \
  --model Qwen3.8-27B \
  --tokenizer "$AGENTX_TOKENIZER" \
  --max-context-length 32768 \
  --custom-dataset-type weka_trace \
  --input-file "$AGENTX_DERIVED_DATASET" \
  --concurrency "$concurrency" \
  --benchmark-duration "$duration" \
  --benchmark-grace-period 600 \
  --agentic-warmup-grace-period 600 \
  --trajectory-start-min-ratio 0.0 \
  --trajectory-start-max-ratio 1.0 \
  --random-seed 20261022 \
  --use-server-token-count \
  --streaming \
  --extra-inputs ignore_eos:true \
  --cache-bust first_turn_prefix \
  --system-idle-gap-cap-seconds 10 \
  --output-artifact-dir "$out" \
  --ui simple
