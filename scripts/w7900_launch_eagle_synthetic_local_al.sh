#!/usr/bin/env bash
set -euo pipefail

# Launch a non-official, locally calibrated synthetic-acceptance server.
# Machine-local model/cache/results paths must be supplied through environment
# variables so no private host layout is recorded in Git.
container="${1:?usage: $0 CONTAINER GPU_LIST PORT CALIBRATION_MANIFEST [DRAFT_TP] [GPU_MEMORY_UTIL]}"
gpus="${2:?physical GPU list}"
port="${3:?port}"
calibration_manifest="${4:?local_calibration_manifest.json}"
draft_tp="${5:-1}"
gpu_memory_utilization="${6:-0.88}"

: "${W7900_TARGET_MODEL:?set W7900_TARGET_MODEL}"
: "${W7900_DRAFT_MODEL:?set W7900_DRAFT_MODEL}"
: "${W7900_RESULTS_ROOT:?set W7900_RESULTS_ROOT}"
: "${W7900_VLLM_CACHE_ROOT:?set W7900_VLLM_CACHE_ROOT}"
: "${W7900_VLLM_IMAGE:?set W7900_VLLM_IMAGE}"

if [[ "${ALLOW_SYNTHETIC_SERVER_START:-0}" != "1" ]]; then
    echo "Refusing to start: set ALLOW_SYNTHETIC_SERVER_START=1 after exclusive resource confirmation." >&2
    exit 2
fi
if [[ ",$gpus," == *,1,* ]]; then
    echo "GPU1 is out of scope and must not be used." >&2
    exit 2
fi
if [[ ! -s "$calibration_manifest" ]]; then
    echo "Missing calibration manifest: $calibration_manifest" >&2
    exit 2
fi

local_al="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["calibration_kind"]=="local_calibration" and d["submission_valid"] is False and d["official"] is False; x=float(d["acceptance_length"]["raw"]); assert 1.0 <= x <= 3.0; print(format(x, ".12g"))' "$calibration_manifest")"
cache="${W7900_VLLM_CACHE_ROOT}/${container}"
results="${W7900_RESULTS_ROOT}/${container}"
spec_config="{\"method\":\"eagle3\",\"model\":\"/models/draft\",\"num_speculative_tokens\":2,\"draft_tensor_parallel_size\":${draft_tp},\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":${local_al}}"

if docker inspect "$container" >/dev/null 2>&1; then
    echo "container already exists: $container" >&2
    exit 2
fi
mkdir -p "$results" "$cache"
cp "$calibration_manifest" "$results/local_calibration_manifest.json"

docker run -d \
  --name "$container" --network host --ipc host --shm-size 32g \
  --device=/dev/kfd --device=/dev/dri --group-add video --ulimit memlock=-1:-1 \
  -e "HIP_VISIBLE_DEVICES=$gpus" -e VLLM_ROCM_USE_SKINNY_GEMM=1 \
  -v "$W7900_TARGET_MODEL:/models/target:ro" \
  -v "$W7900_DRAFT_MODEL:/models/draft:ro" \
  -v "$cache:/root/.cache/vllm" \
  --entrypoint vllm "$W7900_VLLM_IMAGE" \
  serve /models/target --served-model-name Qwen3.8-27B \
  --host 127.0.0.1 --port "$port" --tensor-parallel-size 2 \
  --language-model-only --max-model-len 32768 \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --max-num-seqs 64 --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching --trust-remote-code \
  --speculative-config "$spec_config" >"$results/container_id.txt"

printf '%s\n' "$container"
