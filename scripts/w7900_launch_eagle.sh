#!/usr/bin/env bash
set -euo pipefail

container="${1:?container name}"
gpus="${2:?physical GPU list, e.g. 4,5}"
port="${3:?port}"
num_spec_tokens="${4:?num speculative tokens}"
draft_tp="${5:-1}"
gpu_memory_utilization="${6:-0.88}"
batch_schedule="${7:-}"

target=/nvme0/data/zihaomu/bigssd/llamafactroy_workspace/models/Qwen3.8-27B
draft=/nvme0/data/zhenchen/models/Qwen3.8-27B-EAGLE3-Full-Vocab
results=/nvme0/data/zhenchen/spec-bench-results/qwen3.8-27b-eagle3-w7900
image=specbench:vllm-0.29.0-rocm723-gfx1100
spec_config="{\"method\":\"eagle3\",\"model\":\"/models/draft\",\"num_speculative_tokens\":${num_spec_tokens},\"draft_tensor_parallel_size\":${draft_tp}}"
if [[ -n "$batch_schedule" ]]; then
    spec_config="{\"method\":\"eagle3\",\"model\":\"/models/draft\",\"num_speculative_tokens\":${num_spec_tokens},\"draft_tensor_parallel_size\":${draft_tp},\"num_speculative_tokens_per_batch_size\":${batch_schedule}}"
fi
cache=/nvme0/data/zhenchen/vllm-cache/$container

if docker inspect "$container" >/dev/null 2>&1; then
    echo "container already exists: $container" >&2
    exit 2
fi

mkdir -p "$results/servers/$container" "$cache"
docker run -d \
    --name "$container" \
    --network host \
    --ipc host \
    --shm-size 32g \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ulimit memlock=-1:-1 \
    -e "HIP_VISIBLE_DEVICES=$gpus" \
    -v "$target:/models/Qwen3.8-27B:ro" \
    -v "$draft:/models/draft:ro" \
    -v "$cache:/root/.cache/vllm" \
    --entrypoint vllm \
    "$image" \
    serve /models/Qwen3.8-27B \
    --served-model-name Qwen3.8-27B \
    --host 127.0.0.1 \
    --port "$port" \
    --tensor-parallel-size 2 \
    --language-model-only \
    --max-model-len 32768 \
    --gpu-memory-utilization "$gpu_memory_utilization" \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --speculative-config "$spec_config" \
    >"$results/servers/$container/container_id.txt"

cat >"$results/servers/$container/run_config.txt" <<EOF
container=$container
gpus=$gpus
port=$port
num_speculative_tokens=$num_spec_tokens
draft_tensor_parallel_size=$draft_tp
gpu_memory_utilization=$gpu_memory_utilization
num_speculative_tokens_per_batch_size=$batch_schedule
speculative_config=$spec_config
EOF

echo "$container"
