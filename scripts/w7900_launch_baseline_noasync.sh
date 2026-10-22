#!/usr/bin/env bash
set -euo pipefail

container="${1:?container name}"
gpus="${2:?physical GPU list}"
port="${3:?port}"
target=/nvme0/data/zihaomu/bigssd/llamafactroy_workspace/models/Qwen3.8-27B
results=/nvme0/data/zhenchen/spec-bench-results/qwen3.8-27b-eagle3-w7900
image=specbench:vllm-0.29.0-rocm723-gfx1100
cache=/nvme0/data/zhenchen/vllm-cache/$container

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
    --gpu-memory-utilization 0.88 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    >"$results/servers/$container/container_id.txt"

cat >"$results/servers/$container/run_config.txt" <<EOF
container=$container
gpus=$gpus
port=$port
tensor_parallel_size=2
async_scheduling=false
gpu_memory_utilization=0.88
EOF

echo "$container"
