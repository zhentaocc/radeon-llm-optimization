#!/usr/bin/env bash
set -euo pipefail

image=specbench:vllm-0.29.0-rocm723-gfx1100
target=/nvme0/data/zihaomu/bigssd/llamafactroy_workspace/models/Qwen3.8-27B
draft=/nvme0/data/zhenchen/models/Qwen3.8-27B-EAGLE3-Full-Vocab

common=(
    serve /models/Qwen3.8-27B
    --served-model-name Qwen3.8-27B
    --host 127.0.0.1
    --tensor-parallel-size 2
    --language-model-only
    --max-model-len 32768
    --gpu-memory-utilization 0.88
    --max-num-seqs 64
    --max-num-batched-tokens 16384
    --no-enable-prefix-caching
    --trust-remote-code
)

docker run -d --name realdata-20261022-baseline --network host \
    --ipc host --shm-size 32g \
    --device /dev/kfd --device /dev/dri --group-add video \
    -e HIP_VISIBLE_DEVICES=2,3 \
    -e VLLM_SERVER_DEV_MODE=1 \
    -e VLLM_ROCM_USE_SKINNY_GEMM=1 \
    -v "$target:/models/Qwen3.8-27B:ro" \
    --entrypoint vllm "$image" "${common[@]}" --port 18120

docker run -d --name realdata-20261022-eagle-n2 --network host \
    --ipc host --shm-size 32g \
    --device /dev/kfd --device /dev/dri --group-add video \
    -e HIP_VISIBLE_DEVICES=4,5 \
    -e VLLM_ROCM_USE_SKINNY_GEMM=1 \
    -e VLLM_DISABLE_PYNCCL=0 \
    -v "$target:/models/Qwen3.8-27B:ro" \
    -v "$draft:/models/draft:ro" \
    --entrypoint vllm "$image" "${common[@]}" --port 18121 \
    --speculative-config \
    '{"method":"eagle3","model":"/models/draft","num_speculative_tokens":2,"draft_tensor_parallel_size":1}'
