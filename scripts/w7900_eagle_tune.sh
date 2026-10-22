#!/usr/bin/env bash
set -euo pipefail

arm="${1:?arm}"
port="${2:?port}"
results=/nvme0/data/zhenchen/spec-bench-results/qwen3.8-27b-eagle3-w7900
target=/nvme0/data/zihaomu/bigssd/llamafactroy_workspace/models/Qwen3.8-27B
image=specbench:vllm-0.29.0-rocm723-gfx1100

run_case() {
    local label="$1" isl="$2" osl="$3" conc="$4" prompts="$5" warmups="$6"
    local outdir="$results/tuning/$arm"
    local stem="${label}_isl${isl}_osl${osl}_c${conc}"
    mkdir -p "$outdir"
    if [[ -s "$outdir/$stem.json" ]]; then
        echo "SKIP $arm $stem"
        return
    fi

    docker run --rm \
        --name "client-${arm}-${label}-c${conc}" \
        --network host \
        -v "$target:/models/Qwen3.8-27B:ro" \
        -v "$results:/results" \
        --entrypoint vllm \
        "$image" \
        bench serve \
        --backend openai \
        --base-url "http://127.0.0.1:$port" \
        --endpoint /v1/completions \
        --model Qwen3.8-27B \
        --tokenizer /models/Qwen3.8-27B \
        --dataset-name random \
        --random-input-len "$isl" \
        --random-output-len "$osl" \
        --random-range-ratio 0 \
        --num-prompts "$prompts" \
        --max-concurrency "$conc" \
        --request-rate inf \
        --num-warmups "$warmups" \
        --temperature 0 \
        --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --save-result \
        --save-detailed \
        --result-dir "/results/tuning/$arm" \
        --result-filename "$stem.json" \
        --metadata "arm=$arm" "workload=$label" "isl=$isl" "osl=$osl" "conc=$conc" \
        2>&1 | tee "$outdir/$stem.log"
}

run_case short 128 128 1 8 2
run_case short 128 128 16 32 2
run_case longdecode 512 512 1 4 1
echo "TUNING_DONE $arm"
