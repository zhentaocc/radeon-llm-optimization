#!/usr/bin/env bash
set -euo pipefail

arm="${1:?arm}"
port="${2:?port}"
results=/nvme0/data/zhenchen/spec-bench-results/qwen3.8-27b-eagle3-w7900
target=/nvme0/data/zihaomu/bigssd/llamafactroy_workspace/models/Qwen3.8-27B
image=specbench:vllm-0.29.0-rocm723-gfx1100

run_case() {
    local label="$1" isl="$2" osl="$3" conc="$4" prompts="$5" warmups="$6"
    local outdir="$results/$arm/matrix"
    local stem="${label}_isl${isl}_osl${osl}_c${conc}_r1"
    mkdir -p "$outdir"
    if [[ -s "$outdir/$stem.json" ]]; then
        echo "SKIP $arm $stem"
        return
    fi

    echo "===== $arm $stem $(date -Is) ====="
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
        --result-dir "/results/$arm/matrix" \
        --result-filename "$stem.json" \
        --metadata "arm=$arm" "workload=$label" "isl=$isl" "osl=$osl" "conc=$conc" \
        2>&1 | tee "$outdir/$stem.log"
}

for spec in "short 128 128" "medium 2048 256" "longctx 16384 256" "longdecode 512 512"; do
    read -r label isl osl <<< "$spec"
    for conc in 1 4 16; do
        if [[ "$label" == longctx ]]; then
            warmups=1
            case "$conc" in 1) prompts=3;; 4) prompts=8;; 16) prompts=32;; esac
        elif [[ "$label" == longdecode ]]; then
            warmups=2
            case "$conc" in 1) prompts=4;; 4) prompts=8;; 16) prompts=32;; esac
        else
            warmups=4
            case "$conc" in 1) prompts=8;; 4) prompts=16;; 16) prompts=64;; esac
        fi
        run_case "$label" "$isl" "$osl" "$conc" "$prompts" "$warmups"
    done
done

echo "MATRIX_DONE $arm"
