#!/usr/bin/env bash
set -euo pipefail

arm="${1:?arm}"
port="${2:?port}"
include_long="${3:-0}"
root=/dev/shm/zhenchen-realdata-20261022
image=specbench:vllm-0.29.0-rocm723-gfx1100
target=/nvme0/data/zihaomu/bigssd/llamafactroy_workspace/models/Qwen3.8-27B
loader_patch="$root/datasets.py"

until [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/health" || true)" == 200 ]]; do
    sleep 5
done

run_fixed() {
    local label="$1" conc="$2" prompts="$3" warmups="$4"
    local outdir="$root/results/$arm/$label"
    local logdir="$root/logs/$arm/$label"
    local metricdir="$root/metrics/$arm/$label"
    local stem="${label}_c${conc}_n${prompts}"
    local client="realdata-client-${arm}-${label}-c${conc}"
    mkdir -p "$outdir" "$logdir" "$metricdir"
    if [[ -s "$outdir/${stem}.json" ]]; then
        echo "SKIP existing $arm $stem"
        return
    fi

    args=()
    case "$label" in
        specbench_fixed256)
            args=(
                --dataset-name spec_bench
                --dataset-path /results/datasets/specbench-question.jsonl
                --spec-bench-output-len 256
                --chat-template-kwargs '{"enable_thinking":false}'
            )
            ;;
        longbench_gov_report_8k16k_fixed512)
            args=(
                --dataset-name custom
                --dataset-path /results/datasets/longbench-gov_report-8k16k-custom.jsonl
                --custom-output-len 512
                --chat-template-kwargs '{"enable_thinking":false}'
                --disable-shuffle
            )
            ;;
        *) return 2 ;;
    esac

    curl -s "http://127.0.0.1:${port}/metrics" >"$metricdir/${stem}.pre.prom" || true
    echo "START_FIXED $(date -Is) arm=$arm dataset=$label conc=$conc prompts=$prompts"
    docker run --rm \
        --name "$client" \
        --network host \
        -v "$target:/models/Qwen3.8-27B:ro" \
        -v "$root:/results" \
        -v "$loader_patch:/opt/venv/lib/python3.12/site-packages/vllm/benchmarks/datasets/datasets.py:ro" \
        --entrypoint vllm \
        "$image" \
        bench serve \
        --backend openai \
        --base-url "http://127.0.0.1:${port}" \
        --endpoint /v1/completions \
        --model Qwen3.8-27B \
        --tokenizer /models/Qwen3.8-27B \
        --num-prompts "$prompts" \
        --max-concurrency "$conc" \
        --request-rate inf \
        --num-warmups "$warmups" \
        --seed 42 \
        --temperature 0 \
        --ignore-eos \
        --no-oversample \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --save-result --save-detailed \
        --result-dir "/results/results/$arm/$label" \
        --result-filename "${stem}.json" \
        --metadata \
            "model=Qwen3.8-27B" "platform=W7900D" "tp=2" "arm=$arm" \
            "dataset=$label" "conc=$conc" "num_prompts=$prompts" \
            "temperature=0" "thinking=off" "ignore_eos=true" "seed=42" \
        "${args[@]}" \
        2>&1 | tee "$logdir/${stem}.log"
    curl -s "http://127.0.0.1:${port}/metrics" >"$metricdir/${stem}.post.prom" || true
    echo "DONE_FIXED $(date -Is) arm=$arm dataset=$label conc=$conc prompts=$prompts"
}

for conc in 1 4 16; do
    case "$conc" in
        1) prompts=16 ;;
        4) prompts=32 ;;
        16) prompts=64 ;;
    esac
    run_fixed specbench_fixed256 "$conc" "$prompts" 2
done

if [[ "$include_long" == 1 ]]; then
    run_fixed longbench_gov_report_8k16k_fixed512 1 4 1
    run_fixed longbench_gov_report_8k16k_fixed512 16 16 1
fi

date -Is >"$root/results/$arm/FIXED_COMPLETE"
echo "FIXED_ARM_COMPLETE $(date -Is) arm=$arm include_long=$include_long"

