#!/usr/bin/env bash
set -euo pipefail

arm="${1:?arm: baseline or eagle_n2}"
port="${2:?server port}"
root=/dev/shm/zhenchen-realdata-20261022
image=specbench:vllm-0.29.0-rocm723-gfx1100
target=/nvme0/data/zihaomu/bigssd/llamafactroy_workspace/models/Qwen3.8-27B
sharegpt=/home/zhenchen/speculative-bench-fca1ed6/profile/dataset/sharegpt_custom_qwen35.jsonl
loader_patch="$root/datasets.py"

mkdir -p "$root/results/$arm" "$root/logs/$arm" "$root/metrics/$arm"

until [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/health" || true)" == 200 ]]; do
    sleep 5
done

run_case() {
    local dataset="$1" conc="$2" prompts="$3" warmups="$4"
    local outdir="$root/results/$arm/$dataset"
    local logdir="$root/logs/$arm/$dataset"
    local metricdir="$root/metrics/$arm/$dataset"
    local stem="${dataset}_c${conc}_n${prompts}"
    local container="realdata-client-${arm}-${dataset}-c${conc}"
    mkdir -p "$outdir" "$logdir" "$metricdir"

    if [[ -s "$outdir/${stem}.json" ]]; then
        echo "SKIP existing $arm $stem"
        return
    fi

    curl -s "http://127.0.0.1:${port}/metrics" >"$metricdir/${stem}.pre.prom" || true

    common=(
        --backend openai
        --base-url "http://127.0.0.1:${port}"
        --endpoint /v1/completions
        --model Qwen3.8-27B
        --tokenizer /models/Qwen3.8-27B
        --num-prompts "$prompts"
        --max-concurrency "$conc"
        --request-rate inf
        --num-warmups "$warmups"
        --seed 42
        --temperature 0
        --no-oversample
        --percentile-metrics ttft,tpot,itl,e2el
        --metric-percentiles 50,90,99
        --save-result
        --save-detailed
        --result-dir "/results/results/$arm/$dataset"
        --result-filename "${stem}.json"
        --metadata
        "model=Qwen3.8-27B" "platform=W7900D" "tp=2" "arm=$arm"
        "dataset=$dataset" "conc=$conc" "num_prompts=$prompts"
        "temperature=0" "thinking=off" "ignore_eos=false" "seed=42"
    )

    dataset_args=()
    case "$dataset" in
        sharegpt_short)
            dataset_args=(
                --dataset-name custom
                --dataset-path /datasets/sharegpt_custom_qwen35.jsonl
                --custom-output-len -1
                --chat-template-kwargs '{"enable_thinking":false}'
            )
            ;;
        longbench_gov_report_8k16k)
            dataset_args=(
                --dataset-name custom
                --dataset-path /results/datasets/longbench-gov_report-8k16k-custom.jsonl
                --custom-output-len -1
                --chat-template-kwargs '{"enable_thinking":false}'
                --disable-shuffle
            )
            ;;
        specbench)
            dataset_args=(
                --dataset-name spec_bench
                --dataset-path /results/datasets/specbench-question.jsonl
                --spec-bench-output-len 256
                --chat-template-kwargs '{"enable_thinking":false}'
            )
            ;;
        *)
            echo "unknown dataset: $dataset" >&2
            return 2
            ;;
    esac

    echo "START $(date -Is) arm=$arm dataset=$dataset conc=$conc prompts=$prompts"
    docker run --rm \
        --name "$container" \
        --network host \
        -v "$target:/models/Qwen3.8-27B:ro" \
        -v "$sharegpt:/datasets/sharegpt_custom_qwen35.jsonl:ro" \
        -v "$root:/results" \
        -v "$loader_patch:/opt/venv/lib/python3.12/site-packages/vllm/benchmarks/datasets/datasets.py:ro" \
        --entrypoint vllm \
        "$image" \
        bench serve "${common[@]}" "${dataset_args[@]}" \
        2>&1 | tee "$logdir/${stem}.log"

    curl -s "http://127.0.0.1:${port}/metrics" >"$metricdir/${stem}.post.prom" || true
    echo "DONE $(date -Is) arm=$arm dataset=$dataset conc=$conc prompts=$prompts"
}

for conc in 1 4 16; do
    case "$conc" in
        1) prompts=16 ;;
        4) prompts=32 ;;
        16) prompts=64 ;;
    esac
    run_case sharegpt_short "$conc" "$prompts" 2
done

for conc in 1 4 16; do
    case "$conc" in
        1) prompts=16 ;;
        4) prompts=32 ;;
        16) prompts=64 ;;
    esac
    run_case specbench "$conc" "$prompts" 2
done

for conc in 1 4 16; do
    case "$conc" in
        1) prompts=4 ;;
        4) prompts=8 ;;
        16) prompts=16 ;;
    esac
    run_case longbench_gov_report_8k16k "$conc" "$prompts" 1
done

date -Is >"$root/results/$arm/COMPLETE"
echo "ARM_COMPLETE $(date -Is) arm=$arm"

