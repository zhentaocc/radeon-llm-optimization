# Radeon LLM Optimization

## Current branch: first-layer RMSNorm and concatenation

Status: **micro-validated / E2E infrastructure-blocked**. This branch carries an independent,
stock-based Qwen3 EAGLE3 layer0 input-preparation fusion under
[`experiments/eagle-layer0-rmsnorm-cat`](experiments/eagle-layer0-rmsnorm-cat/README.md).
The measured 2.00-2.44x result applies only to the isolated kernel region. Two
TP2 server attempts failed in PYNCCL initialization before model execution, so
no service-level claim is made.

Reproducible performance engineering for vLLM speculative decoding on AMD Radeon GPUs.

The first tracked workload is Qwen3.8-27B with `amd/Qwen3.8-27B-EAGLE3-Full-Vocab` on 2× Radeon Pro W7900D (`gfx1100`, ROCm 7.2, vLLM 0.29.0). The repository keeps raw-result indexes, launch/benchmark scripts, optimization hypotheses, validated deltas, and PR-ready branches separate.

## Current headline

- EAGLE3 is functionally supported on W7900D with target TP=2 and draft TP=1/2.
- Static `N=2`, draft TP=1 is the best general configuration measured so far.
- Static `N=3`, draft TP=2 is narrowly best for short, batch-1 traffic.
- EAGLE3 helps decode-heavy and low-concurrency workloads, but is harmful for 16K context at batch 4/16.
- Dynamic-N startup is fixed and covered by a regression test, but piecewise graph mode still loses 15.5% at c16, so static-N routing remains recommended.
- HIP graph v1/v2 recover substantial KV capacity, but both fail promotion: v1 collapses on 16K/c16 and v2 loses 4.8% at c1 and 6.7% on long decode.
- A shape-aware RDNA small-M dispatch passes microbenchmarks and 13 unit tests but loses 20.9% at c1 in strict same-GPU A/B; it is retained as a negative result, not a PR candidate.

See [OPTIMIZATION_STATUS.md](OPTIMIZATION_STATUS.md) for the live optimization table and detailed work items, and [reports/w7900-qwen38-eagle3.md](reports/w7900-qwen38-eagle3.md) for the benchmark report.

## Repository workflow

`main` contains only reproduced evidence and the consolidated report. Each optimization is developed on its own `opt/*` branch. A branch becomes PR-ready only after it includes:

1. the exact code/config delta;
2. a clean baseline and candidate run on the same scheduler/runtime;
3. throughput, TTFT, TPOT, E2E, p99, acceptance, and memory impact;
4. regression coverage across short/medium/long-context and batch 1/4/16;
5. a concise upstreaming note.

Large checkpoints, profiler traces, and raw JSON stay on the benchmark hosts. This repository stores summaries, manifests, scripts, and compact derived data.

