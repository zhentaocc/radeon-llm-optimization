# Radeon LLM Optimization

Reproducible performance engineering for vLLM speculative decoding on AMD Radeon GPUs.

The first tracked workload is Qwen3.8-27B with `amd/Qwen3.8-27B-EAGLE3-Full-Vocab` on 2× Radeon Pro W7900D (`gfx1100`, ROCm 7.2, vLLM 0.29.0). The repository keeps raw-result indexes, launch/benchmark scripts, optimization hypotheses, validated deltas, and PR-ready branches separate.

## Current headline

- EAGLE3 is functionally supported on W7900D with target TP=2 and draft TP=1/2.
- Static `N=2`, draft TP=1 is the best general configuration measured so far.
- Static `N=3`, draft TP=2 is narrowly best for short, batch-1 traffic.
- EAGLE3 helps decode-heavy and low-concurrency workloads, but is harmful for 16K context at batch 4/16.
- Dynamic-N startup is fixed and covered by a regression test, but piecewise graph mode still loses 15.5% at c16, so static-N routing remains recommended.
- A reduced HIP graph capture set raises KV capacity from about 125K to 272K tokens; v2 adds the missing verification shapes before promotion.
- Shape-aware RDNA small-M GEMM dispatch is microbenchmark-validated and passes 13 tests; the full service gate awaits recovery from a host HIP IPC failure.

See [OPTIMIZATION_STATUS.md](OPTIMIZATION_STATUS.md) for the live optimization table and detailed work items, and [reports/w7900-qwen38-eagle3.md](reports/w7900-qwen38-eagle3.md) for the benchmark report.

## Repository workflow

`main` contains only reproduced evidence and the consolidated report. Each optimization is developed on its own `opt/*` branch. A branch becomes PR-ready only after it includes:

1. the exact code/config delta;
2. a clean baseline and candidate run on the same scheduler/runtime;
3. throughput, TTFT, TPOT, E2E, p99, acceptance, and memory impact;
4. regression coverage across short/medium/long-context and batch 1/4/16;
5. a concise upstreaming note.

Large checkpoints, profiler traces, and raw JSON stay on the benchmark hosts. This repository stores summaries, manifests, scripts, and compact derived data.

