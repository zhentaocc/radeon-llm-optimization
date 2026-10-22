# Radeon LLM Optimization

Reproducible performance engineering for vLLM speculative decoding on AMD Radeon GPUs.

The first tracked workload is Qwen3.8-27B with `amd/Qwen3.8-27B-EAGLE3-Full-Vocab` on 2× Radeon Pro W7900D (`gfx1100`, ROCm 7.2, vLLM 0.29.0). The repository keeps raw-result indexes, launch/benchmark scripts, optimization hypotheses, validated deltas, and PR-ready branches separate.

## Current headline

- EAGLE3 is functionally supported on W7900D with target TP=2 and draft TP=1/2.
- Static `N=2`, draft TP=1 is the best general configuration measured so far.
- Static `N=3`, draft TP=2 is narrowly best for short, batch-1 traffic.
- EAGLE3 helps decode-heavy and low-concurrency workloads, but is harmful for 16K context at batch 4/16.
- Real-data validation covers ShareGPT, official SpecBench, and LongBench
  `gov_report` at concurrency 1/4/16. Fixed-256 SpecBench gains are
  +86.0%/+31.2%/+52.1%; 8K-16K LongBench gains +170.6%/+34.7% at c1/c4 but
  loses 44.4% at c16 because EAGLE reduces KV capacity and doubles queueing TTFT.
- Dynamic-N startup is fixed and covered by a regression test, but piecewise graph mode still loses 15.5% at c16, so static-N routing remains recommended.
- HIP graph v1/v2 recover substantial KV capacity, but both fail promotion: v1 collapses on 16K/c16 and v2 loses 4.8% at c1 and 6.7% on long decode.
- A shape-aware RDNA small-M dispatch passes microbenchmarks and 13 unit tests but loses 20.9% at c1 in strict same-GPU A/B; it is retained as a negative result, not a PR candidate.

See [OPTIMIZATION_STATUS.md](OPTIMIZATION_STATUS.md) for the live optimization
table and detailed work items,
[reports/w7900-qwen38-eagle3.md](reports/w7900-qwen38-eagle3.md) for the
synthetic benchmark report, and
[experiments/w7900-realdata-eagle3/](experiments/w7900-realdata-eagle3/) for
the reproducible real-data matrix.

## Repository workflow

`main` contains only reproduced evidence and the consolidated report. Each optimization is developed on its own `opt/*` branch. A branch becomes PR-ready only after it includes:

1. the exact code/config delta;
2. a clean baseline and candidate run on the same scheduler/runtime;
3. throughput, TTFT, TPOT, E2E, p99, acceptance, and memory impact;
4. regression coverage across short/medium/long-context and batch 1/4/16;
5. a concise upstreaming note.

Large checkpoints, profiler traces, and raw JSON stay on the benchmark hosts. This repository stores summaries, manifests, scripts, and compact derived data.

