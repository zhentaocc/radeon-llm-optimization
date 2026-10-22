# Qwen3.8-27B + EAGLE3 on Radeon W7900D

Updated: 2026-10-22  
Status: EAGLE3 functional/performance matrix and parameter sweep complete; strict identical-scheduler baseline is running.

## Executive summary

- The AMD full-vocabulary EAGLE3 checkpoint loads and serves correctly on W7900D with vLLM 0.29.0, ROCm 7.2, target TP=2, and draft TP=1/2.
- Static N=2, draft TP=1 is the best general setting. Against the existing async target-only baseline it improves output throughput by 58% for 128/128 c1, 83% for 2048/256 c1, and 54% for 512/512 c1.
- The optimization is workload-sensitive. For 16K/256 it is +128% at c1 but −21% at c4 and −58% at c16; speculation must be disabled or routed away for high-concurrency long-prefill traffic.
- N=3 is best for low concurrency; N=2 is best for high concurrency. N=5/7 consistently lose because additional draft/verification cost grows faster than accepted-token length.
- Draft TP2 gives a narrow +3.6% win for short c1, but loses 1–3% on long output/high batch. Draft TP1 remains the default.
- Static EAGLE3 already uses HIP graph capture. The next graph task is memory/capture-set optimization, not basic enablement.
- Dynamic per-batch N is present in the build but mixed N=3/N=2 fails at startup with a deterministic divide-by-zero in graph candidate construction.

## Environment and methodology

- GPU: 2× Radeon Pro W7900D 48 GiB per service, `gfx1100`
- Target: local BF16 Qwen3.8-27B, TP=2
- Draft: `amd/Qwen3.8-27B-EAGLE3-Full-Vocab`, BF16, one EAGLE layer, hidden 5120, intermediate 17408, full vocab 248320, auxiliary target layers 31/60
- Runtime: vLLM 0.29.0, PyTorch 2.12 development build, ROCm 7.2
- Common server settings: max model length 32768, max sequences 64, max batched tokens 16384, prefix cache off
- Workloads: exact random ISL/OSL, temperature 0, ignore EOS; batch is maximum request concurrency
- Existing baseline uses async scheduling; EAGLE does not. A no-async baseline is being added for strict A/B and will replace the primary delta table.

## Static N=2 full matrix

The delta below is versus the existing production-style async baseline, so it should be treated as an operational comparison until the no-async control completes.

| Workload | Batch | Baseline tok/s | EAGLE tok/s | Δ throughput | Δ mean E2E | Acceptance | Accepted length |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128/128 | 1 | 20.68 | 32.63 | +57.8% | −36.6% | 55.9% | 2.12 |
| 128/128 | 4 | 70.62 | 71.79 | +1.7% | −8.6% | 53.7% | 2.07 |
| 128/128 | 16 | 158.99 | 197.91 | +24.5% | −26.5% | 48.3% | 1.97 |
| 2048/256 | 1 | 12.59 | 23.07 | +83.2% | −45.4% | 53.9% | 2.08 |
| 2048/256 | 4 | 40.34 | 46.24 | +14.6% | −15.3% | 50.3% | 2.01 |
| 2048/256 | 16 | 80.65 | 67.93 | −15.8% | +14.9% | 51.2% | 2.02 |
| 16384/256 | 1 | 3.43 | 7.83 | +128.2% | −56.2% | 52.8% | 2.06 |
| 16384/256 | 4 | 8.73 | 6.93 | −20.6% | +21.1% | 60.8% | 2.22 |
| 16384/256 | 16 | 10.70 | 4.46 | −58.3% | +114.8% | 52.3% | 2.05 |
| 512/512 | 1 | 18.07 | 27.86 | +54.2% | −35.1% | 45.0% | 1.90 |
| 512/512 | 4 | 63.36 | 71.10 | +12.2% | −23.0% | 62.2% | 2.24 |
| 512/512 | 16 | 145.29 | 158.59 | +9.2% | −36.0% | 58.1% | 2.16 |

All 12 EAGLE cells completed without request failures. The high-batch long-context regressions are severe despite healthy acceptance, proving that acceptance rate alone is not a sufficient deployment metric.

## N and draft-TP sweep

| Config | 128/128 c1 tok/s | 128/128 c16 tok/s | 512/512 c1 tok/s | Main conclusion |
|---|---:|---:|---:|---|
| N=2, draft TP1 | 32.24 | 175.86 | 27.86 | Best robust/default choice |
| N=3, draft TP1 | 33.90 | 164.97 | 29.90 | Best low-concurrency static N |
| N=3, draft TP2 | 35.11 | 159.83 | 29.51 | Short c1 niche only |
| N=5, draft TP1 | 25.69 | 121.50 | 23.10 | Reject |
| N=7, draft TP1 | 26.30 | 100.03 | 22.66 | Reject |

N=7 reaches accepted length 3.02 for short c1 but is still 22% slower than N=3. More accepted tokens do not compensate for seven sequential draft proposals and the larger verification shape on W7900.

## Memory, graph, and backend findings

- EAGLE target+draft model load is about 26.93 GiB/rank versus 25.24 GiB/rank target-only.
- Static speculative graph capture uses about 7.3 GiB/rank.
- N=3 leaves about 5.17 GiB/rank KV cache, approximately 124,769 tokens or 3.81× concurrency at 32K.
- N=7 leaves about 5.07 GiB/rank and approximately 98,334 KV tokens (3.00× at 32K) because larger verification graphs increase memory pressure.
- Full attention selects ROCM_ATTN. GDN decode still uses a Triton fallback because the fused MTP path is not compiled for gfx1100.
- TP communication uses PYNCCL; custom/AITER all-reduce is unavailable.
- GPU pairs 0/1, 2/3, and 4/5 initialize TP2. Pair 6/7 fails PYNCCL initialization with `HIP invalid argument`, a machine/topology issue that must not be confused with an N=7 model failure.

## Dynamic N failure

The tested build supports `num_speculative_tokens_per_batch_size`. A schedule of `[[1,4,3],[5,64,2]]` is accepted and forces piecewise graph mode, but engine startup fails in `SpeculatorCudaGraphManager._init_candidates()`:

```text
rounded_num_tokens = round_up(num_tokens, decode_query_len)
ZeroDivisionError: integer division or modulo by zero
```

The dense schedule itself contains only positive N values. The bug is in EAGLE query-length derivation for mixed speculative lengths. Until fixed, use separate replicas or request routing instead of engine-level adaptive N.

## Optimization priorities

### P0

1. Fix EAGLE dynamic-N graph candidate generation and add ROCm startup coverage.
2. Optimize small-M BF16 GEMM/GEMV for draft N and target verification shapes; add N/batch-aware dispatch.
3. Fuse residual+RMSNorm, RMSNorm+projection, gate+SiLU+mul, and EAGLE auxiliary-state preparation.
4. Route or disable speculation for 16K traffic at concurrency ≥4.

### P1

1. Reduce HIP graph capture sizes and graph-pool memory while retaining deployed batch/N shapes.
2. Port/enable the fused GDN decode path for gfx1100.
3. Reduce TP2 PYNCCL cost using fused GEMM+reduce or overlap; separately diagnose the GPU6/7 pair.
4. Tune long-context chunked prefill and attention backend with p99 and decode-starvation gates.

### P2

1. Evaluate FP8 KV cache after graph-memory optimization, with an accuracy gate.
2. Consider a dedicated N=3 draft-TP2 low-latency replica only if repeated production-shaped tests preserve its narrow c1 win.

## Reproduction and artifacts

- Remote result root: `/nvme0/data/zhenchen/spec-bench-results/qwen3.8-27b-eagle3-w7900/`
- Full EAGLE matrix: `eagle_n2_dtp1/matrix/`
- Parameter sweeps: `tuning/`
- Server configs/logs: `servers/`
- Target profile: `profiles/baseline/`
- Compact summaries: `matrix_summary.csv`, `tuning_summary.csv`

