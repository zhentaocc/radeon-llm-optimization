# Qwen3.8-27B + EAGLE3 on Radeon W7900D

Updated: 2026-10-22  
Status: EAGLE3 functional/performance matrix, parameter sweep, and strict identical-scheduler baseline complete; optimization experiments are in progress.

## Executive summary

- The AMD full-vocabulary EAGLE3 checkpoint loads and serves correctly on W7900D with vLLM 0.29.0, ROCm 7.2, target TP=2, and draft TP=1/2.
- Static N=2, draft TP=1 is the best general setting. Against the strict no-async target-only baseline it improves output throughput by 57% for 128/128 c1, 77% for 2048/256 c1, and 55% for 512/512 c1.
- The optimization is workload-sensitive. For 16K/256 it is +128% at c1 but −21% at c4 and −58% at c16; speculation must be disabled or routed away for high-concurrency long-prefill traffic.
- N=3 is best for low concurrency; N=2 is best for high concurrency. N=5/7 consistently lose because additional draft/verification cost grows faster than accepted-token length.
- Draft TP2 gives a narrow +3.6% win for short c1, but loses 1–3% on long output/high batch. Draft TP1 remains the default.
- Static EAGLE3 already uses HIP graphs. V1 raises KV capacity to 272K tokens but collapses on 16K/c16; v2 retains 210,751 tokens but loses 4.8% at c1 and 6.7% on long decode. Neither fixed capture set is promotable.
- A shape-aware small-M dispatch passes isolated microbenchmarks and 13 unit tests but fails strict same-GPU A/B (-20.9% c1, -13.3% long decode). It is retained as a negative result, not an upstream candidate.
- Dynamic per-batch N now starts after a one-line graph-manager guard and a regression test, but it is not performance-ready: piecewise graph mode loses 15.5% at c16 versus static N=2.

## Environment and methodology

- GPU: 2× Radeon Pro W7900D 48 GiB per service, `gfx1100`
- Target: local BF16 Qwen3.8-27B, TP=2
- Draft: `amd/Qwen3.8-27B-EAGLE3-Full-Vocab`, BF16, one EAGLE layer, hidden 5120, intermediate 17408, full vocab 248320, auxiliary target layers 31/60
- Runtime: vLLM 0.29.0, PyTorch 2.12 development build, ROCm 7.2
- Common server settings: max model length 32768, max sequences 64, max batched tokens 16384, prefix cache off
- Workloads: exact random ISL/OSL, temperature 0, ignore EOS; batch is maximum request concurrency
- The primary baseline uses the same no-async scheduler setting as EAGLE. Its 12 cells are within normal run variance of the earlier async baseline, so scheduler choice does not explain the EAGLE regressions.

## Static N=2 full matrix

The delta below is versus the completed target-only no-async baseline.

| Workload | Batch | Baseline tok/s | EAGLE tok/s | Δ throughput | Δ mean E2E | Acceptance | Accepted length |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128/128 | 1 | 20.78 | 32.63 | +57.0% | −36.3% | 55.9% | 2.12 |
| 128/128 | 4 | 70.58 | 71.79 | +1.7% | −8.7% | 53.7% | 2.07 |
| 128/128 | 16 | 160.87 | 197.91 | +23.0% | −25.6% | 48.3% | 1.97 |
| 2048/256 | 1 | 13.07 | 23.07 | +76.5% | −43.3% | 53.9% | 2.08 |
| 2048/256 | 4 | 41.23 | 46.24 | +12.1% | −13.4% | 50.3% | 2.01 |
| 2048/256 | 16 | 81.09 | 67.93 | −16.2% | +15.5% | 51.2% | 2.02 |
| 16384/256 | 1 | 3.46 | 7.83 | +126.5% | −55.8% | 52.8% | 2.06 |
| 16384/256 | 4 | 8.74 | 6.93 | −20.6% | +21.1% | 60.8% | 2.22 |
| 16384/256 | 16 | 10.75 | 4.46 | −58.5% | +115.7% | 52.3% | 2.05 |
| 512/512 | 1 | 18.01 | 27.86 | +54.7% | −35.4% | 45.0% | 1.90 |
| 512/512 | 4 | 62.96 | 71.10 | +12.9% | −23.5% | 62.2% | 2.24 |
| 512/512 | 16 | 145.06 | 158.59 | +9.3% | −36.1% | 58.1% | 2.16 |

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

### HIP graph capture-set optimization

For the deployed maximum concurrency 16 and static N=2, the capture set `[1,2,4,8,16,24,32,40,48,56,64]` covers the active decode/verification shapes while allowing larger prefill shapes to use piecewise execution. It reduces graph capture time to 13–19 seconds and memory to 1.85–1.87 GiB/rank. Available KV memory becomes 10.67 GiB and 272,391 tokens, equivalent to 8.31× theoretical 32K concurrency.

The prototype reaches 31.67/30.98/31.42 tok/s across three 128/128 c1 runs, but a same-period default repeat is 32.88 tok/s, so the median loss is 4.4%. The cause is deterministic: static N=2 verifies three tokens for c1 and shape `3` was omitted, forcing piecewise execution. C16 is within 1.9% of its same-period control and 512/512 c1 is neutral.

The v1 16K/c16 gate was stopped after 16/32 requests and 2,413 seconds. Its 4,096 generated tokens imply about 1.70 tok/s, 61.9% below the completed default result, so the configuration is rejected despite the memory gain.

V2 covers every active batch size 1–16 and every N=2 verification multiple through 48 (`1..16,18,21,...,48,56,64`). It starts after stale TP contexts are cleared, uses 4.26 GiB graph memory/rank, and leaves 210,751 KV tokens (6.43x theoretical 32K concurrency), versus 6.64 GiB and 150,241 tokens (4.59x) for the same N=2 default control. Throughput is 31.33/173.11/26.23 tok/s for short c1, short c16, and 512/512 c1, versus 32.92/173.60/28.12 for the same-period default. V2 passes c16 but fails the 3% gate at c1 (-4.8%) and long decode (-6.7%). The next version should use scheduler-observed shape residency instead of a static arithmetic list.

### Shape-aware small-M GEMM dispatch

The target profile's 2,314 `wvSplitK_hf_sml` launches account for about 54% of classified rank-0 kernel time, but direct W7900 microbenchmarks show that the current `n<=5` rule is too broad. Large MLP/LM-head shapes benefit strongly from `wvSplitK` (1.24–3.0×), while target/draft QKV, O projection, and target MLP-down are 7–23% faster through torch/hipBLASLt. The crossover for K=5120 is between output dimensions 10,240 and 11,264.

A v0.29.0 patch uses `wvSplitK` on RDNA only when `max(m,k)>=11264`, restricts LLMM1 to single-token K≤4096, and leaves CDNA unchanged. The production ROCm image passes all 13 affected unit tests, but the strict same-GPU service A/B rejects it:

| Workload | Patched tok/s | Stock tok/s | Delta |
|---|---:|---:|---:|
| 128/128 c1 | 26.04 | 32.92 | -20.9% |
| 128/128 c16 | 169.43 | 173.60 | -2.4% |
| 512/512 c1 | 24.39 | 28.12 | -13.3% |

The isolated eager microbenchmark is not a sufficient dispatch oracle for compiled graph replay. Retain the branch as a negative result. Re-profile the production graph path and implement a backend-specific in-graph candidate before revisiting this optimization.

Exact QKV isolation confirms where it fails. Routing target+draft QKV to torch gives 27.62 tok/s (-16.1%); draft QKV alone gives 28.18 (-14.4%); target QKV alone has a three-run median of 32.57 (-1.1%, with runs 32.57/33.01/30.74). The draft projection accounts for most of the loss, while target-only has no reproducible win. Existing-backend/threshold tuning is therefore exhausted; the actionable next step is a graph-friendly RDNA draft-QKV kernel or fusion with adjacent norm/rope/layout work.

### Fusion audit

Two originally proposed fusion tasks are already present. Qwen3.8 calls `Qwen3NextRMSNorm(hidden_states, residual)` and the runtime selects the native fused-add RMSNorm path; its dense MLP uses a merged gate/up projection followed by `SiluAndMul()`. Full attention also has a fused gated split + QK-RMSNorm + partial RoPE path. The remaining fusion work should therefore focus on RMSNorm+QKV/GDN projection, post-attention RMSNorm+gate/up projection, EAGLE auxiliary hidden-state preparation, and the 48-layer GDN state/convolution/norm path.

The GDN gap is not a flag: `fused_gdn_decode_kernel.cu` is registered only in the CUDA/CUDA-13 build path, and the ROCm wheel does not expose `fused_gdn_decode_post_conv_mtp`. Enabling it on W7900 requires a HIP port with wave-safe reductions, ROCm build registration, and recurrent-state/MTP parity coverage.

## Dynamic N: correctness fixed, performance rejected

The tested build supports `num_speculative_tokens_per_batch_size`. A schedule of `[[1,4,3],[5,64,2]]` originally forced piecewise graph mode and then failed in `SpeculatorCudaGraphManager._init_candidates()`:

```text
rounded_num_tokens = round_up(num_tokens, decode_query_len)
ZeroDivisionError: integer division or modulo by zero
```

The draft graph manager has `decode_query_len=1`; target verification expansion incorrectly subtracted the maximum N from that value and produced zero. The fix applies that expansion only when `decode_query_len > num_speculative_tokens`. Its unit regression test passes in the v0.29.0 ROCm image, and a real patched EAGLE server starts successfully.

Two schedules were measured after the fix. `[[1,4,3],[5,64,2]]` reaches 36.05/153.64/30.20 tok/s for short c1, short c16, and 512/512 c1. Tightening it to `[[1,1,3],[2,64,2]]` reaches 35.33/148.67/29.55 tok/s. The latter is +9.6% versus static N=2 at short c1, but −15.5% at short c16 and −1.2% versus static N=3 on 512/512 c1. Acceptance is healthy; the regression comes from the dynamic feature forcing piecewise graph execution. Keep the correctness patch, but use separate static-N replicas or request routing until dynamic full-graph performance is recovered.

## Optimization priorities

### P0

1. Restore full-graph-equivalent performance for dynamic N; correctness and startup coverage are complete.
2. Re-profile small-M BF16 GEMM/GEMV inside production graph replay; do not use the rejected eager-microbenchmark threshold policy.
3. Fuse RMSNorm+projection and EAGLE auxiliary-state preparation; residual+RMSNorm and gate+SiLU+mul are already fused at the model/runtime boundary.
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

