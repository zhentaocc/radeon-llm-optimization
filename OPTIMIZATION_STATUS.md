# Optimization status

Updated: 2026-10-22

## Summary table

| Priority | Optimization | Branch | Current evidence | Status | Exit criterion |
|---|---|---|---|---|---|
| P0 | Strict no-async baseline | `exp/baseline-noasync` | 12/12 cells complete; results closely match the earlier async baseline and preserve every major EAGLE conclusion | Complete | 12/12 clean cells complete |
| P0 | Draft/target small-M GEMM dispatch | `opt/eagle-smallm-gemm` | Shape sweep finds blanket `wvSplitK` loses 7–23% on QKV/O/target-down but wins 1.7–3.0× on large MLP; shape-aware patch passes 13 tests | Patch ready; E2E gate blocked by host PYNCCL state | ≥10% decode throughput gain, no matrix regression |
| P0 | EAGLE fused RMSNorm/projection/MLP | `opt/eagle-fusion` | Thousands of norm, copy and elementwise launches; fusion flags are disabled in current graph | Planned | ≥8% batch-1 TPOT improvement |
| P0 | Dynamic N for EAGLE3 | `fix/eagle-dynamic-n` | Startup crash fixed; batch-1-only N=3 gives 35.33 tok/s, but c16 falls to 148.67 tok/s because dynamic mode forces piecewise graphs | Correctness fixed; perf rejected | Recover static N=2 c16 throughput within 3% before enabling |
| P0 | Long-context admission/routing | `opt/longctx-routing` | N=2 changes 16K throughput by +128%/−21%/−58% at batch 1/4/16 | Config-ready | Disable speculation or route separately at batch ≥4 |
| P1 | Draft TP policy | `opt/draft-tp-policy` | TP2 is +3.6% for short c1, −3.1% for short c16, −1.3% for 512/512 c1 | Analyzed | Policy encoded and regression tested |
| P1 | Reduce HIP graph memory/capture set | `opt/eagle-hip-graph` | V1 cuts graph memory 7.3→1.86 GiB/rank and raises KV ~125K→272K, but misses verification shape 3 and loses 4.4% c1 versus same-period control; v2 covers all batch/verify shapes | V2 pending PYNCCL recovery; 16K gate running | Recover KV capacity without >3% decode loss |
| P1 | TP communication / all-reduce | `opt/tp-communication` | Rank1 NCCL wait dominates eager profile; only PYNCCL is active | Planned | ≥10% TP2 decode gain or prove topology limit |
| P1 | GDN fused decode for gfx1100 | `opt/gdn-fused-decode` | Fused GDN MTP op is unavailable; Triton fallback is active | Planned | Fused path selected and ≥5% E2E gain |
| P1 | Long-context attention/prefill | `opt/longctx-attention` | 16K batch16 baseline p99 E2E is ~604 s; EAGLE worsens residency pressure | Planned | ≥15% 16K throughput gain with p99 non-regression |
| P2 | KV-cache capacity/precision | `opt/kv-cache` | EAGLE model+graphs reduce available KV cache; long-context concurrency collapses | Planned | More resident 16K requests with accuracy gate |

## Detailed analysis

### `exp/baseline-noasync`: strict control

The original target-only baseline used async scheduling, while the tested speculative path did not. The strict 12-cell no-async control is now complete and closely matches the earlier baseline: for example 20.78/70.58/160.87 tok/s at 128/128 c1/c4/c16 and 3.46/8.74/10.75 tok/s at 16K/256. The corrected comparison preserves every major conclusion, including the −58.5% EAGLE regression at 16K c16. This branch is the reference for every optimization branch.

### `opt/eagle-smallm-gemm`: draft and verify GEMM

The workload is dominated by small-M linear algebra during autoregressive decode. The existing target profile records 2,314 `wvSplitK_hf_sml` launches and about 286 ms on rank0 across ten profiled steps, roughly 54% of classified rank-0 kernel time. EAGLE adds one 5,120-hidden, 17,408-intermediate draft layer and verifies multiple tokens per target step, creating a new family of M=1/2/3/5/7 shapes.

Direct W7900 microbenchmarks show that the current blanket `n<=5` dispatch is wrong. `wvSplitK` is about 1.7× faster for target MLP-up, 3.0× for draft MLP-up, 1.24–1.34× for draft MLP-down, and 1.76–1.94× for the TP-local LM head. It is 7–23% slower for target/draft QKV and O projections and 8–16% slower for target MLP-down. A threshold sweep places the crossover between output dimensions 10,240 and 11,264 for K=5120.

The branch patch keeps CDNA behavior unchanged and, on RDNA, uses `wvSplitK` only when `max(m,k)>=11264`, limits LLMM1 to single-token K≤4096, and otherwise falls through to torch/hipBLASLt. The exact v0.29.0 test file passes 13/13 cases in the production ROCm image. End-to-end service validation awaits a host/GPU reset: PyNCCL fails during communicator initialization, while the torch.distributed fallback loads the model and then fails at its first all-reduce with the same `hipIpcGetMemHandle` error.

Remaining gate:

- launch patched and stock services on the same healthy GPU pair;
- repeat 128/128 c1/c16 and 512/512 c1 at least three times;
- profile kernel counts to confirm small projections moved off `wvSplitK`;
- gate on the full matrix before promotion.

### `opt/eagle-fusion`: norm and MLP fusion

The current compile configuration reports `fuse_norm_quant=false`, `fuse_act_quant=false`, `fuse_gemm_comms=false`, and no QK-norm/rope fusion. The profile also shows roughly 1,610 RMSNorm reductions per ten steps plus many BF16/FP32 copies and elementwise kernels. Highest-value candidates are `residual + RMSNorm`, `RMSNorm + QKV/GDN projection`, and `gate + SiLU + mul`; for EAGLE, fuse auxiliary hidden-state projection/concatenation where layout permits.

### `fix/eagle-dynamic-n`: scheduler/graph correctness

The build accepts `num_speculative_tokens_per_batch_size=[(start,end,N), ...]` and automatically downgrades full graphs to piecewise graphs. The original batch 1–4→N=3 and 5–64→N=2 schedule failed during `profile_cudagraph_memory()`:

`SpeculatorCudaGraphManager._init_candidates()` calls `round_up(num_tokens, decode_query_len)` with `decode_query_len=0`, causing `ZeroDivisionError`.

The failure was deterministic and came from applying target verification query-length expansion to the one-token draft model graph manager. The fix now guards that expansion with `decode_query_len > num_speculative_tokens`; a unit regression test passes in the v0.29.0 ROCm image, and the patched server starts successfully.

Performance still fails the promotion gate. A conservative batch 1→N=3, batch 2–64→N=2 schedule reaches 35.33 tok/s for 128/128 c1 (+9.6% versus static N=2), but only 148.67 tok/s for c16 (−15.5% versus the same tuning control) and 29.55 tok/s for 512/512 c1 (−1.2% versus static N=3). Acceptance behavior is normal; the remaining loss is the dynamic feature forcing piecewise graph execution. Keep the correctness patch, but do not recommend dynamic N on W7900 until the graph path is optimized.

### `opt/longctx-routing`: workload-aware speculation

Static N=2 is not safe as a universal switch. For 16K/256, throughput changes versus the existing baseline are +128% at batch1, −21% at batch4, and −58% at batch16. The immediate production action is application-level routing: enable EAGLE for low-concurrency/decode-heavy traffic and route high-concurrency long-prefill traffic to a target-only replica. Engine-level dynamic N remains blocked by the issue above.

### `opt/draft-tp-policy`: TP1 versus TP2

Draft TP2 changes N=3 throughput from 33.90 to 35.11 tok/s for 128/128 c1 (+3.6%), from 164.97 to 159.83 tok/s for c16 (−3.1%), and from 29.90 to 29.51 tok/s for 512/512 c1 (−1.3%). Default to draft TP1. TP2 is only a candidate for a dedicated short, batch-1 service and requires more repetitions before promotion.

### `opt/eagle-hip-graph`: graph coverage and memory

Static EAGLE3 already captures and replays HIP/CUDA graphs in this build; “add graph support” is therefore no longer the correct task. Restricting capture sizes to `[1,2,4,8,16,24,32,40,48,56,64]` cuts graph capture time to 13–19 s and graph memory from about 7.3 to 1.85–1.87 GiB/rank. Available KV memory rises to 10.67 GiB and 272,391 tokens, or 8.31× theoretical 32K concurrency—about 2.18× the original token capacity.

The first set misses shape `3`, which is the N=2 verification shape for c1. Its 128/128 c1 results are 31.67, 30.98, and 31.42 tok/s; a same-period default repeat reaches 32.88, so the median loss is 4.4% and fails the gate. C16 is 200.30 versus 204.23 tok/s (−1.9%), and 512/512 c1 is 27.95 tok/s (+0.3% versus the earlier tuning control).

V2 captures every batch size 1–16 plus each verification multiple through 48 (`1..16,18,21,...,48,56,64`). That should restore full-graph execution throughout the deployed range while still reducing the 54-shape default. New TP2 launches are blocked by a host HIP IPC failure under both PyNCCL and torch.distributed and require a host/GPU reset. The v1 16K/c16 residency test remains in progress on the already-running service.

### `opt/tp-communication`

The eager target profile shows similar GEMM time across ranks but far higher NCCL time on rank1. W7900 uses PYNCCL here; custom/AITER all-reduce is unavailable. Compare TP2 with a feasible TP1/PP2 control, then test fused GEMM+reduce and overlap. Treat GPU6/7 separately: that pair fails PYNCCL initialization with `HIP invalid argument`, while GPU0/1, 2/3, and 4/5 work.

### `opt/gdn-fused-decode`

Qwen3.8 contains 48 GDN and 16 full-attention layers. On gfx1100, `fused_gdn_decode_post_conv_mtp` is not compiled and the runtime falls back to Triton. Port/build the fused state-update, gated-delta, convolution and norm path for RDNA, then verify numerical parity before performance testing.

### `opt/longctx-attention`

Long-context behavior is dominated by prefill scheduling, attention/KV traffic, and decode starvation rather than the draft model alone. Scan `max_num_batched_tokens` at 4K/8K/16K/32K, compare ROCM_ATTN and TRITON_ATTN, and add decode-priority/chunk-size controls. Report TTFT and p99 alongside throughput.

### `opt/kv-cache`

Draft weights and graph captures reduce KV headroom. Evaluate a smaller capture set first, then FP8 KV cache with an accuracy gate. Capacity wins matter only if attention overhead and p99 stay controlled.

