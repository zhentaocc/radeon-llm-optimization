# Optimization status

Updated: 2026-10-22

## Summary table

| Priority | Optimization | Branch | Current evidence | Status | Exit criterion |
|---|---|---|---|---|---|
| P0 | Strict no-async baseline | `exp/baseline-noasync` | 12/12 cells complete; results closely match the earlier async baseline and preserve every major EAGLE conclusion | Complete | 12/12 clean cells complete |
| P0 | Draft/target small-M GEMM dispatch | `opt/eagle-smallm-gemm` | Broad patch is -20.9% c1; isolation finds draft-QKV alone -14.4% while target-QKV median is -1.1% | Existing-backend switching exhausted; negative result retained | Implement graph-friendly RDNA/fused draft kernel; candidate must beat stock c1 |
| P0 | Combined EAGLE preparation fusions | `opt/eagle-fusions-combined` | 14/14 component tests; HIP Graph and fullgraph compile pass; 7 launches -> 2 across component profiles; repeated fixed-256 c1 medians: throughput -0.07%, TPOT +0.13%, TTFT -1.05% versus stock | Micro-validated / E2E rejected | Do not upstream this boundary; target a larger fused region with measurable service impact |
| P0 | Dynamic N for EAGLE3 | `fix/eagle-dynamic-n` | Startup crash fixed; batch-1-only N=3 gives 35.33 tok/s, but c16 falls to 148.67 tok/s because dynamic mode forces piecewise graphs | Correctness fixed; perf rejected | Recover static N=2 c16 throughput within 3% before enabling |
| P0 | Long-context admission/routing | `opt/longctx-routing` | N=2 changes 16K throughput by +128%/−21%/−58% at batch 1/4/16 | Config-ready | Disable speculation or route separately at batch ≥4 |
| P1 | Draft TP policy | `opt/draft-tp-policy` | TP2 is +3.6% for short c1, −3.1% for short c16, −1.3% for 512/512 c1 | Analyzed | Policy encoded and regression tested |
| P1 | Reduce HIP graph memory/capture set | `opt/eagle-hip-graph` | V1 raises KV to 272K but 16K/c16 falls to ~1.70 tok/s; v2 leaves 210,751 tokens but loses 4.8% c1 and 6.7% long decode | V1/V2 rejected; redesign scoped | Recover KV capacity without >3% decode loss or long-context collapse |
| P1 | TP communication / all-reduce | `opt/tp-communication` | Rank1 NCCL wait dominates eager profile; only PYNCCL is active | Planned | ≥10% TP2 decode gain or prove topology limit |
| P1 | GDN fused decode for gfx1100 | `opt/gdn-fused-decode` | CUDA-only kernel/CMake path isolated; ROCm image lacks `fused_gdn_decode_post_conv_mtp`, so 48/64 layers use Triton | HIP port scoped | Fused path selected and ≥5% E2E gain |
| P1 | Long-context attention/prefill | `opt/longctx-attention` | 16K batch16 baseline p99 E2E is ~604 s; EAGLE worsens residency pressure | Planned | ≥15% 16K throughput gain with p99 non-regression |
| P2 | KV-cache capacity/precision | `opt/kv-cache` | EAGLE model+graphs reduce available KV cache; long-context concurrency collapses | Planned | More resident 16K requests with accuracy gate |

## Detailed analysis

### `exp/baseline-noasync`: strict control

The original target-only baseline used async scheduling, while the tested speculative path did not. The strict 12-cell no-async control is now complete and closely matches the earlier baseline: for example 20.78/70.58/160.87 tok/s at 128/128 c1/c4/c16 and 3.46/8.74/10.75 tok/s at 16K/256. The corrected comparison preserves every major conclusion, including the −58.5% EAGLE regression at 16K c16. This branch is the reference for every optimization branch.

### `opt/eagle-smallm-gemm`: draft and verify GEMM

The workload is dominated by small-M linear algebra during autoregressive decode. The existing target profile records 2,314 `wvSplitK_hf_sml` launches and about 286 ms on rank0 across ten profiled steps, roughly 54% of classified rank-0 kernel time. EAGLE adds one 5,120-hidden, 17,408-intermediate draft layer and verifies multiple tokens per target step, creating a new family of M=1/2/3/5/7 shapes.

Direct W7900 microbenchmarks show that the current blanket `n<=5` dispatch is wrong. `wvSplitK` is about 1.7× faster for target MLP-up, 3.0× for draft MLP-up, 1.24–1.34× for draft MLP-down, and 1.76–1.94× for the TP-local LM head. It is 7–23% slower for target/draft QKV and O projections and 8–16% slower for target MLP-down. A threshold sweep places the crossover between output dimensions 10,240 and 11,264 for K=5120.

The branch patch keeps CDNA behavior unchanged and, on RDNA, uses `wvSplitK` only when `max(m,k)>=11264`, limits LLMM1 to single-token K≤4096, and otherwise falls through to torch/hipBLASLt. The exact v0.29.0 test file passes 13/13 cases in the production ROCm image.

The strict end-to-end gate rejects it. Patched versus stock on the same recovered GPU0/1 pair is 26.04 versus 32.92 tok/s for 128/128 c1 (-20.9%), 169.43 versus 173.60 for c16 (-2.4%), and 24.39 versus 28.12 for 512/512 c1 (-13.3%). The isolated eager GEMM crossover does not carry into compiled graph replay, and the generic torch path is particularly costly at c1.

Exact-shape isolation closes the remaining threshold question. Target+draft QKV reaches 27.62 tok/s (-16.1%); draft QKV alone reaches 28.18 (-14.4%); target QKV alone produces 32.57/33.01/30.74 with a 32.57 median (-1.1%), so it has no reproducible gain. Draft QKV explains most of the regression. Keep the branch as a reproducible negative result; switching among existing backends is exhausted. The next attempt must implement a graph-friendly RDNA kernel or fused draft projection and beat stock c1 before any full-matrix run.

### `opt/eagle-fusion`: norm and MLP fusion

The profile shows roughly 1,610 RMSNorm reductions per ten steps plus many BF16/FP32 copies and elementwise kernels, but a source/runtime audit narrows the missing work. `Qwen3NextRMSNorm(hidden_states, residual)` already uses native fused-add RMSNorm; dense Qwen3.8 MLP already feeds a merged gate/up projection into `SiluAndMul`; and full attention already has a fused gated split + QK-RMSNorm + partial RoPE path. The disabled compile flags are quantization/fusion passes, not evidence that these model-level fusions are absent.

The actionable boundaries are RMSNorm+QKV/GDN projection, post-attention RMSNorm+gate/up projection, and EAGLE auxiliary hidden-state gather/projection/concatenation. For the 48 GDN layers, fused state update/convolution/norm is likely higher value than another generic MLP activation kernel.

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

The v1 16K/c16 residency test was stopped after 16/32 requests and 2,413 seconds: 4,096 output tokens imply about 1.70 tok/s, 61.9% below the completed default EAGLE result. More KV capacity did not prevent decode starvation, so v1 is rejected.

V2 captures every batch size 1–16 plus each verification multiple through 48 (`1..16,18,21,...,48,56,64`). After stale TP contexts were cleared, it started successfully with 4.26 GiB graph memory/rank, 210,751 KV tokens, and 6.43x theoretical 32K concurrency. Against a same-period default control it reaches 31.33 versus 32.92 tok/s at 128/128 c1 (-4.8%), 173.11 versus 173.60 at c16 (-0.3%), and 26.23 versus 28.12 at 512/512 c1 (-6.7%). V2 passes only the c16 gate. Redesign capture selection from scheduler telemetry and explicitly protect latency-critical c1/long-decode shapes; do not promote either fixed set.

### `opt/tp-communication`

The eager target profile shows similar GEMM time across ranks but far higher NCCL time on rank1. W7900 uses PYNCCL here; custom/AITER all-reduce is unavailable. Compare TP2 with a feasible TP1/PP2 control, then test fused GEMM+reduce and overlap. Treat GPU6/7 separately: that pair fails PYNCCL initialization with `HIP invalid argument`, while GPU0/1, 2/3, and 4/5 work.

### `opt/gdn-fused-decode`

Qwen3.8 contains 48 GDN and 16 full-attention layers. On gfx1100, `fused_gdn_decode_post_conv_mtp` is not compiled and the runtime falls back to Triton. The source exists at `csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu`, but CMake registers it only in the CUDA build for CUDA 13 architecture sets; the Python eligibility path also selects `gdn_decode_kernel == "cuda"`. This is a real HIP port: state update, convolution, gated RMSNorm, MTP state indices, wave-size-safe reductions, ROCm build registration, and BF16/FP32 parity tests are all required before performance testing.

### `opt/longctx-attention`

Long-context behavior is dominated by prefill scheduling, attention/KV traffic, and decode starvation rather than the draft model alone. Scan `max_num_batched_tokens` at 4K/8K/16K/32K, compare ROCM_ATTN and TRITON_ATTN, and add decode-priority/chunk-size controls. Report TTFT and p99 alongside throughput.

### `opt/kv-cache`

Draft weights and graph captures reduce KV headroom. Evaluate a smaller capture set first, then FP8 KV cache with an accuracy gate. Capacity wins matter only if attention overhead and p99 stay controlled.

