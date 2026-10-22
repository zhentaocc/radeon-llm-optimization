# Fused GDN decode for gfx1100

Status: **micro-validated; end-to-end blocked by shared-host PyNccl initialization**. The patch is ready for review and same-GPU service validation, but is not yet a promotion or upstream PR candidate.

Qwen3.8-27B contains 48 GDN layers and 16 full-attention layers. Stock Radeon builds do not compile `torch.ops._C.fused_gdn_decode_post_conv_mtp`, so EAGLE verification uses a launch-heavy fallback for most target layers.

## Patch boundary

[`vllm-v0.29.0-gfx1100-hip.patch`](vllm-v0.29.0-gfx1100-hip.patch) (SHA-256 `17375e2def6a7adeb804aa1d6f5336acd3fcc9d4569f4be3bf00be1b3a141170`) adds a gfx1100-only HIP implementation and leaves CUDA and other ROCm architectures unchanged:

- replaces PTX `cp.async` with aligned synchronous 16-byte global-to-LDS copies and explicitly aligns every LDS array accessed through `float4`;
- uses HIP BF16 conversion and runtime APIs;
- keeps explicit width-32 shuffles, which is designed to make wave64 execute as two independent logical subgroups; runtime validation here used the default gfx1100 wave mode, so wave64 compatibility remains unproven;
- reuses the existing stable-ABI schema and binding, adds ROCm CMake registration, and selects the native path only on gfx1100;
- bounds every source/destination state slot against `state.size(0)`, rejects per-request token counts wider than `state_indices`, and retains fallback for unsupported GQA ratios or shapes above 16 requests / 48 actual verification tokens.

The shape gate is performance-based, not a correctness limitation. It prevents the native path from being selected where the measured HIP kernel loses to the fallback.

## Build and correctness

The isolated vLLM 0.29.0 source built `_C_stable_libtorch` successfully with ROCm 7.2.4, Release mode, and `--offload-arch=gfx1100`. The generated Ninja graph contains the new HIP object and the full target completed successfully. The reviewed rebuild registered `_C.fused_gdn_decode_post_conv_mtp`; its SHA-256 is `12de9e2040be703a94b99f13dd8ce9fcf205820dede6c39c7455ec4cf54e4acd`.

The dependency-free harness [`standalone_validate.py`](standalone_validate.py) passed 480/480 regular cases across:

- MTP widths 1, 2, 3, and 5;
- GQA value/key-head ratios 1, 2, 3, 4, and 8;
- FP32 and BF16 recurrent state;
- FP32 and BF16 RMSNorm weight;
- FP32, BF16, and FP16 `dt_bias`;
- SiLU and sigmoid output gates.

Maximum regular-case output relative L2 error was `2.331e-4`; maximum state absolute error was `9.766e-4` in BF16 and approximately `1e-7` in FP32. Ragged request lengths `(4, 2, 0)` with accepted counts `1`, `2`, and full width `4` passed for all three `dt_bias` dtypes. Negative guards verified zero output and no state mutation when tokens exceed `state_indices` width or the selected source slot is out of range, and verified that an out-of-range destination is skipped without suppressing valid output or valid destination updates. The harness itself captures once and replays twice for FP32 and BF16 state; graph output and mutable state were bit-exact with eager execution. It writes the compact machine-readable [`standalone_validation.json`](standalone_validation.json) directly via `--output` (default: beside the script), so the graph result is not assembled by a separate manual step.

The production image lacks the optional pytest dependency `tblib`, so the repository test suite stops during `conftest.py` import. The same test matrix was therefore run with a dependency-free standalone harness against the rebuilt full extension. This is an image packaging issue, not a kernel failure.

## Microbenchmark and launch profile

Measurements use the real Qwen3.8 TP2-local head shape (`H=8`, `HV=24`, `K=V=128`), BF16 inputs, and FP32 state. Each latency is the median of 15 blocks with 20 HIP-event iterations per block.

| Shape | Native HIP | Stock fallback | Speedup |
|---|---:|---:|---:|
| batch 1, N=1 | 15.044 us | 85.327 us | 5.672x |
| batch 1, N=3 | 20.342 us | 117.607 us | 5.781x |
| batch 4, N=3 | 26.390 us | 117.230 us | 4.442x |
| batch 16, N=3 | 90.182 us | 118.003 us | 1.308x |
| batch 32, N=3 | 425.010 us | 398.077 us | 0.937x |
| batch 16, N=5 | 253.980 us | 225.761 us | 0.889x |

For the production N=2 configuration, `torch.profiler` records one native operator/kernel launch at both c1 and c16. The fallback records five state-copy launches, one recurrent-update kernel, and one gated RMSNorm kernel: seven launches total. Detailed rows are in [`microbench.csv`](microbench.csv), [`profiler_launches.csv`](profiler_launches.csv), and [`validation.csv`](validation.csv).

## End-to-end gate

No service result is claimed. Exact stock configuration was used: target TP2, draft TP1, EAGLE N=2, graphs enabled, `gpu_memory_utilization=0.88`, skinny GEMM enabled, and PyNccl enabled. Only the rebuilt stable extension and Python selector were mounted.

After clean double preflights, both physical GPU2/3 and the previously working GPU0/1 pair failed before model loading and before any patched forward call. In both cases `PyNcclCommunicator.ncclCommInitRank` reported NCCL 2.27.7 `unhandled cuda error` followed by HIP `invalid argument`. No external GPU workload appeared during either attempt, and postflight returned to 0% use / 0% VRAM. Per the retry gate, no further service attempts were made.

This makes the current result **E2E infra-blocked**, not a kernel regression. Re-run candidate then stock serially on one healthy TP2 pair before promotion, using 128/128 c1 and c16 plus 512/512 c1 with at least three repeats. Require stable throughput/TPOT improvement and no TTFT, acceptance, memory, or graph regression.

The model also has an ROCm AITER fast path for eligible decode layouts. A future E2E run must use profiler evidence or an explicit operator counter to prove that `fused_gdn_decode_post_conv_mtp` is actually selected; server startup alone is not evidence that this native kernel ran.

## Reproduction notes

- The patch passes `git apply --check` against the stock vLLM 0.29.0 source snapshot.
- Build only for gfx1100 and verify the generated compile command contains `--offload-arch=gfx1100`.
- Validate operator registration, the 480-case reference matrix, ragged/negative guards, and HIP Graph replay before starting a service.
- Do not claim wave64 support until an explicit `-mwavefrontsize64` build and runtime matrix passes.
- Keep the 16-request / 48-token selector gate until a broader kernel mapping improves the two rejected large-shape cases.
