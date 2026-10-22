# EAGLE target RMSNorm + QKV projection

Branch: `opt/eagle-rmsnorm-qkv`

Status: **rejected / do not promote**. The fused operator is numerically correct,
HIP-Graph safe, and buildable for gfx1100, but its best median component speedup
is 4.02%, below the pre-declared 8% service-A/B admission gate.

## Single optimization boundary

Fuse the input RMSNorm output directly into the TP-local QKV projection for the
16 full-attention layers of the Qwen3.8-27B target verifier. The optimization
must preserve the existing residual semantics and leave linear-attention/GDN,
MLP, EAGLE auxiliary-state preparation, and draft layer-0 preparation
unchanged.

This does not duplicate the published EAGLE auxiliary or layer-0 branches.
Those fuse draft hidden-state normalization and concatenation before the draft
layer. This branch targets the verifier's
`Qwen3NextDecoderLayer.input_layernorm -> Qwen3NextAttention.qkv_proj`
boundary.

## Baseline evidence before implementation

- The existing target profile contains roughly 1,610 RMSNorm reductions over
  ten profiled steps. QKV is also a repeated small-M service hot path.
- Qwen3.8-27B has 64 layers: 16 full-attention layers and 48 GDN layers. The
  proposed boundary executes once in every full-attention layer during target
  verification.
- The target hidden width is 5,120. Attention uses 24 query heads, four KV
  heads, head dimension 256, and output gating. At TP2 the local packed
  projection is Q+gate 6,144 plus K 512 plus V 512, for local output width
  7,168.
- Prior same-service isolation found that switching target QKV to the generic
  torch/hipBLASLt path produced a 32.57 tok/s median versus 32.92 tok/s stock
  (-1.1%). Existing-backend dispatch tuning has therefore not shown a win.
- Source audit confirms the model materializes normalized BF16 activations
  before `QKVParallelLinear`. vLLM/AITER provides fused residual-add RMSNorm
  and fused RMSNorm+FP8 quantization, but no equivalent BF16
  RMSNorm+linear/GEMM path.
- `gate + SiLU + mul` is already represented by the merged gate/up projection
  followed by `SiluAndMul`; residual-add RMSNorm is already fused. Those are
  not candidates in this branch.

## Feasibility gate

The mathematical fusion is valid because the normalized activation has a
single consumer at this boundary:

```
y[m, n] = sum_k (x[m, k] * rsqrt(mean_k(x[m, k]^2) + eps)
                 * rms_weight[k]) * qkv_weight[n, k]
```

The kernel must preserve FP32 RMS statistics and accumulation behavior close
enough to the stock BF16 RMSNorm followed by the production GEMM. Merely moving
the full KxN matmul into a naive Triton program is not acceptable. Implementation
proceeds only if it can reuse a high-performance GEMM prologue/epilogue or
demonstrate a reliable persistent small-M layout competitive with the stock
path. Otherwise this branch will record the feasibility blocker and retain only
the design and tests.

## Implementation and completed gates

The patch extends the production ROCm `wvSplitK` activation-load prologue; it
does not replace the KxN matrix multiply with a generic Triton implementation.
For the real TP2 shape (`M=1..5`, `K=5120`, `N=7168`) it:

1. adds hidden and residual in FP32 and casts back once, matching stock fused
   RMSNorm rounding;
2. writes the updated residual;
3. computes FP32 RMS statistics and applies the RMS weight while loading LDS;
4. reuses the existing persistent `wvSplitK` dot-product body.

The implementation is default-off through
`VLLM_ROCM_USE_FUSED_RMSNORM_QKV=1`. Eligibility is restricted to gfx11/gfx12,
unquantized contiguous FP16/BF16 tensors, no bias, `M=1..5`, `K % 8 == 0`,
and `N > 8`. Unsupported layouts and the first layer without a residual fall
back to stock behavior.

The true vLLM CMake/Ninja ROCm build completed 55/55 targets for gfx1100,
including `skinny_gemms.hip.o` and `_rocm_C.abi3.so`. `compileall` was not used
as a substitute for this gate.

CPU/reference gates add 26 mathematical cases, 18 eligibility/fallback/fake
cases (including M=6 rejection, bias/non-contiguous rejection, and the real
7168x5120 TP-local layout), plus one fullgraph fake-tensor case: 45/45 pass.

On GPU2, 26 real-op correctness cases cover FP16/BF16, M=1..5, epsilon 1e-6
and 1e-5, plus zero, constant, and high-dynamic-range inputs. Updated residuals
are bitwise identical. Maximum QKV absolute error is 0.001953125 for FP16 and
0.015625 for BF16; most cases are bitwise identical. HIP Graph capture plus two
replays passes for both dtypes with stable output addresses.
`torch.compile(fullgraph=True)` passes for both dtypes.

## Performance result

The exact baseline segment is stock `fused_add_rms_norm` followed by stock
`wvSplitK`; the candidate is one `wvSplitKRMSNorm` call. Each timing cell uses
12 warmups and 15 samples on an otherwise-idle physical GPU2. Preflight and
postflight both reported 0% use and 0 MiB allocated. No external contention was
observed.

| M | FP16 baseline/candidate (ms) | FP16 speedup | BF16 baseline/candidate (ms) | BF16 speedup | >=8% gate |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0.10884 / 0.10496 | 1.037x | 0.10968 / 0.10544 | 1.040x | No |
| 2 | 0.11132 / 0.10880 | 1.023x | 0.11236 / 0.10956 | 1.026x | No |
| 3 | 0.11164 / 0.10960 | 1.019x | 0.11320 / 0.11128 | 1.017x | No |
| 4 | 0.11568 / 0.11444 | 1.011x | 0.11768 / 0.11592 | 1.015x | No |
| 5 | 0.11304 / 0.11224 | 1.007x | 0.11640 / 0.11600 | 1.003x | No |

## Decision

Do not run service A/B and do not merge the source patch. The saved launch and
intermediate-tensor traffic is too small relative to QKV GEMM time: the benefit
falls from about 4% at M=1 to effectively zero at M=5. A full service run would
not recover the required 8% component gain and would consume scarce same-pair
GPU time without meeting the admission criterion.

Keep this branch as a reproducible negative result and design reference. The
next optimization should move to a larger target boundary such as the gfx1100
GDN decode path, attention, or MLP rather than eliminating another small norm
launch.

## Reproduction artifacts

- `vllm-v0.29.0.patch`: seven-file, opt-in source patch against the W7900
  vLLM 0.29.0 source snapshot.
- `scripts/build_static.sh`: real ROCm CMake/Ninja build gate for gfx1100.
- `tests/test_reference.py` and `scripts/run_reference.sh`: reference,
  fallback, fake/meta, and fullgraph tests.
- `tests/test_gpu.py` and `scripts/run_gpu.sh`: real-op correctness, HIP Graph,
  `torch.compile`, and microbenchmark test.
- `results/reference_results.json` and `results/gpu_results.json`: machine-
  readable results. These contain metrics only, with no prompts, generations,
  credentials, or model data.
- `checksums.sha256`: SHA256 manifest for every reproducibility artifact.

Environment:

- image: `specbench:vllm-0.29.0-rocm723-gfx1100`
- image ID: `sha256:8bb2ff76f14bb1092ca566724b2dcd8c483f82703fa6b373983ae4a3c44ad0cd`
- physical test GPU: W7900 GPU2, gfx1100, 48 CUs
- remote private build tree: `/dev/shm/zhenchen-rmsnorm-qkv-vllm`
- remote metrics: `/dev/shm/zhenchen-rmsnorm-qkv-vllm/gpu_results.json`
- local staging source: `C:\\Users\\zhenchen\\rmsnorm-qkv-dev`

Apply the patch to a clean vLLM 0.29.0 tree, place the selected test beside the
source as `/workspace/test_reference.py` or `/workspace/test_gpu.py`, and run
the corresponding script in the image with the source mounted at `/workspace`.
For the GPU gate, explicitly set `HIP_VISIBLE_DEVICES` to an approved clean
physical GPU. Never infer a free GPU from the device display name alone; verify
gfx1100, memory capacity, and host process ownership first.
