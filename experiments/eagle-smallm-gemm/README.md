# Shape-aware RDNA small-M GEMM dispatch

Status: patch and microbenchmark validated; end-to-end service gate pending host PYNCCL recovery.

## Finding

vLLM 0.29.0 sends every contiguous BF16/FP16 GEMM with token count `1..5` and output dimension `>8` to `wvSplitK`. On W7900 (`gfx1100`), that blanket policy is suboptimal for Qwen3.8-27B target and EAGLE3 projection shapes.

Representative median kernel latency:

| TP-local weight shape | Role | `wvSplitK` | torch/hipBLASLt | Best path |
|---|---|---:|---:|---|
| 4096×5120 | target QKV | 64–70 µs | 57 µs | torch, 1.13–1.23× |
| 5120×3072 | target O | 53–65 µs | 56–58 µs | LLMM1 at M=1; torch at M≥2 |
| 17408×5120 | target MLP up | 229–239 µs | 399–407 µs | `wvSplitK`, 1.70–1.76× |
| 5120×8704 | target MLP down | 121–131 µs | 112–114 µs | torch, 1.08–1.16× |
| 8192×5120 | draft QKV | 117–126 µs | 109 µs | torch, 1.07–1.15× |
| 5120×6144 | draft O | 95–110 µs | 86–94 µs | torch, 1.01–1.23× |
| 34816×5120 | draft MLP up | 439–454 µs | 1314–1347 µs | `wvSplitK`, ~3.0× |
| 5120×17408 | draft MLP down | 229–245 µs | 304–306 µs | `wvSplitK`, 1.24–1.34× |
| 124160×5120 | TP-local LM head | 1519–1541 µs | 2671–2971 µs | `wvSplitK`, 1.76–1.94× |

The output-dimension sweep places the W7900 crossover between 10,240 and 11,264 elements for `K=5120`. The patch therefore retains `wvSplitK` on RDNA when `max(M,K) >= 11264`, uses LLMM1 only for single-token `K<=4096`, and otherwise falls through to torch/hipBLASLt. CDNA behavior is unchanged.

## Validation

- Microbenchmarks: 60 timed iterations for production shapes plus threshold sweep.
- Numerical check: maximum BF16 absolute difference is consistent with existing kernel accumulation behavior.
- Production vLLM 0.29.0 ROCm image: `13 passed` in `test_rocm_unquantized_gemm.py`.
- Source commit: `6562a7e` in the local vLLM branch `opt/eagle-smallm-gemm`.
- Patch: `vllm-v0.29.0.patch`.

The full EAGLE server A/B is blocked temporarily because new TP2 communicators on GPU0/1 and GPU2/3 fail before model load with PYNCCL `HIP invalid argument`; those pairs completed earlier matrices successfully. Do not attribute that host-state failure to this patch.
