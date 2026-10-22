# Shape-aware RDNA small-M GEMM dispatch

Status: microbenchmark and unit tests pass, but strict same-GPU end-to-end A/B rejects the dispatch policy.

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

After stale TP contexts were cleared, patched and stock services were run serially on the same GPU0/1 pair with identical N=2, draft-TP1, graph, scheduler, and memory settings.

| Workload | Patched tok/s | Stock tok/s | Delta |
|---|---:|---:|---:|
| 128/128 c1 | 26.04 | 32.92 | -20.9% |
| 128/128 c16 | 169.43 | 173.60 | -2.4% |
| 512/512 c1 | 24.39 | 28.12 | -13.3% |

The patch is a useful negative result, not a PR candidate. Isolated eager GEMM latency does not predict compiled graph-replay performance, and replacing several small projections with the generic torch path is especially costly at c1. Keep the branch and raw data for reproducibility, but rework the optimization around profiler-observed production shapes and an in-graph backend/kernel implementation. Any next candidate must first beat stock on c1 before running the full matrix.

## QKV isolation

The c1 regression was isolated with exact `(output_dim, input_dim)` allowlists while every other shape stayed on stock `wvSplitK`:

| Candidate | 128/128 c1 tok/s | Delta vs stock |
|---|---:|---:|
| Stock | 32.92 | - |
| Target + draft QKV | 27.62 | -16.1% |
| Draft QKV only `(8192,5120)` | 28.18 | -14.4% |
| Target QKV only `(4096,5120)`, 3-run median | 32.57 | -1.1% |

The target-only runs were 32.57, 33.01, and 30.74 tok/s; their median does not establish a gain. Draft QKV accounts for most of the regression even though its isolated eager GEMM favored torch/hipBLASLt. This closes the backend-switch/threshold direction: further work requires a graph-friendly RDNA kernel or fused draft projection, not another dispatch allowlist.
