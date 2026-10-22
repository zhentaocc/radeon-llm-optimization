# Combined EAGLE RMSNorm preparation fusions

Status: **micro-validated / E2E rejected**

This candidate stacks two Qwen3 EAGLE3 launch-reduction changes:

1. auxiliary hidden-state preparation: two concatenations plus two FC
   RMSNorms become one custom-op launch;
2. first draft-layer input preparation: two RMSNorms plus concatenation become
   one custom-op launch.

The patch preserves each component's dtype, device, shape, epsilon, and model
semantic guards. Unsupported cases use the original implementation.

## Artifact

- Patch: [`vllm-v0.29.0.patch`](vllm-v0.29.0.patch)
- Stock snapshot: `9ca3a6b`
- Validated stacked implementation head: `e35ab11`
- Patch SHA256:
  `489fdd47404acbf24a75455b72f39ee8856e20f2d5583d5c82149e5a7b1458df`

The patch is a stock-based mbox series and does not require either independent
experiment branch to be applied first.

## Evidence

- GPU: Radeon Pro W7900D (`gfx1100`), GPU7 only
- Runtime: ROCm 7.2.3, PyTorch 2.12, vLLM 0.29.0
- Component correctness suite: 14/14 tests passed
- Coverage includes reference parity, semantic fallbacks, HIP Graph
  capture/replay, and `torch.compile(fullgraph=True)`
- Component profiler sum: seven stock launches become two fused launches
- Auxiliary microbenchmark: 2.59-3.01x for its isolated kernel region
- Layer0 microbenchmark: 2.00-2.44x for its isolated kernel region

The launch total is the sum of two separate component profiles, not a full
model trace. The timing data in [`microbench.csv`](microbench.csv) and
[`profile.csv`](profile.csv) are isolated kernel evidence, not serving gains.

## Service result

Stock and combined servers ran serially on GPU0/1 with target TP2, draft TP1,
static N=2, identical seed and request order. The initial four-case gate found
ShareGPT changes of -0.32% throughput at c1 and -1.91% at c16; fixed-256
SpecBench changed +3.98% at c1 and +1.02% at c16. All requests succeeded.

The larger c1 signal was retested because the first stock run was low. Three
fixed-256 c1 runs per arm produced these medians:

| Metric | Stock median | Combined median | Delta |
| --- | ---: | ---: | ---: |
| Output throughput | 34.249 tok/s | 34.227 tok/s | -0.07% |
| Mean TPOT | 27.843 ms | 27.878 ms | +0.13% (worse) |
| Mean TTFT | 374.367 ms | 370.424 ms | -1.05% |

Run ranges were 3.6-3.8% for stock and 5.5-5.8% for combined. An independent
100,000-resample bootstrap over the three run-level observations gave 95%
intervals of [-5.38%, +3.98%] for throughput, [-3.73%, +5.82%] for TPOT, and
[-5.80%, +3.09%] for TTFT. Every interval crosses zero. Per-run values are in
[`e2e.csv`](e2e.csv).

The combined candidate is therefore rejected for promotion: it reduces launch
count in isolation but has no reproducible service-level benefit. A future
attempt should fuse a larger boundary instead of adding another small
preparation kernel.
