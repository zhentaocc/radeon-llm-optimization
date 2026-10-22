# EAGLE auxiliary RMSNorm preparation fusion

Status: **micro-validated / E2E not promotable**

This candidate targets Qwen3 EAGLE3 with two auxiliary hidden states. It
replaces two concatenations plus two per-state RMSNorm launches with one
Triton custom op that writes the normalized states directly into the FC input.
The fast path is guarded by model type, state count, shape, dtype, device,
weight layout, and equal RMSNorm epsilon; all other cases use stock behavior.

## Artifact

- Patch: [`vllm-v0.29.0.patch`](vllm-v0.29.0.patch)
- Stock snapshot: `9ca3a6b`
- Validated W7900 implementation head: `5e7f227`
- Patch SHA256:
  `f5f9e556c403c341d2e68eab7b352bc63f2be729f8deda7a194e925e4836e670`

The patch is an mbox series containing the functional fusion followed by the
`torch.compile`-safe custom-op registration.

## Evidence

- GPU: Radeon Pro W7900D (`gfx1100`), GPU7 only
- Runtime: ROCm 7.2.3, PyTorch 2.12, vLLM 0.29.0
- Shape: two BF16 `[tokens, 5120]` states
- Correctness: 12/12 tests passed
- Coverage: reference parity, Qwen3 wiring, unequal-epsilon fallback,
  HIP Graph capture/replay, and `torch.compile(fullgraph=True)`
- Profiler: four stock launches to one fused launch
- Microbenchmark: 2.59-3.01x kernel-region speedup for 1-64 tokens

Detailed timings and kernel counts are in
[`microbench.csv`](microbench.csv) and [`profile.csv`](profile.csv).
These are isolated kernel results, not serving throughput or latency gains.

## Service gate

Stock and aux-only servers ran serially on the same GPU0/1 pair with target
TP2, draft TP1, static N=2, identical seed and request order. The four single
runs covered ShareGPT short and fixed-256 SpecBench at concurrency 1 and 16;
all requests succeeded. Full values are in [`e2e.csv`](e2e.csv).

- ShareGPT throughput changed by +0.46% at c1 and -0.12% at c16; TPOT changed
  by -0.10% and -0.20%, respectively.
- SpecBench throughput changed by +4.58% at c1 and +1.63% at c16 in the first
  stock/aux sequence.
- The apparent c1 gain was caused by a low stock control. Two added stock runs
  measured 34.249 and 34.354 tok/s versus 33.085 tok/s initially. The stock
  three-run median is 34.249 tok/s, making the single aux result 34.602 tok/s,
  or +1.03%, inside the observed control range.

Because the only larger signal disappears after control repetition and the
other cases are within roughly 2%, this candidate is not promotable as a
standalone service optimization. The patch remains useful evidence that this
launch boundary is too small to move end-to-end performance.
