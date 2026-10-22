# EAGLE first-layer RMSNorm and concatenation fusion

Status: **micro-validated / E2E infrastructure-blocked**

This stock-based candidate targets Qwen3 EAGLE3 draft layer 0. It replaces
embedding RMSNorm, hidden-state RMSNorm, and concatenation with one Triton
custom op. It is independent of the auxiliary-state integration change; the
only shared plumbing is the generic two-input RMSNorm-and-concat custom op.

The fast path requires `norm_before_residual == false`, equal input shapes,
matching input/weight dtype and device, and equal RMSNorm epsilon. Unsupported
semantics fall back to the original layer implementation.

## Artifact

- Patch: [`vllm-v0.29.0.patch`](vllm-v0.29.0.patch)
- Stock snapshot: `9ca3a6b`
- Validated W7900 implementation head: `7053e93`
- Patch SHA256:
  `e9f1efba43ed7757b3d7306c9aba1db79bb2da5a075ca69e078a51aeb20f5952`

## Evidence

- GPU: Radeon Pro W7900D (`gfx1100`), GPU7 only
- Runtime: ROCm 7.2.3, PyTorch 2.12, vLLM 0.29.0
- Shape: two BF16 `[tokens, 5120]` inputs
- Correctness: 4/4 tests passed
- Coverage: Qwen3 layer wiring, `norm_before_residual` fallback,
  HIP Graph capture/replay, and `torch.compile(fullgraph=True)`
- Profiler: three stock launches to one fused launch
- Microbenchmark: 2.00-2.44x kernel-region speedup for 1-64 tokens

Detailed timings and kernel counts are in
[`microbench.csv`](microbench.csv) and [`profile.csv`](profile.csv).
These are isolated kernel results, not serving throughput or latency gains.

## Service gate

The layer0-only patch was mounted from the independent `7053e93` worktree and
started twice on GPU2/3 at port 18123 with the same TP2 target, TP1 draft,
static N=2 configuration used by the stock service. Both attempts failed in
PYNCCL initialization with `NCCL error: unhandled cuda error` and HIP
`invalid argument`, before weights, graph capture, or any request reached the
patched code. See [`e2e.csv`](e2e.csv).

This is an infrastructure-blocked E2E result, not a patch regression. The
single-GPU correctness, compile, graph replay, launch-count, and microbenchmark
evidence remains valid. Re-run the service gate only on a GPU pair with a
known-good TP2 PYNCCL start; no end-to-end performance claim is currently
supported.
