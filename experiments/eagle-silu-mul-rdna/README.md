# EAGLE SiLU-and-mul RDNA3 specialization

Status: micro-validated negative result; no service A/B and no vLLM source patch.

This candidate does not add another gate + SiLU + multiply API fusion. vLLM
already performs that fusion through `SiluAndMul`. It tests whether the existing
single-workgroup-per-token kernel leaves gfx1100 compute idle at the small token
counts used by target decode and the one-layer EAGLE3 draft.

## Evidence and actual shapes

- Target config: 64 layers, intermediate width 17,408. TP2 shards the fused
  gate/up projection, so each rank calls SiLU-and-mul with input width 17,408
  and output width `d=8,704`.
- Draft config: one layer, TP1, intermediate width 17,408, so its input width
  is 34,816 and output width is `d=17,408`.
- The autoregressive proposer performs one initial draft forward plus
  `range(1, num_speculative_steps)`. At the deployed `N=2`, source therefore
  implies two one-layer draft activation calls per proposer step. A runtime
  trace is still required to confirm padding and actual M values.
- Existing target trace: each rank contains 640 BF16 launches, totaling
  2.634 ms (rank 0) and 2.655 ms (rank 1). Rank 0 has 64 calls at `M=11` and
  576 calls at `M=1`; the per-launch median is 4.079 us. Rank 1 is 4.119 us.
- No prior draft trace records the activation input shape or launch count. That
  must be measured during a clean GPU window; the config- and source-derived
  statements above are not being presented as measured runtime counts.

The stock vLLM HIP/C++ kernel maps one token to one workgroup and uses 128-bit
packed loads/stores. For BF16/FP16 at these widths it launches 1,024 threads per
token. At `M=1`, only one workgroup is available to the whole W7900 even though
the row contains 8,704 or 17,408 outputs.

The candidate uses a two-dimensional grid, `M x ceil(d/BLOCK)`, to expose more
workgroups at small M. The sweep compares 1,024/2,048/4,096-element tiles with
four or eight Triton warps. This mapping change is the hypothesis under test;
there is no source patch or service integration unless the isolated median is
at least 10% faster and stable.

## Benchmark contract

`microbench_silu_mul.py` compares preallocated-output implementations:

1. stock `torch.ops._C.silu_and_mul`;
2. installed AITER `aiter.silu_and_mul`;
3. the tiled Triton RDNA3 candidate.

Coverage is BF16 and FP16, target/draft widths, and
`M=1,2,3,4,8,16`. Every row checks a FP32 reference before timing. Triton
variants also compile and capture/replay under HIP graph for `M=1` and `M=16`.
Timing uses nine repeat medians by default and reports p10/p90 and coefficient
of variation.

Promotion gate:

- correct for every dtype/shape;
- compile and HIP graph capture/replay pass;
- at least 10% median gain over the faster usable stock/AITER implementation;
- stable across repeats, with no gain based only on a single outlier shape.

Only a candidate passing all gates advances to a same-GPU service A/B.

## W7900 result

The formal run used physical GPU0 after two clean preflight snapshots. A
five-second process monitor observed no returning `sweep_mamba`/`csgrid`
workload, and the postflight snapshot was clean. Each timing is the median of
nine samples, with 500 launches per sample after 50 warmups.

| dtype | local output width | stock range, M=1..16 | best tiled Triton range | Triton slowdown |
|---|---:|---:|---:|---:|
| BF16 | 8,704 target TP2 | 5.49–5.57 us | 11.65–11.74 us | 109.0–113.2% |
| BF16 | 17,408 draft TP1 | 6.95–7.06 us | 11.66–11.78 us | 65.6–69.5% |
| FP16 | 8,704 target TP2 | 5.17–5.24 us | 11.66–11.74 us | 122.6–126.3% |
| FP16 | 17,408 draft TP1 | 6.32–6.42 us | 11.63–11.73 us | 81.3–85.0% |

All 144 Triton dtype/shape/config rows pass the FP32-reference correctness
check. All 48 requested HIP graph capture/replay checks pass. The large and
consistent regression shows that splitting each token row into more
workgroups costs more launch/scheduling overhead than it recovers through
additional CU occupancy on gfx1100. No block/warp choice wins any tested cell.

The installed AITER activation is not a valid comparison candidate in this
image. Import and registration succeed, but all 24 shapes fail before launch:
the `develop=True` wrapper converts `torch.Tensor` to the pybind
`aiter_tensor_t`, then the packaged `module_activation` rejects that same type.
Calling the registered dispatcher directly reaches the identical binding
failure. This is recorded as unavailable, not as a performance result.

The tiled specialization fails the 10% promotion gate by a wide margin, so the
experiment stops before service A/B. Keep the compact raw data in
[`results/microbench.csv`](results/microbench.csv) and environment metadata in
[`results/metadata.json`](results/metadata.json). The next useful activation
work would need to modify the existing native packed HIP kernel or fuse the
activation into an adjacent projection; another Triton block mapping is not
justified by these results.
