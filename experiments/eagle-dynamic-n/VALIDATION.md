# Validation: EAGLE3 dynamic N startup fix

## Reproducer

- Hardware: 2× Radeon Pro W7900D (`gfx1100`)
- Runtime: vLLM 0.29.0, ROCm 7.2, target TP=2, draft TP=1
- Model: Qwen3.8-27B + `amd/Qwen3.8-27B-EAGLE3-Full-Vocab`
- Config: max N=3, schedule `[[1,4,3],[5,64,2]]`

Unpatched startup fails in `CudaGraphManager._init_candidates()` when the autoregressive draft manager derives `decode_query_len=0` and passes it to `round_up()`.

## Fix

Apply dynamic query-length expansion only to graph managers whose query contains the maximum speculative tokens plus at least one newly sampled token. The per-step autoregressive draft manager always has `decode_query_len=1` and must retain that static shape.

## Verification

| Check | Result |
|---|---|
| New CPU regression test in production vLLM image | 1 passed |
| Unpatched real service startup | Deterministic `ZeroDivisionError` |
| Patched real W7900 service startup | Passed; OpenAI endpoint ready |
| Patched graph capture | Completed; 7.78 GiB/rank graph pool |
| 128/128 c1 | 36.05 tok/s |
| 128/128 c16 | 153.64 tok/s |
| 512/512 c1 | 30.20 tok/s |

For comparison, static N=3/TP1 produced 33.90, 164.97, and 29.90 tok/s for the same three cells; static N=2/TP1 produced 32.24, 175.86, and 27.86 tok/s. The fix is functionally valid. The first schedule improves low-batch performance but does not yet beat static N=2 at high batch, so schedule tuning continues separately from correctness.

