# Validation: EAGLE3 dynamic N startup fix

## Reproducer and fix

- Hardware: 2× Radeon Pro W7900D (`gfx1100`)
- Runtime: vLLM 0.29.0, ROCm 7.2, target TP=2, draft TP=1
- Model: Qwen3.8-27B + `amd/Qwen3.8-27B-EAGLE3-Full-Vocab`
- Config: max N=3 with a mixed N=3/N=2 batch schedule

Unpatched startup fails in `CudaGraphManager._init_candidates()` because dynamic target query-length expansion is also applied to the autoregressive draft graph manager. Its `decode_query_len=1` becomes zero and reaches `round_up()`.

The fix applies expansion only when `decode_query_len > num_speculative_tokens`. The one-token draft manager retains its static shape.

## Verification

| Check | Result |
|---|---|
| Unit regression test in production vLLM image | 1 passed |
| Unpatched real service startup | Deterministic `ZeroDivisionError` |
| Patched real W7900 service startup | Passed; endpoint ready |
| Patched graph capture | Completed; 7.78 GiB/rank |

| Schedule | 128/128 c1 | 128/128 c16 | 512/512 c1 |
|---|---:|---:|---:|
| `[[1,4,3],[5,64,2]]` | 36.05 | 153.64 | 30.20 |
| `[[1,1,3],[2,64,2]]` | 35.33 | 148.67 | 29.55 |
| Static N=3/TP1 | 33.90 | 164.97 | 29.90 |
| Static N=2/TP1 | 32.24 | 175.86 | 27.86 |

Correctness is fixed, but performance is not ready for promotion. Dynamic N forces piecewise graph execution: the tighter schedule still loses 15.5% at c16 versus static N=2. Keep the patch and regression test, but deploy separate static-N replicas or request routing until the dynamic graph path recovers within 3% of static throughput.
