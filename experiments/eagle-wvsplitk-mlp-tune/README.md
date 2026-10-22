# Native `wvSplitK` MLP parameter sweep

Status: Stage A passed correctness and HIP Graph gates, but stopped early because no dominant production cell improved by 5%. This is a negative result; production dispatch is unchanged.

## Why this candidate

The target profile contains 2,314 `wvSplitK_hf_sml` launches in ten steps (about 286 ms and 54% of classified rank-0 kernel time). Existing backend-switch experiments showed that `wvSplitK` is already the right backend for the large Qwen3.8 MLP-up and draft MLP-down shapes, so this experiment tunes the production HIP kernel itself rather than redirecting it to torch/hipBLASLt.

The Qwen3.8-27B target has hidden size 5,120, intermediate size 17,408, 64 layers, and TP2. The EAGLE3 draft has the same hidden/intermediate sizes, one layer, and TP1. The resulting dominant decode cells are:

| Cell | TP-local weight `(out,in)` | Tokens |
|---|---:|---:|
| target up | `(17408,5120)` | 3 |
| target down | `(5120,8704)` | 3 |
| draft up | `(34816,5120)` | 1 |
| draft down | `(5120,17408)` | 1 |

On gfx1100, stock uses 32 threads/wave, 16 waves/group, `A_CHUNK=8`, 48 CUs, and either `(YTILE,UNRL)=(2,2)` for M=1 or `(4,2)` for M=3. `benchmark-only-wvsplitk-sweep.patch` adds a separate operator that sweeps 10 bounded combinations of waves/group, grid multiplier, YTILE, and UNRL. It does not alter `wvSplitK` or its dispatch.

## Stage A result

Environment: W7900D/gfx1100, ROCm 7.2.3 production image `specbench:vllm-0.29.0-rocm723-gfx1100`, BF16, seed 20261022, 12 warmups, 21 randomized interleaved samples per arm, physical GPU0 only.

The real ROCm build completed all 55/55 CMake/Ninja targets, including `skinny_gemms.hip.o`, linked `_rocm_C.abi3.so`, and copied it into the isolated vLLM tree. A later optional package step reported that no Rust compiler was installed; this occurred after the requested HIP extension was built and did not affect the loaded operator or GPU run.

| Cell | Stock median | Best config | Best median | p10–p90 | Gain | CV |
|---|---:|---:|---:|---:|---:|---:|
| target up M3 | 234.880 us | 8 | 230.961 us | 229.680–232.280 us | +1.70% | 0.48% |
| target down M3 | 123.120 us | 0 | 123.080 us | 122.640–124.001 us | +0.03% | 0.48% |
| draft up M1 | 438.321 us | 9 | 434.641 us | 434.200–435.081 us | +0.85% | 0.10% |
| draft down M1 | 229.161 us | 8 | 226.641 us | 226.040–227.161 us | +1.11% | 0.18% |

All 40 candidate/shape comparisons are bitwise equal to the stock operator (`max_abs=0`). Stock-equivalent config 0 passes HIP Graph capture and two replays for all four shapes with stable output pointers. The host monitor records GPU0 at 0% use and 0% VRAM before the run; the only process using GPU0 during timing was this experiment's Python process. Retained vLLM workers were on GPU1.

No cell reaches the approved +5% Stage B admission threshold, far below the +10% production-patch gate. Therefore FP16, M=2/4/5, additional `A_CHUNK`/UNRL combinations, and service A/B were intentionally not run.

## Reproduction and artifacts

1. Apply `benchmark-only-wvsplitk-sweep.patch` to clean vLLM 0.29.0.
2. Build the ROCm extension with `scripts/build_static.sh` inside the production image.
3. Place the built tree and scripts under `/dev/shm/zhenchen-wvsplitk-mlp-tune-vllm` on the W7900 host.
4. Run `scripts/launch_stage_a_host.sh` on the host.

Compact result: `results/stage_a_summary.json`. Full private artifacts remain at:

- remote: `/dev/shm/zhenchen-wvsplitk-mlp-tune-vllm/stage_a_results.json`
- local staging: `C:\Users\zhenchen\wvsplitk-mlp-dev\stage_a_results.json`
- full JSON SHA-256: `eaafd81e623cd197e5a8c36acb51d7347677814f367914e8e30732c7cc3b7314`
- GPU monitor SHA-256: `0b0bd61ed12896624fbcbc20a2315b4246662300e253c0d12057b8ae0e40f34c`
- run log SHA-256: `8ac6e8aebe6aceacf483fa9013bb607fc2c015b1594d323e1b974eadceb2c4c0`

The full JSON contains only synthetic random tensors and timing data, no prompts, generated text, or credentials.
