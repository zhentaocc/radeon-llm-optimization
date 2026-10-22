# W7900 AgentX-derived 32K optimization smoke

## Scope and validity

This is a local optimization experiment, not an official InferenceX AgentX result.

- `official=false`
- `submission_valid=false`
- The official `semianalysisai/cc-traces-weka-062126` corpus has **0/393** traces eligible under a 32,768-token server limit. No official Qwen3.8-27B + EAGLE3 N=2 golden curve was available.
- To exercise real trajectory shapes on W7900, a fixed-seed derived corpus keeps valid prefixes below 31,744 recorded tokens, preserving event order and timing. It contains 14 traces and 57 requests; no branch survived the dependency-preserving truncation.
- Candidate acceptance is synthetic and driven by a local, non-official SPEED-Bench coding calibration. It is not the measured acceptance of the derived AgentX requests.

## Result

| Concurrency | Server | Completed requests | Elapsed (s) | Mean / p50 latency (s) | Mean / p50 TTFT (s) | Mean / p50 ITL (ms) | Output tok/s | Mean / p50 OSL | Errors / cancelled |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | target only | 1 | 520.394 | 520.394 / 520.394 | 19.085 / 19.085 | 362.217 / 362.217 | 2.661 | 1385 / 1385 | 0 / 0 |
| 1 | synthetic EAGLE3 N=2 | 1 | 175.928 | 175.928 / 175.928 | 18.822 / 18.822 | 113.515 / 113.515 | 7.872 | 1385 / 1385 | 0 / 0 |
| 4 | target only | 3 | 576.556 | 287.016 / 159.789 | 43.280 / 48.544 | 487.735 / 406.282 | 3.112 | 598 / 259 | 0 / 0 |
| 4 | synthetic EAGLE3 N=2 | 3 | 274.800 | 180.951 / 139.379 | 65.962 / 72.389 | 333.575 / 160.544 | 6.528 | 598 / 259 | 0 / 0 |

The c1 candidate improves elapsed time and throughput by **2.958x**, and mean ITL by **3.191x**. The c4 candidate improves elapsed time and throughput by **2.098x**, mean latency by **1.586x**, and mean ITL by **1.462x**.

At c4, mean TTFT regresses from 43.280 s to 65.962 s (**+52.4%**), even while decode and end-to-end time improve. Synthetic acceptance reduces decode work but does not reduce prefill. This small smoke includes long input sequences (20,709-31,486 tokens), concurrent TP=2 prefills on W7900, and disabled prefix caching. With only three completed requests, TTFT is strongly schedule-sensitive and must not be treated as a steady-state production result.

The c4 request shape is the same for baseline and candidate: two root sessions were sent, one completed, one was released while still open, and one branch was spawned and completed. Three HTTP requests completed with no errors or cancellations. One of four requested lanes could not acquire a root trajectory in the short send window.

## Local acceptance calibration

The local calibration uses the official NVIDIA SPEED-Bench revision `fbd6dc64353a3153bf24252a113ebd6d2cff9dec`: 80 coding rows resolving to 89 HTTP turns. All 89 requests succeeded at concurrency 1, temperature 0, OSL 4096, and seed 20261022.

- accepted tokens: 120,008
- drafts: 110,868
- draft tokens: 221,736
- local acceptance length: `120008 / 110868 + 1 = 2.0824403795504565`
- local acceptance rate: `120008 / 221736 = 0.5412201897752282`

The candidate server uses `rejection_sample_method=synthetic`, N=2, and `synthetic_acceptance_length=2.08244037955`. Realized smoke-run AL was 2.0610 at c1 and 2.0740 at c4.

## Derivation and limitations

The server context limit is 32,768 tokens. A 1,024-token safety headroom sets an effective recorded-context cap of 31,744. For each stream, the derivation keeps the prefix before the first request that exceeds this cap; branches are kept only if spawn/join dependencies remain valid. Traces need at least two root requests. Selection uses seed 20261022 and preserves event ordering, inter-event timing, and hashed identifiers. Both baseline and candidate use streaming, `ignore_eos=true`, server token counts, and first-turn cache busting.

This procedure scanned 393 source traces, selected 14 derivable traces, and dropped 379. The maximum retained peak is 31,574 tokens. It changes the official workload distribution and cannot be submitted as AgentX. The 60-second send window and tiny completed sample are intended only for optimization direction. Raw traces, prompt/response records, host paths, IPs, and service logs are intentionally excluded from Git; cryptographic hashes in the manifest preserve auditability.

## Reproduction

1. Produce the fixed derived trace manifest outside the repository; do not modify the official corpus.
2. Run real EAGLE3 N=2 on SPEED-Bench coding and merge disjoint counters with `scripts/merge_speedbench_calibrations.py`.
3. Start a clearly labeled synthetic-acceptance server with `scripts/w7900_launch_eagle_synthetic_local_al.sh` only after exclusive GPU/port ownership is confirmed.
4. Run c1 and c4 with `scripts/agentx_derived_smoke.sh`.
5. Summarize each run with `scripts/summarize_agentx_derived.py` and compare against `results/w7900-agentx-derived/manifest.json`.

The synthetic server used for this experiment was stopped after evidence capture; GPU4/5 and port 18121 were released.
