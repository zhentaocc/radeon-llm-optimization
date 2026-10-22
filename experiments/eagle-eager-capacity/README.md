# EAGLE eager-mode KV-capacity control

Status: **configured / E2E infrastructure-blocked**

This experiment tests whether removing graph residency can recover enough KV
cache to improve 8K-16K, concurrency-16 serving. It uses stock vLLM 0.29.0 and
the established Qwen3.8-27B + full-vocabulary EAGLE3 configuration. The only
runtime delta is `--enforce-eager`; exact settings are in
[`launch-config.txt`](launch-config.txt).

`--enforce-eager` is a deliberately broad capacity control. vLLM reports that
it disables both CUDAGraphs and `torch.compile`, so any eventual performance
result would measure their combined removal rather than graph memory alone.

## Planned gate

The two planned cases preserve the existing real-data seed, prompt count,
ordering, and output policy:

| Dataset | Concurrency | Prompts | Warmups | Output policy |
| --- | ---: | ---: | ---: | --- |
| LongBench `gov_report` 8K-16K | 16 | 16 | 1 | Dataset-native, thinking off, shuffle disabled |
| SpecBench fixed256 | 16 | 64 | 2 | 256 tokens, ignore EOS, thinking off |

Both cases use seed 42 and temperature 0. The intended comparisons are stock
default EAGLE N=2 and target-only baseline, reporting output throughput, TTFT,
TPOT, end-to-end latency, acceptance, KV capacity, and VRAM.

## Startup evidence

No benchmark request was sent. [`startup-attempts.csv`](startup-attempts.csv)
records all starts:

- The first GPU0/1 diagnostic start hit the known transient PYNCCL HIP
  `invalid argument` failure.
- Its permitted retry passed PYNCCL and entered model loading, but a root-owned
  shared-host sweep appeared on the same GPUs. The server was stopped before
  readiness or sampling.
- A formal GPU0/1 start followed two clean preflight snapshots and passed
  PYNCCL, but new sweep processes appeared during model loading. Contamination
  evidence was saved and the test server was stopped with zero requests.
- A formal GPU2/3 start followed two clean preflight snapshots, each showing
  0% GPU use and 0% VRAM with no sweep process. It failed during PYNCCL
  initialization with `NCCL error: unhandled cuda error` and HIP
  `invalid argument`, before model execution.

The last failure exhausted the agreed retry budget. This result neither
supports nor rejects eager mode as a capacity optimization; it establishes an
infrastructure blocker. Re-run only on an exclusively reserved TP2 pair whose
PYNCCL startup has already been verified.

## Reference capacity, not eager results

Preserved stock logs provide context only:

| Configuration | Graph residency | Available KV memory | KV tokens | 32K maximum concurrency |
| --- | ---: | ---: | ---: | ---: |
| Default EAGLE N=2 | 6.64 GiB speculator + 6.63 GiB target | 5.62 GiB | 143,384 | 4.38x |
| Target-only baseline | 2.54 GiB + 2.56 GiB | 12.05 GiB | 366,273 | 11.18x |

These values motivated the experiment but are not measurements of
`--enforce-eager`. None of the uncontaminated eager starts reached the KV
capacity report.
