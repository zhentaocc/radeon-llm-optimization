# W7900 real-data EAGLE3 benchmark

This experiment compares target-only Qwen3.8-27B TP2 with EAGLE3 `N=2`
(target TP2, draft TP1) on two W7900D pairs. Both arms use temperature 0,
thinking disabled, prefix caching disabled, seed 42, and identical prompt
selection at concurrency 1/4/16.

## Result summary

The complete matrix contains 18 natural-EOS cases plus six fixed-output
SpecBench cases. Every request completed, and paired input-length lists and
input-token totals match exactly.

| Workload | C | Baseline output tok/s | EAGLE output tok/s | Delta | EAGLE p50 TTFT | EAGLE p50 TPOT |
|---|---:|---:|---:|---:|---:|---:|
| SpecBench fixed 256 | 1 | 18.52 | 34.44 | +85.98% | 362.80 ms | 27.40 ms |
| SpecBench fixed 256 | 4 | 59.65 | 78.24 | +31.16% | 372.71 ms | 47.50 ms |
| SpecBench fixed 256 | 16 | 140.84 | 214.16 | +52.06% | 674.14 ms | 63.66 ms |
| LongBench gov_report 8K-16K | 1 | 4.74 | 12.83 | +170.55% | 7.45 s | 63.50 ms |
| LongBench gov_report 8K-16K | 4 | 12.95 | 17.44 | +34.68% | 25.73 s | 169.14 ms |
| LongBench gov_report 8K-16K | 16 | 26.48 | 14.73 | -44.36% | 150.75 s | 408.20 ms |

The long-context c16 regression is a capacity/admission failure rather than a
decode-kernel failure: EAGLE still reduces p50 TPOT by 12.05%, but its p50
TTFT rises 112.85%. Startup capacity falls from 366,273 to 150,241 KV-cache
tokens, while graph memory grows from 2.56 to 6.63 GiB. Until that is fixed,
route high-concurrency 8K-16K traffic to a target-only replica.

See [report.md](report.md) for all latency/throughput percentiles,
[summary.csv](summary.csv) for machine-readable metrics, and
[validation.csv](validation.csv) for paired-load and acceptance checks.
[results/](results/) contains all 24 detailed result JSONs after whitelist-only
sanitization; input/output length arrays, latency samples, aggregates, metadata,
and speculative acceptance counters are preserved.

## Reproduce

1. Pin the model, image, dataset revisions, and runtime settings recorded in
   [provenance.json](provenance.json).
2. Run `prepare_client_overlay.sh` to extract the client loader from the exact
   image and apply [vllm-specbench-loader.patch](vllm-specbench-loader.patch).
   This only forwards the requested seed and chat-template arguments for
   SpecBench; server code is unchanged.
3. Download the pinned LongBench `gov_report` parquet, then build the derived
   8K-16K JSONL with `prepare_longbench.py`. Verify its SHA256 against
   `longbench-gov_report-8k16k-manifest.json`.
4. Provide the ShareGPT and official SpecBench files at the paths expected by
   the run scripts, or adjust the variables at the top of the scripts.
5. Start both services with `launch_servers.sh`, run `run_realdata_arm.sh`
   once per arm, then run `run_fixed_arm.sh` once per arm for strict
   fixed-256 SpecBench comparisons.
6. Run `validate_realdata.py` and `summarize_realdata.py` over the result root.
   `sample_manifest.json` contains the prompt hashes and requested output
   lengths used in the reported run.
7. Before publishing detailed JSON, run `sanitize_results.py`. The checked-in
   [sanitization manifest](results/sanitization_manifest.json) records all
   source/sanitized hashes, the whitelist, removed fields, and sanitizer SHA256
   `e43b5451f05a3953a05f73a3ea329bd92c8de256ba94889e2c2044e2a671d2b9`.

The natural-EOS SpecBench c1 output totals differ by 5.16%, so its raw output
tok/s is not the primary A/B metric. Use `specbench_fixed256` for performance
claims. Deterministic sampling and temperature 0 do not imply bit-identical
generated text across target-only and speculative numerical paths; validation
therefore gates prompt/token matching separately from text equality.

## Full artifacts

Raw result JSON with generated text, client/server logs, Prometheus snapshots,
and source datasets are intentionally excluded from git. Public result JSON is
rebuilt from a strict whitelist; `generated_texts` and `errors` are never
copied. Only their paired exact-match count is retained as a derived aggregate.
The complete private 121-file artifact set is retained at:

- W7900 persistent mirror: `/nvme0/data/zhenchen/radeon-realdata-results/20261022-final`
- W7900 run directory: `/dev/shm/zhenchen-realdata-20261022`
- Windows backup: `C:\Users\zhenchen\realdata-staging-20261022\w7900-results-final`

Raw artifact SHA256 values before Git line-ending normalization:

- `summary.csv`: `b4f4f6ecef89b43b15cd65ec4fbac4be70899f324290b572d24c768c77b3cfbe`
- `validation.csv`: `87e90120886487a882c943747c1ab0989b1cf4696c753767aeb9cdec92d9d439`
- `report.md`: `5c8c33fcbdb39a05e6b29c3e27443c072c36498299309832cb18389f27ded13d`
