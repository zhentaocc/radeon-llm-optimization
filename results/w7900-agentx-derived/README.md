# Sanitized AgentX-derived result

`manifest.json` is the reviewable, whitelisted result for the W7900 32K optimization smoke. It contains no prompt/response text, raw trace IDs, host paths, IP addresses, credentials, or service logs.

Raw profiler exports and server logs are intentionally not committed. Their SHA-256 values are recorded in the manifest so a repository reviewer with access to the private staging bundle can verify provenance.

This result is `official=false` and `submission_valid=false`. The official AgentX corpus has zero traces eligible at 32K; the committed numbers describe a fixed-seed, locally derived workload and synthetic acceptance calibrated from local SPEED-Bench coding traffic.
