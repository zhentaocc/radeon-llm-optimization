# Result storage

Compact summaries are versioned here. Raw JSON, logs, traces, and checkpoints remain on the benchmark host at:

`/nvme0/data/zhenchen/spec-bench-results/qwen3.8-27b-eagle3-w7900/`

The source-of-truth summary files are `matrix_summary.csv` and `tuning_summary.csv`. Contaminated early 16K runs are quarantined under the remote `contaminated/` directory and are not used in reports.

