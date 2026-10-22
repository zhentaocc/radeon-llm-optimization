# Long-context attention and prefill

Purpose: improve 16K throughput without worsening p99 or starving decode.

Sweep chunk sizes and attention backends, record prefill/decode overlap, and test admission-control changes across concurrency 1/4/16.

