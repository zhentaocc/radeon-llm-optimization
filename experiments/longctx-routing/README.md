# Long-context routing

Purpose: prevent EAGLE regressions on 16K prompt traffic at concurrency 4 and above.

Implement a reproducible policy for separate speculative and target-only replicas. Validate mixed traffic, queueing, p99, and failure behavior.

