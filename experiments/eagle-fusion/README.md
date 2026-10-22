# EAGLE fusion

Purpose: reduce norm, copy, elementwise, and launch overhead in target/draft decode.

Candidates: residual+RMSNorm, RMSNorm+projection, gate+SiLU+mul, and auxiliary hidden-state preparation. Require numerical parity and at least 8% batch-1 TPOT gain.

