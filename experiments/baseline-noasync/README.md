# Strict no-async baseline

Status: complete (12/12 cells, zero request failures).

This target-only control uses the same no-async scheduler mode as EAGLE3. It closely matches the earlier async baseline and therefore rules scheduler choice out as the cause of EAGLE's workload-dependent regressions.

| Workload | c1 tok/s | c4 tok/s | c16 tok/s |
|---|---:|---:|---:|
| 128/128 | 20.78 | 70.58 | 160.87 |
| 2048/256 | 13.07 | 41.23 | 81.09 |
| 16384/256 | 3.46 | 8.74 | 10.75 |
| 512/512 | 18.01 | 62.96 | 145.06 |

Against this strict control, static EAGLE N=2/TP1 changes throughput by +57.0%/+1.7%/+23.0% for 128/128, +76.5%/+12.1%/−16.2% for 2048/256, +126.5%/−20.6%/−58.5% for 16K/256, and +54.7%/+12.9%/+9.3% for 512/512 at c1/c4/c16.

The consolidated `results/matrix_summary.csv` is the canonical machine-readable result.
