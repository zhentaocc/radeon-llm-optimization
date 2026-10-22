# EAGLE small-M GEMM

Purpose: optimize BF16 GEMM/GEMV shapes introduced by draft N=2/3 and target verification.

Work: collect shapes, microbenchmark available backends, add N/batch-aware dispatch, then run the full matrix. Accept only with at least 10% decode gain and no material regression.

