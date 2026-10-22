# TP communication

Purpose: reduce rank1 PYNCCL wait and improve W7900 TP2 scaling.

Compare topology pairs, fused GEMM+reduce, overlap, and feasible TP/PP controls. Keep the GPU6/7 communicator issue isolated from kernel changes.

