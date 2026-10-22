# Fix EAGLE dynamic N

Purpose: make mixed per-batch speculative lengths start and run on ROCm.

Reproducer: N=3 maximum with schedule `[[1,4,3],[5,64,2]]` fails in graph candidate generation because `decode_query_len` becomes zero. Add a regression test and validate against static N=2/3.

