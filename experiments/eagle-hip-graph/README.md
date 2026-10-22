# EAGLE HIP graph capture-set optimization

Status: candidate; 16K/c16 residency gate running.

Static N=2 needs full-graph shapes only for the deployed decode/verification range. Launch with:

```text
--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64]}'
```

| Metric | Default capture set | Reduced capture set | Change |
|---|---:|---:|---:|
| Graph memory/rank | ~7.3 GiB | 1.85–1.87 GiB | ~−74.5% |
| Available KV memory/rank | ~5.17 GiB (N=3 reference) | 10.67 GiB | +106% |
| KV token capacity | ~124,769 | 272,391 | +118% |
| Theoretical 32K concurrency | 3.81× | 8.31× | +118% |
| Graph capture time | ~45 s | 13–19 s | −58% to −71% |

Throughput gates:

- 128/128 c1: 31.67, 30.98, 31.42 tok/s; median 31.42, −2.5% versus the 32.24 static-N tuning control.
- 128/128 c16: 200.30 tok/s on the repeat, neutral versus the 197.91 full-matrix control.
- 512/512 c1: 27.95 tok/s, +0.3% versus the 27.86 tuning control.

This is the strongest configuration-only optimization so far. Promote after the 16K/c16 result confirms that the added KV residency translates into throughput/p99 improvement.
