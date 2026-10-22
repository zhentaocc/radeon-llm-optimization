# EAGLE HIP graph capture-set optimization

Status: v1 and v2 measured; both reduce graph memory, but both fail throughput/long-context promotion gates.

The first reduced set was:

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

Throughput gates for v1:

- 128/128 c1: 31.67, 30.98, 31.42 tok/s; median 31.42. A same-period default repeat reaches 32.88, making the loss 4.4%, outside the 3% gate.
- 128/128 c16: 200.30 tok/s; the same-period default is 204.23, a 1.9% loss.
- 512/512 c1: 27.95 tok/s, +0.3% versus the 27.86 tuning control.
- 16K/256 c16: stopped after 16/32 requests and 2,413 s. The partial run produced 4,096 tokens at about 1.70 tok/s, 61.9% below the completed default EAGLE result (4.46 tok/s). V1 is rejected for long-context residency despite its larger KV cache.

The c1 loss is explained by omitting shape `3`: static N=2 verifies three tokens for a single request, so c1 falls to piecewise execution. V2 captures every active batch size 1–16 plus every N=2 verification multiple through 48:

```text
[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,21,24,27,30,33,36,39,42,45,48,56,64]
```

After stale TP contexts were cleared, v2 started successfully. It uses 4.26 GiB of graph memory per rank and leaves 210,751 KV tokens (6.43x theoretical 32K concurrency), versus 6.64 GiB and 150,241 tokens (4.59x) for the same N=2 default control.

| Workload | V2 tok/s | Same-period default tok/s | Delta |
|---|---:|---:|---:|
| 128/128 c1 | 31.33 | 32.92 | -4.8% |
| 128/128 c16 | 173.11 | 173.60 | -0.3% |
| 512/512 c1 | 26.23 | 28.12 | -6.7% |

V2 therefore recovers about 40% KV capacity and passes the c16 throughput gate, but misses the 3% gate at c1 and long decode. Do not promote either fixed capture set. The next design should derive shapes from scheduler telemetry and separately retain latency-critical c1/long-decode graphs rather than optimizing only for graph count.
