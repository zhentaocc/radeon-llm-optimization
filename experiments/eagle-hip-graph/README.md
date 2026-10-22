# EAGLE HIP graph capture-set optimization

Status: v1 memory win validated but misses c1 verification shape; v2 defined, startup blocked by host PYNCCL state; 16K/c16 v1 residency gate running.

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

The c1 loss is explained by omitting shape `3`: static N=2 verifies three tokens for a single request, so c1 falls to piecewise execution. V2 captures every active batch size 1–16 plus every N=2 verification multiple through 48:

```text
[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,21,24,27,30,33,36,39,42,45,48,56,64]
```

V2 should preserve full-graph decode across the deployed concurrency range while still cutting the default 54-shape set nearly in half. Its first startup attempt was blocked by `hipIpcGetMemHandle failed: invalid argument` affecting every new TP2 service on GPU0/1 and GPU2/3. Disabling PyNCCL reaches model load, but torch.distributed fails identically at the first all-reduce. Promote only after a host/GPU reset, then require v2 to stay within 3% at c1/c4/c16 and pass the 16K residency gate.
