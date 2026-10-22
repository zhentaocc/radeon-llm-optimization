#!/usr/bin/env python3
"""Compare stock, AITER, and tiled Triton SiLU-and-mul on RDNA3.

The benchmark intentionally calls preallocated-output operators.  Allocation is
outside the timed region so the result measures the GPU kernel plus launch, not
Python tensor allocation.  It covers the Qwen3.8 target TP2 and EAGLE3 draft
TP1 local intermediate widths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import vllm._custom_ops  # noqa: F401 -- registers torch.ops._C

try:
    import aiter
except Exception as exc:  # pragma: no cover - runtime-dependent
    aiter = None
    AITER_IMPORT_ERROR = repr(exc)
else:
    AITER_IMPORT_ERROR = ""


TARGET_TP2_WIDTH = 8_704
DRAFT_TP1_WIDTH = 17_408
MS = (1, 2, 3, 4, 8, 16)
DTYPES = (torch.bfloat16, torch.float16)
TRITON_CONFIGS = (
    (1_024, 4),
    (1_024, 8),
    (2_048, 4),
    (2_048, 8),
    (4_096, 4),
    (4_096, 8),
)


@triton.jit
def _silu_mul_tiled(
    x_ptr,
    out_ptr,
    d: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < d
    input_row = x_ptr + row * (2 * d)
    gate = tl.load(input_row + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(input_row + d + offsets, mask=mask, other=0.0).to(tl.float32)
    value = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
    tl.store(out_ptr + row * d + offsets, value, mask=mask)


def triton_silu_mul(
    out: torch.Tensor,
    x: torch.Tensor,
    *,
    block: int,
    num_warps: int,
) -> None:
    m = x.numel() // x.shape[-1]
    d = x.shape[-1] // 2
    _silu_mul_tiled[(m, triton.cdiv(d, block))](
        x,
        out,
        d=d,
        BLOCK=block,
        num_warps=num_warps,
    )


@dataclass
class Result:
    impl: str
    dtype: str
    m: int
    d: int
    median_us: float | None
    p10_us: float | None
    p90_us: float | None
    cv_pct: float | None
    max_abs_error: float | None
    max_rel_error: float | None
    correct: bool
    hip_graph: str
    error: str


def reference(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    return (F.silu(gate) * up).to(x.dtype)


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    delta = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-5)
    return delta.max().item(), (delta / denom).max().item()


def tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 4e-2, 4e-2
    return 1e-2, 1e-2


def time_op(op: Callable[[], None], warmup: int, inner: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            op()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1_000.0 / inner)
    return samples


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = q * (len(ordered) - 1)
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    weight = index - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def graph_check(op: Callable[[], None], actual: torch.Tensor, expected: torch.Tensor) -> str:
    # Compile outside capture and warm the allocator on a side stream.
    op()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            op()
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        op()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    atol, rtol = tolerance(actual.dtype)
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError("HIP graph replay changed output")
    return "pass"


def make_impls(out: torch.Tensor, x: torch.Tensor) -> list[tuple[str, Callable[[], None]]]:
    impls: list[tuple[str, Callable[[], None]]] = [
        ("stock_vllm", lambda: torch.ops._C.silu_and_mul(out, x)),
    ]
    if aiter is not None:
        # AITER's decorated Python wrapper converts torch.Tensor to its
        # experimental aiter_tensor_t.  That adapter is incompatible with the
        # Torch 2.12 build in the stock image, while the registered dispatcher
        # schema accepts ordinary torch tensors.  Call the same operator
        # directly so the comparison reaches the packaged GPU kernel.
        impls.append(
            ("aiter", lambda: torch.ops.aiter.silu_and_mul(out, x, 0.0))
        )
    for block, warps in TRITON_CONFIGS:
        impls.append(
            (
                f"triton_b{block}_w{warps}",
                lambda block=block, warps=warps: triton_silu_mul(
                    out, x, block=block, num_warps=warps
                ),
            )
        )
    return impls


def stable_error(name: str, exc: Exception) -> str:
    message = repr(exc)
    if name == "aiter" and "aiter_tensor_t" in message:
        return "AITER aiter_tensor_t binding mismatch in stock Torch/AITER image"
    return message


def run_case(
    *,
    dtype: torch.dtype,
    m: int,
    d: int,
    warmup: int,
    inner: int,
    repeats: int,
) -> list[Result]:
    torch.manual_seed(1234 + m + d)
    x = torch.randn((m, 2 * d), dtype=dtype, device="cuda")
    expected = reference(x)
    out = torch.empty((m, d), dtype=dtype, device="cuda")
    atol, rtol = tolerance(dtype)
    results = []
    for name, op in make_impls(out, x):
        out.fill_(float("nan"))
        graph_status = "not-run"
        try:
            op()
            torch.cuda.synchronize()
            max_abs, max_rel = error_metrics(out, expected)
            correct = torch.allclose(out, expected, atol=atol, rtol=rtol)
            if not correct:
                raise AssertionError(
                    f"incorrect output: abs={max_abs:.6g}, rel={max_rel:.6g}"
                )
            samples = time_op(op, warmup, inner, repeats)
            if name.startswith("triton_") and m in (1, 16):
                graph_status = graph_check(op, out, expected)
            median = statistics.median(samples)
            mean = statistics.mean(samples)
            cv = statistics.pstdev(samples) / mean * 100.0 if mean else 0.0
            results.append(
                Result(
                    impl=name,
                    dtype=str(dtype).removeprefix("torch."),
                    m=m,
                    d=d,
                    median_us=median,
                    p10_us=percentile(samples, 0.10),
                    p90_us=percentile(samples, 0.90),
                    cv_pct=cv,
                    max_abs_error=max_abs,
                    max_rel_error=max_rel,
                    correct=True,
                    hip_graph=graph_status,
                    error="",
                )
            )
        except Exception as exc:
            torch.cuda.synchronize()
            results.append(
                Result(
                    impl=name,
                    dtype=str(dtype).removeprefix("torch."),
                    m=m,
                    d=d,
                    median_us=None,
                    p10_us=None,
                    p90_us=None,
                    cv_pct=None,
                    max_abs_error=None,
                    max_rel_error=None,
                    correct=False,
                    hip_graph=graph_status,
                    error=stable_error(name, exc),
                )
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--inner", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device_name = torch.cuda.get_device_name(0)
    metadata = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "device": device_name,
        "warmup": args.warmup,
        "inner": args.inner,
        "repeats": args.repeats,
        "aiter_import_error": AITER_IMPORT_ERROR,
        "widths": {
            "target_tp2": TARGET_TP2_WIDTH,
            "draft_tp1": DRAFT_TP1_WIDTH,
        },
        "m_values": MS,
        "dtypes": [str(x).removeprefix("torch.") for x in DTYPES],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    results: list[Result] = []
    for dtype in DTYPES:
        for d in (TARGET_TP2_WIDTH, DRAFT_TP1_WIDTH):
            for m in MS:
                case = run_case(
                    dtype=dtype,
                    m=m,
                    d=d,
                    warmup=args.warmup,
                    inner=args.inner,
                    repeats=args.repeats,
                )
                results.extend(case)
                for row in case:
                    print(json.dumps(asdict(row), sort_keys=True), flush=True)

    fields = list(Result.__dataclass_fields__)
    with (args.output_dir / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in results)


if __name__ == "__main__":
    main()
