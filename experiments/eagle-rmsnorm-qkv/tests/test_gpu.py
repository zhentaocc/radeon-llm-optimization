#!/usr/bin/env python3
import json
import math
import statistics
from pathlib import Path

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.utils import rocm_fused_rmsnorm_qkv


K = 5120
N = 7168
EPSILONS = (1e-6, 1e-5)
TOKENS = (1, 2, 3, 4, 5)
DTYPES = (torch.float16, torch.bfloat16)
WARMUP = 12
SAMPLES = 15


def baseline(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    epsilon: float,
    cu_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ops.fused_add_rms_norm(hidden, residual, norm_weight, epsilon)
    return ops.wvSplitK(qkv_weight, hidden, cu_count), residual


def candidate(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    epsilon: float,
    cu_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return ops.wvSplitKRMSNorm(
        qkv_weight, hidden, residual, norm_weight, epsilon, cu_count
    )


def errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-5)
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "max_rel": (delta / denom).max().item(),
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    return {
        "median_ms": statistics.median(values),
        "p10_ms": percentile(values, 0.10),
        "p90_ms": percentile(values, 0.90),
        "cv": statistics.pstdev(values) / mean if mean else 0.0,
    }


def time_arm(fn, source_hidden, source_residual, norm_weight, qkv_weight, eps, cu_count):
    hidden = torch.empty_like(source_hidden)
    residual = torch.empty_like(source_residual)
    for _ in range(WARMUP):
        hidden.copy_(source_hidden)
        residual.copy_(source_residual)
        fn(hidden, residual, norm_weight, qkv_weight, eps, cu_count)
    torch.cuda.synchronize()

    samples = []
    for _ in range(SAMPLES):
        hidden.copy_(source_hidden)
        residual.copy_(source_residual)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(hidden, residual, norm_weight, qkv_weight, eps, cu_count)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def correctness_case(tokens, dtype, eps, norm_weight, qkv_weight, cu_count, kind="random"):
    if kind == "zero":
        hidden = torch.zeros(tokens, K, device="cuda", dtype=dtype)
        residual = torch.zeros_like(hidden)
    elif kind == "constant":
        hidden = torch.full((tokens, K), 0.75, device="cuda", dtype=dtype)
        residual = torch.full_like(hidden, -0.25)
    elif kind == "high_range":
        hidden = torch.linspace(-100, 100, tokens * K, device="cuda").reshape(tokens, K).to(dtype)
        residual = torch.linspace(1, -1, tokens * K, device="cuda").reshape(tokens, K).to(dtype)
    else:
        hidden = (torch.randn(tokens, K, device="cuda") * 2).to(dtype)
        residual = (torch.randn(tokens, K, device="cuda") * 0.5).to(dtype)

    expected_qkv, expected_residual = baseline(
        hidden.clone(), residual.clone(), norm_weight, qkv_weight, eps, cu_count
    )
    actual_qkv, actual_residual = candidate(
        hidden, residual, norm_weight, qkv_weight, eps, cu_count
    )
    torch.cuda.synchronize()
    qkv_error = errors(actual_qkv, expected_qkv)
    residual_error = errors(actual_residual, expected_residual)
    tolerance = 0.06 if dtype == torch.bfloat16 else 0.015
    torch.testing.assert_close(
        actual_qkv, expected_qkv, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "tokens": tokens,
        "epsilon": eps,
        "kind": kind,
        "qkv_error": qkv_error,
        "residual_error": residual_error,
    }


def hip_graph_gate(dtype, norm_weight, qkv_weight, cu_count):
    hidden = torch.randn(3, K, device="cuda", dtype=dtype)
    residual = torch.randn_like(hidden)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        qkv, residual_out = candidate(
            hidden, residual, norm_weight, qkv_weight, 1e-6, cu_count
        )
    qkv_ptr = qkv.data_ptr()
    residual_ptr = residual_out.data_ptr()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert qkv.data_ptr() == qkv_ptr
    assert residual_out.data_ptr() == residual_ptr
    expected_qkv, expected_residual = baseline(
        hidden.clone(), residual.clone(), norm_weight, qkv_weight, 1e-6, cu_count
    )
    torch.testing.assert_close(
        qkv,
        expected_qkv,
        rtol=0.06 if dtype == torch.bfloat16 else 0.015,
        atol=0.06 if dtype == torch.bfloat16 else 0.015,
    )
    torch.testing.assert_close(residual_out, expected_residual, rtol=0, atol=0)
    return {"dtype": str(dtype).removeprefix("torch."), "replays": 2, "status": "pass"}


def compile_gate(dtype, norm_weight, qkv_weight):
    hidden = torch.randn(3, K, device="cuda", dtype=dtype)
    residual = torch.randn_like(hidden)
    compiled = torch.compile(rocm_fused_rmsnorm_qkv, backend="eager", fullgraph=True)
    actual_qkv, actual_residual = compiled(
        hidden, residual, norm_weight, qkv_weight, 1e-6
    )
    expected_qkv, expected_residual = candidate(
        hidden, residual, norm_weight, qkv_weight, 1e-6,
        torch.cuda.get_device_properties(0).multi_processor_count,
    )
    torch.testing.assert_close(actual_qkv, expected_qkv, rtol=0, atol=0)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    return {"dtype": str(dtype).removeprefix("torch."), "status": "pass"}


def main():
    torch.manual_seed(20261022)
    assert torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "")
    assert "gfx1100" in arch, (props.name, arch)
    assert props.total_memory >= 40 * 1024**3, props.total_memory
    cu_count = props.multi_processor_count
    result = {
        "device": props.name,
        "arch": arch,
        "cu_count": cu_count,
        "shape": {"K": K, "N": N},
        "correctness": [],
        "hip_graph": [],
        "torch_compile": [],
        "timing": [],
    }

    for dtype in DTYPES:
        norm_weight = (torch.randn(K, device="cuda") * 0.1 + 1).to(dtype)
        qkv_weight = (torch.randn(N, K, device="cuda") * 0.02).to(dtype)
        for tokens in TOKENS:
            for eps in EPSILONS:
                result["correctness"].append(
                    correctness_case(
                        tokens, dtype, eps, norm_weight, qkv_weight, cu_count
                    )
                )
        for kind in ("zero", "constant", "high_range"):
            result["correctness"].append(
                correctness_case(1, dtype, 1e-6, norm_weight, qkv_weight, cu_count, kind)
            )

        result["hip_graph"].append(
            hip_graph_gate(dtype, norm_weight, qkv_weight, cu_count)
        )
        result["torch_compile"].append(
            compile_gate(dtype, norm_weight, qkv_weight)
        )

        for tokens in TOKENS:
            source_hidden = torch.randn(tokens, K, device="cuda", dtype=dtype)
            source_residual = torch.randn_like(source_hidden)
            baseline_samples = time_arm(
                baseline, source_hidden, source_residual, norm_weight, qkv_weight, 1e-6, cu_count
            )
            candidate_samples = time_arm(
                candidate, source_hidden, source_residual, norm_weight, qkv_weight, 1e-6, cu_count
            )
            baseline_stats = summarize(baseline_samples)
            candidate_stats = summarize(candidate_samples)
            speedup = baseline_stats["median_ms"] / candidate_stats["median_ms"]
            result["timing"].append(
                {
                    "dtype": str(dtype).removeprefix("torch."),
                    "tokens": tokens,
                    "baseline": baseline_stats,
                    "candidate": candidate_stats,
                    "speedup": speedup,
                    "stable_8pct_gate": speedup >= 1.08
                    and baseline_stats["cv"] <= 0.05
                    and candidate_stats["cv"] <= 0.05,
                }
            )

    result["status"] = "pass"
    output = Path("/workspace/gpu_results.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
