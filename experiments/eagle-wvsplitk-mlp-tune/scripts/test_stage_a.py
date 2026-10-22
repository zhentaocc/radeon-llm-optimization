#!/usr/bin/env python3
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
import vllm._rocm_C  # noqa: F401 - registers torch.ops._rocm_C


SHAPES = {
    "target_up_m3": (17408, 5120, 3),
    "target_down_m3": (5120, 8704, 3),
    "draft_up_m1": (34816, 5120, 1),
    "draft_down_m1": (5120, 17408, 1),
}
CONFIGS = tuple(range(10))
WARMUP = 12
SAMPLES = 21
SEED = 20261022


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summary(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    return {
        "median_us": statistics.median(values),
        "p10_us": percentile(values, 0.10),
        "p90_us": percentile(values, 0.90),
        "cv": statistics.pstdev(values) / mean if mean else 0.0,
    }


def error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-5)
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "max_rel": (delta / denom).max().item(),
        "bitwise_equal": torch.equal(actual, expected),
    }


def stock(weight: torch.Tensor, x: torch.Tensor, cu_count: int) -> torch.Tensor:
    return torch.ops._rocm_C.wvSplitK(weight, x, None, cu_count)


def candidate(
    weight: torch.Tensor, x: torch.Tensor, cu_count: int, config: int
) -> torch.Tensor:
    return torch.ops._rocm_C.wvSplitKSweep(weight, x, None, cu_count, config)


def elapsed_us(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    end.synchronize()
    assert out.is_cuda
    return start.elapsed_time(end) * 1000.0


def graph_gate(weight, x, cu_count, config) -> dict[str, object]:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        static_out = candidate(weight, x, cu_count, config)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = candidate(weight, x, cu_count, config)
    pointer = static_out.data_ptr()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    return {
        "status": "pass",
        "replays": 2,
        "stable_output_pointer": static_out.data_ptr() == pointer,
    }


def main() -> None:
    assert torch.cuda.device_count() == 1, "container must expose only physical GPU0"
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "")
    assert arch.startswith("gfx1100"), arch
    cu_count = props.multi_processor_count
    assert cu_count == 48, cu_count
    torch.manual_seed(SEED)
    random.seed(SEED)

    result = {
        "status": "running",
        "timestamp_unix": time.time(),
        "device": props.name,
        "arch": arch,
        "cu_count": cu_count,
        "dtype": "bfloat16",
        "warmup": WARMUP,
        "samples": SAMPLES,
        "seed": SEED,
        "shapes": {},
        "hip_graph": [],
    }

    for shape_name, (out_features, in_features, tokens) in SHAPES.items():
        weight = torch.randn(
            out_features, in_features, dtype=torch.bfloat16, device="cuda"
        ) * 0.02
        x = torch.randn(
            tokens, in_features, dtype=torch.bfloat16, device="cuda"
        ) * 0.02
        torch.cuda.synchronize()
        expected = stock(weight, x, cu_count)
        correctness = {}
        for config in CONFIGS:
            actual = candidate(weight, x, cu_count, config)
            torch.cuda.synchronize()
            metrics = error(actual, expected)
            if metrics["max_abs"] > 0.015625:
                raise AssertionError((shape_name, config, metrics))
            correctness[str(config)] = metrics

        arms = ["stock", *CONFIGS]
        for _ in range(WARMUP):
            stock(weight, x, cu_count)
            for config in CONFIGS:
                candidate(weight, x, cu_count, config)
        torch.cuda.synchronize()

        samples: dict[str, list[float]] = {str(arm): [] for arm in arms}
        for sample_idx in range(SAMPLES):
            order = list(arms)
            random.Random(SEED + sample_idx).shuffle(order)
            for arm in order:
                if arm == "stock":
                    value = elapsed_us(lambda: stock(weight, x, cu_count))
                else:
                    value = elapsed_us(
                        lambda arm=arm: candidate(weight, x, cu_count, arm)
                    )
                samples[str(arm)].append(value)

        timing = {arm: summary(values) for arm, values in samples.items()}
        stock_median = timing["stock"]["median_us"]
        for config in CONFIGS:
            timing[str(config)]["speedup_vs_stock"] = (
                stock_median / timing[str(config)]["median_us"]
            )
        result["shapes"][shape_name] = {
            "out_features": out_features,
            "in_features": in_features,
            "tokens": tokens,
            "correctness": correctness,
            "timing": timing,
            "raw_samples_us": samples,
        }
        del expected, weight, x
        torch.cuda.empty_cache()

    graph_configs = {0}
    for shape_data in result["shapes"].values():
        for config in CONFIGS:
            if shape_data["timing"][str(config)]["speedup_vs_stock"] >= 1.05:
                graph_configs.add(config)

    for config in sorted(graph_configs):
        for shape_name, (out_features, in_features, tokens) in SHAPES.items():
            weight = torch.zeros(
                out_features, in_features, dtype=torch.bfloat16, device="cuda"
            )
            x = torch.zeros(
                tokens, in_features, dtype=torch.bfloat16, device="cuda"
            )
            gate = graph_gate(weight, x, cu_count, config)
            gate.update({"config": config, "shape": shape_name})
            result["hip_graph"].append(gate)
            del weight, x
            torch.cuda.empty_cache()

    best = {}
    any_five_percent = False
    for shape_name, shape_data in result["shapes"].items():
        candidates = {
            int(config): data["speedup_vs_stock"]
            for config, data in shape_data["timing"].items()
            if config != "stock"
        }
        winner = max(candidates, key=candidates.get)
        best[shape_name] = {"config": winner, "speedup": candidates[winner]}
        any_five_percent |= candidates[winner] >= 1.05
    result["best"] = best
    result["any_candidate_ge_5pct"] = any_five_percent
    result["stage_b_admitted"] = any_five_percent
    result["status"] = "pass"

    output_path = Path("/workspace/stage_a_results.json")
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "best": best,
                "stage_b_admitted": any_five_percent,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
