#!/usr/bin/env python3
"""Compare vLLM's RDNA skinny GEMM with the torch ROCm GEMM path."""

import argparse
import csv
import statistics

import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops


SHAPES = [
    (4096, 5120, "target_qkv"),
    (5120, 3072, "target_o"),
    (17408, 5120, "target_mlp_up"),
    (5120, 8704, "target_mlp_down"),
    (8192, 5120, "draft_qkv"),
    (9216, 5120, "sweep_9k"),
    (10240, 5120, "sweep_10k"),
    (11264, 5120, "sweep_11k"),
    (12288, 5120, "sweep_12k"),
    (14336, 5120, "sweep_14k"),
    (15360, 5120, "sweep_15k"),
    (16384, 5120, "sweep_16k"),
    (5120, 6144, "draft_o"),
    (34816, 5120, "draft_mlp_up"),
    (5120, 17408, "draft_mlp_down"),
    (124160, 5120, "target_lm_head"),
]


def measure(fn, warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples), statistics.quantiles(samples, n=100)[98]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    cu_count = props.multi_processor_count
    rows = []

    for out_features, in_features, name in SHAPES:
        weight = torch.randn(
            out_features, in_features, device=device, dtype=torch.bfloat16
        )
        for tokens in (1, 2, 3, 4, 5):
            x = torch.randn(tokens, in_features, device=device, dtype=torch.bfloat16)
            skinny = lambda: ops.wvSplitK(weight, x, cu_count, None)
            torch_gemm = lambda: F.linear(x, weight)

            y_skinny = skinny()
            y_torch = torch_gemm()
            max_abs = (y_skinny.float() - y_torch.float()).abs().max().item()
            skinny_median, skinny_p99 = measure(
                skinny, args.warmup, args.iterations
            )
            torch_median, torch_p99 = measure(
                torch_gemm, args.warmup, args.iterations
            )
            llmm1_median = None
            llmm1_p99 = None
            if tokens == 1 and in_features <= 8192:
                llmm1 = lambda: ops.LLMM1(weight, x, 4)
                llmm1_median, llmm1_p99 = measure(
                    llmm1, args.warmup, args.iterations
                )
            rows.append(
                {
                    "name": name,
                    "tokens": tokens,
                    "out_features": out_features,
                    "in_features": in_features,
                    "skinny_median_us": skinny_median,
                    "skinny_p99_us": skinny_p99,
                    "torch_median_us": torch_median,
                    "torch_p99_us": torch_p99,
                    "llmm1_median_us": llmm1_median,
                    "llmm1_p99_us": llmm1_p99,
                    "skinny_speedup": torch_median / skinny_median,
                    "max_abs_diff": max_abs,
                }
            )
        del weight
        torch.cuda.empty_cache()

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        llmm1_text = (
            f" llmm1={row['llmm1_median_us']:.1f}us"
            if row["llmm1_median_us"] is not None
            else ""
        )
        print(
            f"{row['name']:9s} M={row['tokens']} "
            f"skinny={row['skinny_median_us']:.1f}us "
            f"torch={row['torch_median_us']:.1f}us "
            f"speedup={row['skinny_speedup']:.2f}x "
            f"max_abs={row['max_abs_diff']:.4f}{llmm1_text}"
        )


if __name__ == "__main__":
    main()
