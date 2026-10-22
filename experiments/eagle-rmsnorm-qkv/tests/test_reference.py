#!/usr/bin/env python3
import json
import os

os.environ.setdefault("VLLM_ROCM_USE_FUSED_RMSNORM_QKV", "0")

import torch

from vllm.model_executor.layers.utils import (
    can_use_rocm_fused_rmsnorm_qkv,
    is_rocm_fused_rmsnorm_qkv_shape_supported,
    rocm_fused_rmsnorm_qkv,
    rocm_fused_rmsnorm_qkv_fake,
)


def rounded_residual(
    hidden: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    # This explicitly mirrors scalar_t z = hidden + residual in the current
    # fused_add_rms_norm kernel: add, then round to the activation dtype.
    return (hidden.float() + residual.float()).to(hidden.dtype)


def reference(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_out = rounded_residual(hidden, residual)
    inv_rms = torch.rsqrt(residual_out.float().square().mean(-1, keepdim=True) + epsilon)
    normalized = (residual_out.float() * inv_rms * norm_weight.float()).to(hidden.dtype)
    qkv = (normalized.float() @ qkv_weight.float().T).to(hidden.dtype)
    return qkv, residual_out, normalized


def test_reference_math() -> int:
    torch.manual_seed(20261022)
    cases = 0
    for dtype in (torch.float16, torch.bfloat16):
        for tokens in range(1, 6):
            for epsilon in (1e-6, 1e-5):
                hidden = (torch.randn(tokens, 64) * 3).to(dtype)
                residual = (torch.randn(tokens, 64) * 0.5).to(dtype)
                norm_weight = (torch.randn(64) * 0.25 + 1).to(dtype)
                qkv_weight = (torch.randn(112, 64) * 0.02).to(dtype)
                qkv, residual_out, normalized = reference(
                    hidden, residual, norm_weight, qkv_weight, epsilon
                )
                expected_residual = (hidden + residual).to(dtype)
                torch.testing.assert_close(residual_out, expected_residual, rtol=0, atol=0)
                expected_inv = torch.rsqrt(
                    expected_residual.float().square().mean(-1, keepdim=True)
                    + epsilon
                )
                expected_norm = (
                    expected_residual.float() * expected_inv * norm_weight.float()
                ).to(dtype)
                torch.testing.assert_close(normalized, expected_norm, rtol=0, atol=0)
                assert qkv.shape == (tokens, 112)
                assert qkv.dtype == dtype
                cases += 1

        for values in (
            torch.zeros(2, 64),
            torch.ones(2, 64),
            torch.linspace(-100, 100, 128).reshape(2, 64),
        ):
            hidden = values.to(dtype)
            residual = (-0.25 * values).to(dtype)
            norm_weight = torch.ones(64, dtype=dtype)
            qkv_weight = torch.eye(64, dtype=dtype)
            qkv, residual_out, normalized = reference(
                hidden, residual, norm_weight, qkv_weight, 1e-6
            )
            assert torch.isfinite(qkv).all()
            torch.testing.assert_close(qkv, normalized, rtol=0, atol=0)
            assert residual_out.dtype == dtype
            cases += 1
    return cases


def test_shape_gate_and_fake() -> int:
    cases = 0
    for dtype in (torch.float16, torch.bfloat16):
        norm_weight = torch.empty(5120, device="meta", dtype=dtype)
        qkv_weight = torch.empty(7168, 5120, device="meta", dtype=dtype)
        for tokens in range(1, 6):
            hidden = torch.empty(tokens, 5120, device="meta", dtype=dtype)
            residual = torch.empty_like(hidden)
            assert is_rocm_fused_rmsnorm_qkv_shape_supported(
                hidden, residual, norm_weight, qkv_weight, None
            )
            qkv, residual_out = rocm_fused_rmsnorm_qkv_fake(
                hidden, residual, norm_weight, qkv_weight, 1e-6
            )
            assert qkv.shape == (tokens, 7168)
            assert residual_out.shape == (tokens, 5120)
            assert qkv.dtype == residual_out.dtype == dtype
            cases += 1

        hidden6 = torch.empty(6, 5120, device="meta", dtype=dtype)
        assert not is_rocm_fused_rmsnorm_qkv_shape_supported(
            hidden6, torch.empty_like(hidden6), norm_weight, qkv_weight, None
        )
        hidden = torch.empty(3, 5120, dtype=dtype)
        residual = torch.empty_like(hidden)
        norm_cpu = torch.empty(5120, dtype=dtype)
        qkv_cpu = torch.empty(7168, 5120, dtype=dtype)
        assert not can_use_rocm_fused_rmsnorm_qkv(
            hidden, residual, norm_cpu, qkv_cpu, None
        )
        assert not is_rocm_fused_rmsnorm_qkv_shape_supported(
            hidden, residual, norm_cpu, qkv_cpu, torch.empty(7168, dtype=dtype)
        )
        assert not is_rocm_fused_rmsnorm_qkv_shape_supported(
            hidden[:, ::2],
            residual[:, ::2],
            norm_cpu[::2],
            qkv_cpu[:, ::2],
            None,
        )
        cases += 4
    return cases


def test_fullgraph_fake() -> int:
    hidden = torch.empty(3, 5120, device="meta", dtype=torch.bfloat16)
    residual = torch.empty_like(hidden)
    norm_weight = torch.empty(5120, device="meta", dtype=torch.bfloat16)
    qkv_weight = torch.empty(7168, 5120, device="meta", dtype=torch.bfloat16)
    compiled = torch.compile(rocm_fused_rmsnorm_qkv, backend="eager", fullgraph=True)
    qkv, residual_out = compiled(
        hidden, residual, norm_weight, qkv_weight, 1e-6
    )
    assert qkv.shape == (3, 7168)
    assert residual_out.shape == (3, 5120)
    return 1


def main() -> None:
    result = {
        "reference_cases": test_reference_math(),
        "shape_and_fake_cases": test_shape_gate_and_fake(),
        "fullgraph_fake_cases": test_fullgraph_fake(),
        "tp_local_qkv_layout": [7168, 5120],
        "status": "pass",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
