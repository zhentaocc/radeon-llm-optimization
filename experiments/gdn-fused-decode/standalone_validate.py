from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
import vllm

import vllm._C_stable_libtorch  # noqa: F401
from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn


assert "/work/vllm/" in vllm.__file__
assert hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp")
op = torch.ops._C.fused_gdn_decode_post_conv_mtp


@torch.inference_mode()
def run_case(
    mtp_width: int,
    head_ratio: int,
    state_dtype: torch.dtype,
    norm_dtype: torch.dtype,
    dt_bias_dtype: torch.dtype,
    activation: str,
) -> dict[str, object]:
    h = 2
    hv = h * head_ratio
    k = v = 128
    num_tokens = mtp_width
    num_slots = mtp_width + 1
    scale = k**-0.5
    eps = 1e-6

    dtype_seed = {
        torch.float32: 0,
        torch.bfloat16: 1,
        torch.float16: 2,
    }[dt_bias_dtype]
    seed = (
        20000
        + 100 * mtp_width
        + 10 * head_ratio
        + int(state_dtype == torch.bfloat16)
        + 3 * dtype_seed
    )
    torch.manual_seed(seed)
    mixed_qkv = torch.randn(
        num_tokens, 2 * h * k + hv * v, dtype=torch.bfloat16, device="cuda"
    )
    query, key, value = torch.split(mixed_qkv, [h * k, h * k, hv * v], dim=-1)
    query = query.view(1, num_tokens, h, k)
    key = key.view(1, num_tokens, h, k)
    value = value.view(1, num_tokens, hv, v)
    ba = torch.randn(num_tokens, 2 * hv, dtype=torch.bfloat16, device="cuda")
    b, a = ba.chunk(2, dim=-1)
    a_log = 0.5 * torch.randn(hv, dtype=torch.float32, device="cuda")
    dt_bias = (
        0.1 * torch.randn(hv, dtype=torch.float32, device="cuda")
    ).to(dt_bias_dtype)
    output_gate = torch.randn(
        num_tokens, hv, v, dtype=torch.bfloat16, device="cuda"
    )
    norm_weight = torch.randn(v, dtype=norm_dtype, device="cuda")
    state_ref = (
        0.01
        * torch.randn(num_slots, hv, v, k, dtype=torch.float32, device="cuda")
    ).to(state_dtype)
    state_actual = state_ref.clone()
    state_indices = torch.arange(
        1, num_slots, dtype=torch.int32, device="cuda"
    ).view(1, mtp_width)
    cu_seqlens = torch.tensor([0, mtp_width], dtype=torch.int32, device="cuda")
    accepted = torch.tensor([mtp_width], dtype=torch.int32, device="cuda")

    raw_ref, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=a_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=state_ref,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=accepted,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )
    expected = rmsnorm_fn(
        raw_ref.squeeze(0),
        norm_weight,
        None,
        z=output_gate,
        eps=eps,
        norm_before_gate=True,
        activation=activation,
    )
    actual = torch.empty_like(output_gate)
    op(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        state_indices,
        cu_seqlens,
        accepted,
        state_actual,
        output_gate,
        norm_weight,
        actual,
        scale,
        eps,
        activation,
    )
    torch.cuda.synchronize()

    output_rel_l2 = float(
        ((actual.float() - expected.float()).norm()
         / expected.float().norm().clamp_min(1e-20)).item()
    )
    state_abs_max = float((state_actual.float() - state_ref.float()).abs().max().item())
    torch.testing.assert_close(state_actual, state_ref, atol=3e-2, rtol=3e-2)
    assert output_rel_l2 < 5e-4, output_rel_l2
    return {
        "mtp_width": mtp_width,
        "head_ratio": head_ratio,
        "state_dtype": str(state_dtype),
        "norm_dtype": str(norm_dtype),
        "dt_bias_dtype": str(dt_bias_dtype),
        "activation": activation,
        "output_relative_l2": output_rel_l2,
        "state_abs_max": state_abs_max,
    }


@torch.inference_mode()
def run_ragged(dt_bias_dtype: torch.dtype) -> dict[str, object]:
    torch.manual_seed(30000 + int(dt_bias_dtype == torch.bfloat16) + 2 * int(
        dt_bias_dtype == torch.float16
    ))
    query_lengths = (4, 2, 0)
    accepted_steps = (1, 2, 4)
    num_requests = len(query_lengths)
    state_width = 4
    num_tokens = sum(query_lengths)
    h, ratio, k, v = 2, 3, 128, 128
    hv = h * ratio
    scale = k**-0.5
    eps = 1e-6

    mixed_qkv = torch.randn(
        num_tokens, 2 * h * k + hv * v, dtype=torch.bfloat16, device="cuda"
    )
    query, key, value = torch.split(mixed_qkv, [h * k, h * k, hv * v], dim=-1)
    query = query.view(1, num_tokens, h, k)
    key = key.view(1, num_tokens, h, k)
    value = value.view(1, num_tokens, hv, v)
    ba = torch.randn(num_tokens, 2 * hv, dtype=torch.bfloat16, device="cuda")
    b, a = ba.chunk(2, dim=-1)
    a_log = 0.5 * torch.randn(hv, dtype=torch.float32, device="cuda")
    dt_bias = (
        0.1 * torch.randn(hv, dtype=torch.float32, device="cuda")
    ).to(dt_bias_dtype)
    output_gate = torch.randn(
        num_tokens, hv, v, dtype=torch.bfloat16, device="cuda"
    )
    norm_weight = torch.randn(v, dtype=torch.float32, device="cuda")
    num_slots = num_requests * state_width + 1
    state_ref = 0.01 * torch.randn(
        num_slots, hv, v, k, dtype=torch.float32, device="cuda"
    )
    state_actual = state_ref.clone()
    state_indices = torch.arange(
        1, num_slots, dtype=torch.int32, device="cuda"
    ).view(num_requests, state_width)
    state_indices[-1].zero_()
    cu_seqlens = torch.tensor(
        [0, 4, 6, 6], dtype=torch.int32, device="cuda"
    )
    accepted = torch.ones(num_requests, dtype=torch.int32, device="cuda")

    max_output_relative_l2 = 0.0
    for accepted_count in accepted_steps:
        accepted.fill_(accepted_count)
        accepted[-1] = 1
        raw_ref, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=a_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=query,
            k=key,
            v=value,
            initial_state=state_ref,
            inplace_final_state=True,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=state_indices,
            num_accepted_tokens=accepted,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )
        expected = rmsnorm_fn(
            raw_ref.squeeze(0),
            norm_weight,
            None,
            z=output_gate,
            eps=eps,
            norm_before_gate=True,
            activation="silu",
        )
        actual = torch.empty_like(output_gate)
        op(
            mixed_qkv,
            a,
            b,
            a_log,
            dt_bias,
            state_indices,
            cu_seqlens,
            accepted,
            state_actual,
            output_gate,
            norm_weight,
            actual,
            scale,
            eps,
            "silu",
        )
        torch.cuda.synchronize()
        output_relative_l2 = float(
            ((actual.float() - expected.float()).norm()
             / expected.float().norm().clamp_min(1e-20)).item()
        )
        max_output_relative_l2 = max(max_output_relative_l2, output_relative_l2)
        assert output_relative_l2 < 5e-4, output_relative_l2
        torch.testing.assert_close(state_actual, state_ref, atol=3e-2, rtol=3e-2)

    return {
        "query_lengths": query_lengths,
        "accepted_steps": accepted_steps,
        "dt_bias_dtype": str(dt_bias_dtype),
        "max_output_relative_l2": max_output_relative_l2,
    }


@torch.inference_mode()
def run_negative_guards() -> dict[str, bool]:
    torch.manual_seed(31000)
    h, hv, k, v = 1, 3, 128, 128

    def inputs(num_tokens: int):
        mixed_qkv = torch.randn(
            num_tokens, 2 * h * k + hv * v, dtype=torch.bfloat16, device="cuda"
        )
        ba = torch.randn(num_tokens, 2 * hv, dtype=torch.bfloat16, device="cuda")
        b, a = ba.chunk(2, dim=-1)
        return mixed_qkv, a, b

    mixed_qkv, a, b = inputs(3)
    state = torch.randn(4, hv, v, k, dtype=torch.float32, device="cuda")
    state_before = state.clone()
    out = torch.full((3, hv, v), 7.0, dtype=torch.bfloat16, device="cuda")
    op(
        mixed_qkv,
        a,
        b,
        torch.randn(hv, dtype=torch.float32, device="cuda"),
        torch.randn(hv, dtype=torch.float16, device="cuda"),
        torch.tensor([[1, 2]], dtype=torch.int32, device="cuda"),
        torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        state,
        torch.randn(3, hv, v, dtype=torch.bfloat16, device="cuda"),
        torch.randn(v, dtype=torch.float32, device="cuda"),
        out,
        k**-0.5,
        1e-6,
        "silu",
    )
    torch.cuda.synchronize()
    assert torch.count_nonzero(out) == 0
    torch.testing.assert_close(state, state_before, atol=0, rtol=0)

    mixed_qkv, a, b = inputs(1)
    state = torch.randn(3, hv, v, k, dtype=torch.float32, device="cuda")
    state_before = state.clone()
    out = torch.full((1, hv, v), 7.0, dtype=torch.bfloat16, device="cuda")
    op(
        mixed_qkv,
        a,
        b,
        torch.randn(hv, dtype=torch.float32, device="cuda"),
        torch.randn(hv, dtype=torch.bfloat16, device="cuda"),
        torch.tensor([[3]], dtype=torch.int32, device="cuda"),
        torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        state,
        torch.randn(1, hv, v, dtype=torch.bfloat16, device="cuda"),
        torch.randn(v, dtype=torch.float32, device="cuda"),
        out,
        k**-0.5,
        1e-6,
        "silu",
    )
    torch.cuda.synchronize()
    assert torch.count_nonzero(out) == 0
    torch.testing.assert_close(state, state_before, atol=0, rtol=0)

    mixed_qkv, a, b = inputs(2)
    state = torch.randn(3, hv, v, k, dtype=torch.float32, device="cuda")
    state_before = state.clone()
    out = torch.zeros((2, hv, v), dtype=torch.bfloat16, device="cuda")
    op(
        mixed_qkv,
        a,
        b,
        torch.randn(hv, dtype=torch.float32, device="cuda"),
        torch.randn(hv, dtype=torch.float32, device="cuda"),
        torch.tensor([[1, 3]], dtype=torch.int32, device="cuda"),
        torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        state,
        torch.randn(2, hv, v, dtype=torch.bfloat16, device="cuda"),
        torch.randn(v, dtype=torch.float32, device="cuda"),
        out,
        k**-0.5,
        1e-6,
        "silu",
    )
    torch.cuda.synchronize()
    assert torch.count_nonzero(out) > 0
    assert torch.count_nonzero(state[1] - state_before[1]) > 0
    torch.testing.assert_close(state[0], state_before[0], atol=0, rtol=0)
    torch.testing.assert_close(state[2], state_before[2], atol=0, rtol=0)
    return {
        "tokens_exceed_width": True,
        "source_slot_oob": True,
        "destination_slot_oob": True,
    }


@torch.inference_mode()
def run_hip_graph(state_dtype: torch.dtype) -> dict[str, object]:
    torch.manual_seed(32000 + int(state_dtype == torch.bfloat16))
    num_tokens, h, ratio, k, v = 2, 2, 3, 128, 128
    hv = h * ratio
    mixed_qkv = torch.randn(
        num_tokens, 2 * h * k + hv * v, dtype=torch.bfloat16, device="cuda"
    )
    ba = torch.randn(num_tokens, 2 * hv, dtype=torch.bfloat16, device="cuda")
    b, a = ba.chunk(2, dim=-1)
    a_log = torch.randn(hv, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(hv, dtype=torch.float32, device="cuda")
    state_initial = (
        0.01 * torch.randn(3, hv, v, k, dtype=torch.float32, device="cuda")
    ).to(state_dtype)
    output_gate = torch.randn(
        num_tokens, hv, v, dtype=torch.bfloat16, device="cuda"
    )
    norm_weight = torch.randn(v, dtype=torch.float32, device="cuda")
    state_indices = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    accepted = torch.tensor([2], dtype=torch.int32, device="cuda")

    def invoke(state: torch.Tensor, output: torch.Tensor) -> None:
        op(
            mixed_qkv,
            a,
            b,
            a_log,
            dt_bias,
            state_indices,
            cu_seqlens,
            accepted,
            state,
            output_gate,
            norm_weight,
            output,
            k**-0.5,
            1e-6,
            "silu",
        )

    eager_state = state_initial.clone()
    eager_output = torch.empty_like(output_gate)
    invoke(eager_state, eager_output)
    torch.cuda.synchronize()

    graph_state = state_initial.clone()
    graph_output = torch.empty_like(output_gate)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke(graph_state, graph_output)

    replay_results = []
    for replay in range(2):
        graph_state.copy_(state_initial)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output, eager_output, atol=0, rtol=0)
        torch.testing.assert_close(graph_state, eager_state, atol=0, rtol=0)
        replay_results.append({"replay": replay + 1, "bit_exact": True})

    return {
        "state_dtype": str(state_dtype),
        "capture": True,
        "replays": replay_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("standalone_validation.json"),
    )
    cli_args = parser.parse_args()
    rows = []
    matrix = itertools.product(
        (1, 2, 3, 5),
        (1, 2, 3, 4, 8),
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.bfloat16, torch.float16),
        ("silu", "sigmoid"),
    )
    for case_args in matrix:
        rows.append(run_case(*case_args))

    ragged = [
        run_ragged(dtype)
        for dtype in (torch.float32, torch.bfloat16, torch.float16)
    ]
    negative_guards = run_negative_guards()
    hip_graph = [run_hip_graph(dtype) for dtype in (torch.float32, torch.bfloat16)]
    summary = {
        "cases": len(rows),
        "max_output_relative_l2": max(r["output_relative_l2"] for r in rows),
        "max_state_abs_error": max(r["state_abs_max"] for r in rows),
        "device": torch.cuda.get_device_name(0),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "ragged": ragged,
        "negative_guards": negative_guards,
        "hip_graph": hip_graph,
    }
    cli_args.output.parent.mkdir(parents=True, exist_ok=True)
    cli_args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("FULL_EXTENSION_CORRECTNESS", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
