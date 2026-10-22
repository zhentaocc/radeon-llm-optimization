# Qwen3.8/EAGLE3 fusion audit on W7900

Status: audited; two roadmap fusions already exist, remaining work narrowed to norm+projection and EAGLE auxiliary-state preparation.

## Already fused in vLLM 0.29.0

- `residual + RMSNorm`: `Qwen3NextRMSNorm(hidden_states, residual)` returns the normalized tensor and updated residual through the native fused-add RMSNorm path. Runtime logs select `fused_add_rms_norm=['native']`.
- `gate + SiLU + mul`: dense Qwen3.8 MLP imports `Qwen2MoeMLP`, whose merged gate/up projection feeds `SiluAndMul()` before the down projection.
- Full-attention Q/K processing: Qwen3.8 has a fused gated split + QK-RMSNorm + partial NeoX RoPE + gate-copy path.

Treating these as missing features would duplicate existing work. They may still need gfx1100 kernel tuning, but the API/model fusion boundary is already present.

## Remaining high-value boundaries

1. Fuse input RMSNorm with QKV/GDN projection reads so the 5120-wide activation is not reread and rewritten between kernels.
2. Fuse post-attention RMSNorm with merged gate/up projection where residual semantics permit.
3. Fuse EAGLE3 auxiliary hidden-state gather/projection/concatenation into the draft pre-projection path.
4. For 48 GDN layers, prioritize state-update/convolution/norm fusion over adding another generic MLP activation kernel.

## Validation plan

- Compare kernel counts and HBM bytes per target step and per draft step.
- Require numerical parity across GDN and full-attention layers.
- Gate on 128/128 c1/c16 and 512/512 c1 before running the full matrix.
- Do not claim an E2E win until the host HIP IPC state is reset and TP2 service A/B is available again.
