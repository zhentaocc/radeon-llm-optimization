# GDN fused decode for gfx1100

Purpose: enable a fused Qwen3.8 GDN decode path on RDNA instead of the current Triton fallback.

Require numerical parity, explicit backend-selection evidence, kernel trace reduction, and end-to-end gain.

