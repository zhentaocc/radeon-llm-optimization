# Fused GDN decode for gfx1100

Status: source/build gap isolated; implementation requires a HIP kernel port and cannot be completed in the current binary-only W7900 image.

Qwen3.8-27B has 48 linear-attention GDN layers and 16 full-attention layers. On W7900, vLLM logs `GDN decode kernel: triton`; the fused MTP path is unavailable because `torch.ops._C.fused_gdn_decode_post_conv_mtp` is not built.

The upstream kernel exists at `csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu`, but CMake adds it only inside the CUDA build and only for CUDA 13-compatible architectures. The Python eligibility check also describes CUDA compute capability 8.0+ and selects `gdn_decode_kernel == "cuda"`.

Required work:

1. port the fused post-convolution state update, recurrent accumulation, gated RMSNorm, and MTP state-index handling to HIP;
2. replace warp-size assumptions with wave32/wave64-safe primitives for gfx1100;
3. add ROCm CMake registration and operator binding;
4. add BF16/FP32 recurrent-state parity tests for N=1/2/3/5 and GQA ratios 1/2/3/4/8;
5. compare against the Triton path on 128/128 c1/c16 and 512/512 c1 before a full matrix.

This remains a high-value P1 because the path runs in 48/64 target layers and every draft verification step, but it is not a configuration-only change.
