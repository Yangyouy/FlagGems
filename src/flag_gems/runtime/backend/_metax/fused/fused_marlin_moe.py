# SPDX-License-Identifier: Apache-2.0
"""
Fused Marlin MoE — MetaX specialization.

MetaX's Triton compiler cannot lower the in-kernel INT4 unpack + tl.dot pattern
(unrealized_conversion_cast error in MLIR→LLIR). This specialization works around
the issue by pre-dequantizing packed INT4 weights to float16 in PyTorch, then
delegating to the existing fused_experts_impl (pure float16 MoE kernel) which
compiles and runs correctly on MetaX hardware.
"""

from typing import Any, Callable, Optional

import torch

from flag_gems.fused.fused_moe import fused_experts_impl

QUANT_TYPE_UINT4B8 = 0
QUANT_TYPE_UINT8B128 = 1
_QUANT_TYPE_INT4 = {QUANT_TYPE_UINT4B8}
_QUANT_TYPE_INT8 = {QUANT_TYPE_UINT8B128}
_SUPPORTED_QUANT_TYPES = _QUANT_TYPE_INT4 | _QUANT_TYPE_INT8


# ---------- Dequantization ----------

_DEQUANT_CACHE: dict = {}


def _dequantize_int4_packed(
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Dequantize packed uint4b8 weights to float16.

    Args:
        w_packed: (E, N, K//2) uint8, two nibbles per byte (low=even, high=odd)
        scales: (E, N, K//group_size) float16/bf16
        group_size: int

    Returns:
        w_fp: (E, N, K) float16, dequantized weights
    """
    key = (w_packed.data_ptr(), scales.data_ptr(), group_size)
    cached = _DEQUANT_CACHE.get(key)
    if cached is not None:
        return cached

    E, N, K_half = w_packed.shape
    K = K_half * 2
    dtype = scales.dtype

    # Unpack two nibbles per byte
    low = (w_packed & 0xF).to(torch.int8)   # even indices
    high = ((w_packed >> 4) & 0xF).to(torch.int8)  # odd indices

    # Interleave: result[:, :, 0::2] = low, result[:, :, 1::2] = high
    unpacked = torch.empty(E, N, K, device=w_packed.device, dtype=torch.int8)
    unpacked[:, :, 0::2] = low
    unpacked[:, :, 1::2] = high

    # uint4b8 convention: stored value = original + 8, so original = stored - 8
    # Range: stored [0, 15] -> original [-8, 7] (but quantized to [-7, 7] -> [1, 15])
    signed_vals = unpacked.to(dtype) - 8.0

    # Apply per-group scales: scales shape (E, N, K//gs)
    # Expand scales to (E, N, K)
    scales_expanded = scales.unsqueeze(-1).expand(E, N, -1, group_size).reshape(E, N, K)
    w_fp = (signed_vals * scales_expanded).to(dtype)

    _DEQUANT_CACHE[key] = w_fp
    return w_fp


# ---------- Public API ----------


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: Any = None,
    activation_func: Optional[Callable] = None,
    moe_sum: Optional[Callable] = None,
    expert_map: Optional[torch.Tensor] = None,
    input_global_scale1: Optional[torch.Tensor] = None,
    input_global_scale2: Optional[torch.Tensor] = None,
    global_scale1: Optional[torch.Tensor] = None,
    global_scale2: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    intermediate_cache13: Optional[torch.Tensor] = None,
    intermediate_cache2: Optional[torch.Tensor] = None,
    is_k_full: bool = True,
    output: Optional[torch.Tensor] = None,
    input_dtype: Optional[torch.dtype] = None,
    inplace: bool = False,
    clamp_limit: Optional[float] = None,
    group_size: int = 128,
) -> torch.Tensor:
    # --- Input validation (same as NVIDIA version) ---
    if quant_type_id not in _SUPPORTED_QUANT_TYPES:
        raise NotImplementedError(
            f"MVP supports quant_type_id in {_SUPPORTED_QUANT_TYPES}, "
            f"got {quant_type_id}"
        )
    if g_idx1 is not None or g_idx2 is not None:
        raise NotImplementedError("act_order (g_idx) not yet supported in MVP")
    if sort_indices1 is not None or sort_indices2 is not None:
        raise NotImplementedError("act_order (sort_indices) not yet supported in MVP")
    if input_dtype is not None:
        raise NotImplementedError("FP8 / INT8 input quantization not supported")
    if clamp_limit is not None:
        raise NotImplementedError("clamp_limit (GLM-4 swiglu) not supported")
    if input_global_scale1 is not None or input_global_scale2 is not None:
        raise NotImplementedError("input_global_scale not supported in MVP")
    if global_scale1 is not None or global_scale2 is not None:
        raise NotImplementedError("global_scale not supported in MVP")

    activation_str = "silu"
    if activation is not None:
        for attr in ("value", "name"):
            v = getattr(activation, attr, None)
            if isinstance(v, str):
                activation_str = v.lower()
                break
        if isinstance(activation, str):
            activation_str = activation.lower()
    if activation_str != "silu":
        raise NotImplementedError(
            f"MVP only supports SiLU/SwiGLU activation, got {activation_str}"
        )

    if inplace and output is not None:
        raise ValueError("Cannot pass both inplace=True and output")

    # --- MetaX specialization: pre-dequantize INT4 → float16 ---
    # w1: (E, 2*intermediate_size, K//2) packed uint8
    # w2: (E, hidden_size, intermediate_size//2) packed uint8
    # w1_scale: (E, 2*intermediate_size, K//group_size)
    # w2_scale: (E, hidden_size, intermediate_size//group_size)
    w1_fp = _dequantize_int4_packed(w1, w1_scale, group_size)
    w2_fp = _dequantize_int4_packed(w2, w2_scale, group_size)

    # --- Delegate to standard float16 fused_experts_impl ---
    result = fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1_fp,
        w2=w2_fp,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation=activation_str,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=None,
        w2_scale=None,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        w1_bias=bias1,
        w2_bias=bias2,
    )

    if output is not None:
        output.copy_(result)
        return output
    return result


__all__ = ["fused_marlin_moe", "QUANT_TYPE_UINT4B8", "QUANT_TYPE_UINT8B128"]
