# SPDX-License-Identifier: Apache-2.0
"""
Hygon specialization for fused_marlin_moe.

The default INT4 Triton kernels (bit-shift + tl.dot) produce incorrect results
on Hygon hardware. This specialization pre-dequantizes packed INT4 weights to
float16/bf16 and delegates to fused_experts_impl which uses standard float-type
Triton kernels that work correctly on Hygon.
"""

from typing import Any, Callable, Optional

import torch

from flag_gems.fused.fused_marlin_moe import (
    QUANT_TYPE_UINT4B8,
    QUANT_TYPE_UINT8B128,
    _QUANT_TYPE_INT4,
    _SUPPORTED_QUANT_TYPES,
)
from flag_gems.fused.fused_moe import fused_experts_impl


def _dequant_int4_packed(
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Unpack GPTQ uint4b8 packed weights and dequantize to float16/bf16.

    Args:
        w_packed: (E, N, K//2) uint8 — two 4-bit values per byte
        w_scale:  (E, N, K//group_size) float16/bf16 — per-group scales
        group_size: quantization group size
        dtype: target dtype (float16 or bfloat16)

    Returns:
        w_float: (E, N, K) in target dtype
    """
    E, N, K_half = w_packed.shape
    K = K_half * 2

    low = (w_packed & 0xF).to(torch.int8)
    high = ((w_packed >> 4) & 0xF).to(torch.int8)

    w_unpacked = torch.empty(E, N, K, device=w_packed.device, dtype=torch.int8)
    w_unpacked[:, :, 0::2] = low
    w_unpacked[:, :, 1::2] = high

    w_float = (w_unpacked.to(torch.float32) - 8.0)

    num_groups = K // group_size
    w_float = w_float.reshape(E, N, num_groups, group_size)
    scale_expanded = w_scale.to(torch.float32).unsqueeze(-1)
    w_float = (w_float * scale_expanded).reshape(E, N, K).to(dtype)

    return w_float


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

    use_int4 = quant_type_id in _QUANT_TYPE_INT4
    orig_dtype = hidden_states.dtype

    if use_int4:
        compute_dtype = torch.float16
        w1_float = _dequant_int4_packed(w1, w1_scale, group_size, compute_dtype)
        w2_float = _dequant_int4_packed(w2, w2_scale, group_size, compute_dtype)
        compute_hidden = hidden_states.to(compute_dtype)
        topk_w = topk_weights.to(compute_dtype)
    else:
        w1_float = w1.to(orig_dtype)
        w2_float = w2.to(orig_dtype)
        compute_hidden = hidden_states
        topk_w = topk_weights

    result = fused_experts_impl(
        hidden_states=compute_hidden,
        w1=w1_float,
        w2=w2_float,
        topk_weights=topk_w,
        topk_ids=topk_ids,
        inplace=False,
        activation=activation_str,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )

    if result.dtype != orig_dtype:
        result = result.to(orig_dtype)

    if inplace:
        hidden_states.copy_(result)
        result = hidden_states

    if output is not None:
        output.copy_(result)
        return output
    return result


__all__ = ["fused_marlin_moe", "QUANT_TYPE_UINT4B8", "QUANT_TYPE_UINT8B128"]
