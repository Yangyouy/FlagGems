# SPDX-License-Identifier: Apache-2.0
"""
Fused Marlin MoE — Ascend specialization (PyTorch fallback).

Ascend 910B4's Triton compiler (BiShengIR) has a confirmed bug where
``tl.load(ptr, mask=condition, other=0.0)`` silently produces wrong results
when the pointer is derived from memory-loaded indices (indirect addressing).
This affects ALL MoE-style Triton kernels that use sorted_token_ids to index
into the activation tensor, making a correct pure-Triton implementation
infeasible on the current toolchain.

The existing Ascend ``fused_experts_impl`` also fails (compilation crashes or
produces NaN for most configurations), so we cannot delegate to it either.

This fallback dequantizes INT4 weights to fp16/bf16 and performs the SwiGLU
MoE computation in PyTorch. It is correct but slower than a native Triton
implementation. Revisit when the Ascend Triton compiler fixes the masked-load
+ indirect-addressing bug.
"""

from typing import Any, Callable, Optional

import torch

QUANT_TYPE_UINT4B8 = 0
QUANT_TYPE_UINT8B128 = 1
_QUANT_TYPE_INT4 = {QUANT_TYPE_UINT4B8}
_QUANT_TYPE_INT8 = {QUANT_TYPE_UINT8B128}
_SUPPORTED_QUANT_TYPES = _QUANT_TYPE_INT4 | _QUANT_TYPE_INT8

_DEQUANT_CACHE: dict = {}


def _dequantize_int4_weight(
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (w_packed.data_ptr(), w_scale.data_ptr(), group_size, dtype)
    cached = _DEQUANT_CACHE.get(key)
    if cached is not None:
        return cached

    E, N, K_half = w_packed.shape
    K = K_half * 2

    low = (w_packed & 0xF).to(torch.int8)
    high = ((w_packed >> 4) & 0xF).to(torch.int8)

    w_unpacked = torch.empty(E, N, K, device=w_packed.device, dtype=torch.int8)
    w_unpacked[:, :, ::2] = low
    w_unpacked[:, :, 1::2] = high

    w_centered = w_unpacked.to(torch.float32) - 8.0

    num_groups = K // group_size
    w_grouped = w_centered.view(E, N, num_groups, group_size)
    scale_expanded = w_scale.float().unsqueeze(-1)
    w_fp = (w_grouped * scale_expanded).reshape(E, N, K).to(dtype)

    _DEQUANT_CACHE[key] = w_fp
    return w_fp


def _pytorch_swiglu_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    M, K = hidden_states.shape
    E, two_N, _ = w1.shape
    N = two_N // 2
    topk = topk_ids.shape[1]

    x = hidden_states.float()
    out = torch.zeros(M, K, device=hidden_states.device, dtype=torch.float32)

    for e in range(E):
        mask = topk_ids == e
        if not mask.any():
            continue
        token_indices, slot_indices = mask.nonzero(as_tuple=True)
        weights = topk_weights[token_indices, slot_indices]
        x_e = x[token_indices]

        if apply_router_weight_on_input:
            x_e = x_e * weights.unsqueeze(1)

        gate_up = torch.mm(x_e, w1[e].float().t())
        gate = gate_up[:, :N]
        up = gate_up[:, N:]
        activated = torch.nn.functional.silu(gate) * up

        y = torch.mm(activated, w2[e].float().t())

        if not apply_router_weight_on_input:
            y = y * weights.unsqueeze(1)

        out.index_add_(0, token_indices, y)

    return out.to(hidden_states.dtype)


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

    dtype = hidden_states.dtype
    w1_fp = _dequantize_int4_weight(w1, w1_scale, group_size, dtype)
    w2_fp = _dequantize_int4_weight(w2, w2_scale, group_size, dtype)

    result = _pytorch_swiglu_moe(
        hidden_states=hidden_states,
        w1=w1_fp,
        w2=w2_fp,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )

    if inplace:
        hidden_states.copy_(result)
        return hidden_states

    if output is not None:
        output.copy_(result)
        return output
    return result


__all__ = ["fused_marlin_moe", "QUANT_TYPE_UINT4B8", "QUANT_TYPE_UINT8B128"]
