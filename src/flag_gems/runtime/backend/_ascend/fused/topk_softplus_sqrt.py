# SPDX-License-Identifier: Apache-2.0
# Ascend specialization of topk_softplus_sqrt kernel.
#
# Key adaptations for Ascend 910B:
# 1. _fused_topk_kernel: Replace scalar store/load loop pattern with vector buffer
#    approach. The original per-element tl.store in tl.static_range loops causes
#    incorrect results when grid size (num_tokens) >= 128 on Ascend.
# 2. _hash_kernel: Split into two kernels (compute_scores + gather) to avoid UB
#    overflow. The original kernel keeps both BLOCK_E and BLOCK_K vectors alive
#    inside a tl.static_range loop, which exhausts Ascend UB at BLOCK_E >= 64.
# 3. Use num_warps=1 as required by Ascend compiler.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _fused_topk_kernel_ascend(
    gating_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    token_expert_indices_ptr,
    e_score_correction_bias_ptr,
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    renormalize: tl.constexpr,
    routed_scaling_factor,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_E)
    emask = expert_offsets < num_experts

    row_base = pid * num_experts
    x = tl.load(gating_ptr + row_base + expert_offsets, mask=emask, other=0.0).to(
        tl.float32
    )

    # Fused softplus + sqrt
    x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    raw = tl.sqrt(x)

    # Scores for top-k selection (with optional bias)
    if HAS_BIAS:
        bias = tl.load(
            e_score_correction_bias_ptr + expert_offsets, mask=emask, other=0.0
        ).to(tl.float32)
        scores = raw + bias
    else:
        scores = raw
    scores = tl.where(emask, scores, -float("inf"))

    # Use vector buffers for weights and indices (avoid scalar store/load pattern)
    k_offsets = tl.arange(0, BLOCK_K)
    topk_w = tl.zeros([BLOCK_K], dtype=tl.float32)
    topk_idx = tl.zeros([BLOCK_K], dtype=tl.int32)

    weight_sum = 0.0

    for k_idx in tl.static_range(topk):
        max_score = tl.max(scores, axis=0)
        is_max = scores == max_score
        match_priority = tl.where(is_max, BLOCK_E - expert_offsets, 0)
        best_slot = BLOCK_E - tl.max(match_priority, axis=0)
        eidx = best_slot.to(tl.int32)

        if HAS_BIAS:
            # Extract bias at eidx using vector approach
            bias_at_eidx = tl.sum(tl.where(expert_offsets == eidx, bias, 0.0))
            w = max_score - bias_at_eidx
        else:
            w = max_score

        weight_sum += w

        # Accumulate into vector buffers instead of scalar stores
        topk_w = tl.where(k_offsets == k_idx, w, topk_w)
        topk_idx = tl.where(k_offsets == k_idx, eidx, topk_idx)

        # Zero out winner
        scores = tl.where(expert_offsets == eidx, -float("inf"), scores)

    # Apply renormalization + scaling
    if renormalize:
        scale = routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    else:
        scale = routed_scaling_factor

    topk_w = topk_w * scale

    # Single burst store for all outputs
    out_base = pid * topk
    kmask = k_offsets < topk
    tl.store(topk_weights_ptr + out_base + k_offsets, topk_w, mask=kmask)
    tl.store(topk_indices_ptr + out_base + k_offsets, topk_idx, mask=kmask)
    tei = (pid * topk + k_offsets).to(tl.int32)
    tl.store(token_expert_indices_ptr + out_base + k_offsets, tei, mask=kmask)


@triton.jit
def _compute_scores_kernel(
    gating_ptr,
    scores_ptr,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Pre-compute softplus+sqrt scores per row. Separate kernel to avoid
    UB overflow when combining with the gather loop."""
    pid = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_E)
    emask = expert_offsets < num_experts
    row_base = pid * num_experts
    x = tl.load(gating_ptr + row_base + expert_offsets, mask=emask, other=0.0).to(
        tl.float32
    )
    x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    x = tl.sqrt(x)
    tl.store(scores_ptr + row_base + expert_offsets, x, mask=emask)


@triton.jit
def _hash_gather_kernel(
    scores_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    token_expert_indices_ptr,
    input_tokens_ptr,
    hash_indices_table_ptr,
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    renormalize: tl.constexpr,
    routed_scaling_factor,
    BLOCK_K: tl.constexpr,
):
    """Hash mode gather kernel: uses scalar loads from pre-computed scores
    to avoid keeping BLOCK_E vectors alive inside the topk loop.
    Uses vector buffers (BLOCK_K) to avoid scalar store pattern."""
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    token_id = tl.load(input_tokens_ptr + pid)
    table_base = token_id * topk
    row_base = pid * num_experts

    k_offsets = tl.arange(0, BLOCK_K)
    kmask = k_offsets < topk
    topk_w = tl.zeros([BLOCK_K], dtype=tl.float32)
    topk_idx = tl.zeros([BLOCK_K], dtype=tl.int32)

    weight_sum = 0.0

    for k_idx in tl.static_range(topk):
        eidx = tl.load(hash_indices_table_ptr + table_base + k_idx)
        # Direct scalar load from pre-computed scores — no BLOCK_E vector needed
        w = tl.load(scores_ptr + row_base + eidx)
        weight_sum += w
        # Accumulate into vector buffers
        topk_w = tl.where(k_offsets == k_idx, w, topk_w)
        topk_idx = tl.where(k_offsets == k_idx, eidx, topk_idx)

    # Apply renorm + scale
    if renormalize:
        scale = routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    else:
        scale = routed_scaling_factor

    topk_w = topk_w * scale

    # Single burst store
    out_base = pid * topk
    tl.store(topk_weights_ptr + out_base + k_offsets, topk_w, mask=kmask)
    tl.store(topk_indices_ptr + out_base + k_offsets, topk_idx, mask=kmask)
    tei = (pid * topk + k_offsets).to(tl.int32)
    tl.store(token_expert_indices_ptr + out_base + k_offsets, tei, mask=kmask)


def topk_softplus_sqrt(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize,
    routed_scaling_factor,
    correction_bias=None,
    input_ids=None,
    tid2eid=None,
):
    """Fused topk + softplus + sqrt kernel for MoE gating (Ascend specialization).

    Interface aligned with vLLM CUDA operator.
    """
    logger.debug("GEMS_ASCEND TOPK_SOFTPLUS_SQRT")
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[1]

    if num_tokens == 0:
        return

    BLOCK_E = triton.next_power_of_2(num_experts)
    BLOCK_K = triton.next_power_of_2(topk)

    if input_ids is not None and tid2eid is not None:
        # Hash mode: two-kernel approach to avoid UB overflow
        # Kernel 1: compute softplus+sqrt scores
        scores = torch.empty(
            (num_tokens, num_experts), dtype=torch.float32, device=gating_output.device
        )
        with torch_device_fn.device(gating_output.device):
            _compute_scores_kernel[(num_tokens,)](
                gating_output,
                scores,
                num_experts=num_experts,
                BLOCK_E=BLOCK_E,
                num_warps=1,
                num_stages=1,
            )

            # Kernel 2: gather weights using scalar loads
            _hash_gather_kernel[(num_tokens,)](
                scores,
                topk_weights,
                topk_indices,
                token_expert_indices,
                input_ids,
                tid2eid,
                num_tokens=num_tokens,
                num_experts=num_experts,
                topk=topk,
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
                BLOCK_K=BLOCK_K,
                num_warps=1,
                num_stages=1,
            )
        return

    # Standard mode: single fused kernel with vector buffers
    grid = (num_tokens,)
    with torch_device_fn.device(gating_output.device):
        _fused_topk_kernel_ascend[grid](
            gating_output,
            topk_weights,
            topk_indices,
            token_expert_indices,
            correction_bias if correction_bias is not None else gating_output,
            num_tokens=num_tokens,
            num_experts=num_experts,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            HAS_BIAS=correction_bias is not None,
            BLOCK_E=BLOCK_E,
            BLOCK_K=BLOCK_K,
            num_warps=1,
            num_stages=1,
        )
