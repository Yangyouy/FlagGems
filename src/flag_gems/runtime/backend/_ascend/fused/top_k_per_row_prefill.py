"""Ascend-specialized top_k_per_row_prefill for DeepSeek V4 sparse attention.

Adapted from the default NVIDIA implementation with two Ascend workarounds:
1. Reduced BLOCK_SIZE (2048 vs 8192) to avoid Ascend 910B UB overflow.
2. Read-modify-write masking pattern (tl.load + tl.where + tl.store) instead
   of masked tl.store with scalar -inf, which silently drops writes on Ascend.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mask_invalid_kernel(
    logits_ptr,
    row_starts_ptr,
    row_ends_ptr,
    stride0,
    BLOCK_SIZE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_per_row = tl.cdiv(VOCAB_SIZE, BLOCK_SIZE)
    row_id = pid // num_blocks_per_row
    block_id = pid % num_blocks_per_row

    start = tl.load(row_starts_ptr + row_id)
    end = tl.load(row_ends_ptr + row_id)

    if start == 0 and end >= VOCAB_SIZE:
        return

    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    vocab_mask = offs < VOCAB_SIZE
    addr = logits_ptr + row_id * stride0 + offs

    vals = tl.load(addr, mask=vocab_mask, other=0.0)
    out_of_range = (offs < start) | (offs >= end)
    new_vals = tl.where(out_of_range, float("-inf"), vals)
    tl.store(addr, new_vals, mask=vocab_mask)


@triton.jit
def _fused_postprocess_kernel(
    src_ptr,
    dst_ptr,
    row_starts_ptr,
    num_rows: tl.constexpr,
    top_k: tl.constexpr,
    src_stride0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= num_rows:
        return

    row_start = tl.load(row_starts_ptr + row_id)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < top_k

    src_idx = row_id * src_stride0 + offs
    src_vals = tl.load(src_ptr + src_idx, mask=mask, other=0)

    dst_vals = (src_vals - row_start).to(tl.int32)

    dst_idx = row_id * top_k + offs
    tl.store(dst_ptr + dst_idx, dst_vals, mask=mask)


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    vocab_size = logits.shape[1]

    if top_k > vocab_size:
        raise ValueError(f"top_k ({top_k}) must not exceed vocab_size ({vocab_size})")

    MASK_BS = 2048
    num_mask_blocks = (vocab_size + MASK_BS - 1) // MASK_BS
    _mask_invalid_kernel[(num_rows * num_mask_blocks,)](
        logits,
        row_starts,
        row_ends,
        stride0,
        BLOCK_SIZE=MASK_BS,
        VOCAB_SIZE=vocab_size,
        num_warps=1,
    )

    POSTPROC_BLOCK = triton.next_power_of_2(top_k)

    if num_rows == 1:
        sorted_idx = torch.argsort(logits, dim=1, descending=True, stable=False)
        _fused_postprocess_kernel[(1,)](
            sorted_idx,
            indices,
            row_starts,
            num_rows=1,
            top_k=top_k,
            src_stride0=vocab_size,
            BLOCK_SIZE=POSTPROC_BLOCK,
            num_warps=1,
        )
    else:
        _, top_idx = torch.topk(logits, top_k, dim=1, largest=True, sorted=False)
        _fused_postprocess_kernel[(num_rows,)](
            top_idx,
            indices,
            row_starts,
            num_rows=num_rows,
            top_k=top_k,
            src_stride0=top_k,
            BLOCK_SIZE=POSTPROC_BLOCK,
            num_warps=1,
        )
