import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend import heuristics_config_utils as _hcu
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
)
@triton.heuristics(_hcu.HEURISTICS_CONFIGS["mm"])
@triton.jit
def router_gemm_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """GEMM kernel for MoE router gate: bf16 inputs -> fp32 output."""
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    ram = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rbn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A, mask=(ram < M)[:, None], other=0.0)
            b = tl.load(B, mask=(rbn < N)[None, :], other=0.0)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(
                A,
                mask=(rk[None, :] < k_remaining) & (ram < M)[:, None],
                other=0.0,
            )
            b = tl.load(
                B,
                mask=(rk[:, None] < k_remaining) & (rbn < N)[None, :],
                other=0.0,
            )
        if a.dtype != b.dtype:
            a = a.to(tl.bfloat16)
            b = b.to(tl.bfloat16)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    # Store fp32 result directly (no downcast — router gate needs fp32 output)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


def router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """bf16 x bf16 -> fp32 GEMM for MoE router gate. weight shape: (N, K)."""
    logger.debug("GEMS_ASCEND ROUTER_GEMM")
    if x.stride(0) > 1 and x.stride(1) > 1:
        x = x.contiguous()
    M, K = x.shape
    N = weight.shape[0]
    # Transpose weight: (N, K) -> (K, N)
    b = weight.t().contiguous()
    # Output is fp32
    c = torch.empty((M, N), device=x.device, dtype=torch.float32)
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META.get("SPLIT_K", 1),
    )
    with torch_device_fn.device(x.device):
        router_gemm_kernel[grid](
            x,
            b,
            c,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            GROUP_M=8,
        )
    return c
