"""Fused Triton kernels for the PoC transforms: the decode wrappers (router
override, per-layer reflection, embedding synthesis) and the prefill
Householder hook; and MiniMax-M2's q/k RMSNorm in the PoC forward.

Each kernel computes the plugin's torch expression bit for bit: the same
elementwise ops in the same order with the same rounding (no FMA contraction,
IEEE division and sqrt, the libdevice log/sin/cos that PyTorch itself calls),
and the reflection dot product stays ``torch.sum``; the q/k RMSNorm variance
follows torch's own reduction order. They
only remove the launches and memory round trips between those ops.
``GONKA_POC_FUSED=0`` turns them off.
"""

import os
from contextlib import contextmanager

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None


def enabled() -> bool:
    return triton is not None and os.environ.get("GONKA_POC_FUSED", "1") != "0"


def usable(x: torch.Tensor) -> bool:
    return enabled() and x.is_cuda and x.is_contiguous()


if triton is not None:

    @triton.jit
    def _murmur(key, seed):
        # gpu_random._batched_murmur3_32 on int64 lanes holding 32-bit values.
        M: tl.constexpr = 0xFFFFFFFF
        h = seed & M
        k = key & M
        k = (k * 0xCC9E2D51) & M
        k = ((k << 15) | (k >> 17)) & M
        k = (k * 0x1B873593) & M
        h = h ^ k
        h = ((h << 13) | (h >> 19)) & M
        h = (h * 5 + 0xE6546B64) & M
        h = h ^ (h >> 16)
        h = (h * 0x85EBCA6B) & M
        h = h ^ (h >> 13)
        h = (h * 0xC2B2AE35) & M
        h = h ^ (h >> 16)
        return h

    @triton.jit
    def _router_kernel(logits_ptr, out_ptr, base_ptr, step_ptr, mask_ptr,
                       E: tl.constexpr, K: tl.constexpr, LAD: tl.constexpr,
                       BLOCK_E: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_E)
        cm = cols < E
        x = tl.load(logits_ptr + row * E + cols, mask=cm)
        m = tl.load(mask_ptr + row)
        base = tl.load(base_ptr + row).to(tl.int64)
        step = tl.load(step_ptr + row).to(tl.int32).to(tl.int64)
        start = _murmur(step, base) % E
        off = (cols.to(tl.int64) - start + E) % E
        forced = tl.where(off < K, (K - off + LAD).to(tl.float32), -1.0e4)
        forced = forced.to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * E + cols, tl.where(m != 0, forced, x), mask=cm)

    @triton.jit
    def _gather_mul_kernel(x_ptr, t_ptr, rg_ptr, p_ptr, D, LINES_PER_ROW,
                           BLOCK: tl.constexpr):
        line = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        cm = cols < D
        g = tl.load(rg_ptr + line // LINES_PER_ROW)
        xb = tl.load(x_ptr + line * D + cols, mask=cm)
        v = tl.load(t_ptr + g * D + cols, mask=cm).to(xb.dtype).to(tl.float32)
        tl.store(p_ptr + line * D + cols,
                 (xb.to(tl.float32) * v).to(xb.dtype), mask=cm)

    @triton.jit
    def _reflect_tail_kernel(x_ptr, t_ptr, rg_ptr, dot_ptr, mask_ptr, o_ptr, D,
                             LINES_PER_ROW, BLOCK: tl.constexpr):
        line = tl.program_id(0)
        row = line // LINES_PER_ROW
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        cm = cols < D
        g = tl.load(rg_ptr + row)
        m = tl.load(mask_ptr + row)
        xb = tl.load(x_ptr + line * D + cols, mask=cm)
        v = tl.load(t_ptr + g * D + cols, mask=cm).to(xb.dtype).to(tl.float32)
        d2 = (tl.load(dot_ptr + line).to(tl.float32) * 2.0).to(xb.dtype)
        t = (d2.to(tl.float32) * v).to(xb.dtype)
        y = (xb.to(tl.float32) - t.to(tl.float32)).to(xb.dtype)
        tl.store(o_ptr + line * D + cols, tl.where(m != 0, y, xb), mask=cm)

    @triton.jit
    def _householder_tail_kernel(x_ptr, v_ptr, dot_ptr, o_ptr, D,
                                 BLOCK: tl.constexpr):
        line = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        cm = cols < D
        xb = tl.load(x_ptr + line * D + cols, mask=cm)
        v = tl.load(v_ptr + cols, mask=cm).to(tl.float32)
        d2 = (tl.load(dot_ptr + line).to(tl.float32) * 2.0).to(xb.dtype)
        t = (d2.to(tl.float32) * v).to(xb.dtype)
        tl.store(o_ptr + line * D + cols,
                 (xb.to(tl.float32) - t.to(tl.float32)).to(xb.dtype), mask=cm)

    @triton.jit
    def _warp_sum_vec8(acc):
        # torch's bf16 row sum: one warp, 8 consecutive elements per lane (16-byte loads)
        # accumulated in fp32 and folded in order, then shfl_down 16, 8, 4, 2, 1.
        e, o = tl.split(tl.reshape(acc, (32, 4, 2)))
        e04, e26 = tl.split(tl.reshape(e, (32, 2, 2)))
        o15, o37 = tl.split(tl.reshape(o, (32, 2, 2)))
        a0, a4 = tl.split(e04)
        a2, a6 = tl.split(e26)
        a1, a5 = tl.split(o15)
        a3, a7 = tl.split(o37)
        s = ((((((a0 + a1) + a2) + a3) + a4) + a5) + a6) + a7
        s = _half(_half(_half(_half(s, 16), 8), 4), 2)
        sx, sy = tl.split(tl.reshape(s, (1, 2)))
        return tl.sum(sx + sy, axis=0)

    @triton.jit
    def _householder_row_kernel(x_ptr, v_ptr, o_ptr, N: tl.constexpr):
        # apply_householder on one bf16 row: dot = bf16(sum(bf16(x*v))), out = bf16(x - bf16(bf16(2*dot)*v))
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, 256)
        acc = tl.zeros((256,), dtype=tl.float32)
        for t in tl.static_range(N // 256):
            xb = tl.load(x_ptr + row * N + t * 256 + offs)
            vb = tl.load(v_ptr + t * 256 + offs)
            acc = acc + (xb * vb).to(tl.float32)
        dot = _warp_sum_vec8(acc).to(tl.bfloat16)
        d2 = (dot.to(tl.float32) * 2.0).to(tl.bfloat16)
        for t in tl.static_range(N // 256):
            xb = tl.load(x_ptr + row * N + t * 256 + offs)
            vb = tl.load(v_ptr + t * 256 + offs)
            tt = (d2.to(tl.float32) * vb.to(tl.float32)).to(tl.bfloat16)
            tl.store(o_ptr + row * N + t * 256 + offs,
                     (xb.to(tl.float32) - tt.to(tl.float32)).to(tl.bfloat16))

    @triton.jit
    def _embed_kernel(e_ptr, o_ptr, base_ptr, step_ptr, pk_ptr, pe_ptr, mask_ptr,
                      H, NP, MIXA: tl.constexpr, MIXB: tl.constexpr,
                      SALT: tl.constexpr, BLOCK: tl.constexpr):
        M: tl.constexpr = 0xFFFFFFFF
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        cm = cols < H
        e = tl.load(e_ptr + row * H + cols, mask=cm)
        m = tl.load(mask_ptr + row)
        pk = tl.load(pk_ptr + row).to(tl.int64)
        step = tl.load(step_ptr + row).to(tl.int64)
        key = ((pk & M) * MIXA + (step * MIXB + SALT)) & M
        seed = _murmur(key, tl.load(base_ptr + row).to(tl.int64))
        first = cols < NP
        i1 = tl.where(first, cols, cols - NP).to(tl.int64)
        i2 = tl.where(first, cols + NP, cols).to(tl.int64)
        u1 = libdevice.div_rn(_murmur(i1, seed).to(tl.float32), 4294967296.0)
        u2 = libdevice.div_rn(_murmur(i2, seed).to(tl.float32), 4294967296.0)
        u1 = tl.maximum(u1, 1e-10)
        r = libdevice.sqrt_rn(-2.0 * libdevice.log(u1))
        ang = 6.283185307179586 * u2
        z = tl.where(first, r * libdevice.cos(ang), r * libdevice.sin(ang))
        z = z.to(e.dtype)
        pe = tl.load(pe_ptr + row * H + cols, mask=cm).to(e.dtype)
        poc_e = tl.where(pk >= 0, z, pe)
        tl.store(o_ptr + row * H + cols, tl.where(m != 0, poc_e, e), mask=cm)


    @triton.jit
    def _half(v, H: tl.constexpr):
        # lane l adds lane l + H (shfl_down by H): [2H] -> [H]
        x, y = tl.split(tl.trans(tl.reshape(v, (2, H))))
        return x + y

    @triton.jit
    def _norm_row(src, dst, w_ptr, N: tl.constexpr, factor, eps):
        # torch's mean of an fp32 row: one warp, four accumulators per lane over
        # float4 loads strided by 32 lanes, folded in order, then shfl_down 16..1.
        offs = tl.arange(0, 128)
        acc = tl.zeros((128,), dtype=tl.float32)
        for t in tl.static_range(N // 128):
            x = tl.load(src + t * 128 + offs).to(tl.float32)
            acc = acc + x * x
        e, o = tl.split(tl.reshape(acc, (32, 2, 2)))
        a0, a2 = tl.split(e)
        a1, a3 = tl.split(o)
        v = _half(_half(_half(_half(((a0 + a1) + a2) + a3, 16), 8), 4), 2)
        x, y = tl.split(tl.reshape(v, (1, 2)))
        r = libdevice.rsqrt(tl.sum(x + y, axis=0) * factor + eps)
        for t in tl.static_range(N // 128):
            x = tl.load(src + t * 128 + offs).to(tl.float32)
            w = tl.load(w_ptr + t * 128 + offs).to(tl.float32)
            tl.store(dst + t * 128 + offs, ((x * r) * w).to(dst.dtype.element_ty))

    @triton.jit
    def _qk_norm_kernel(q_ptr, k_ptr, q_stride, k_stride, qw_ptr, kw_ptr, q_out, k_out,
                        Q: tl.constexpr, K: tl.constexpr, q_factor, k_factor, q_eps, k_eps):
        row = tl.program_id(0).to(tl.int64)
        _norm_row(q_ptr + row * q_stride, q_out + row * Q, qw_ptr, Q, q_factor, q_eps)
        _norm_row(k_ptr + row * k_stride, k_out + row * K, kw_ptr, K, k_factor, k_eps)


def router_override(logits, base, step, mask, n_experts: int, top_k: int,
                    ladder: int):
    """torch.where(mask[:,None], expert_logits_from_base(base, step, ...)
    .to(logits.dtype), logits) for [n, n_experts] logits."""
    out = torch.empty_like(logits)
    _router_kernel[(logits.shape[0],)](
        logits, out, base, step, mask, n_experts, top_k, ladder,
        triton.next_power_of_2(n_experts))
    return out


def reflect_rows(x, table, row_group, mask):
    """native._reflect_torch(x, table[row_group] as x.dtype, mask) for
    x [n, *pad, D]; the dot product is still x*v summed by torch.sum."""
    n, d = x.shape[0], x.shape[-1]
    lines = x.numel() // d
    grid = (lines, triton.cdiv(d, 1024))
    p = torch.empty_like(x)
    _gather_mul_kernel[grid](x, table, row_group, p, d, lines // n, BLOCK=1024,
                             enable_fp_fusion=False)
    dot = p.sum(-1, keepdim=True)
    out = torch.empty_like(x)
    _reflect_tail_kernel[grid](x, table, row_group, dot, mask, out, d, lines // n,
                               BLOCK=1024, enable_fp_fusion=False)
    return out


def householder(x, v):
    """gpu_random.apply_householder(x, v) with v already in x.dtype."""
    d = x.shape[-1]
    lines = x.numel() // d
    if (x.dtype == torch.bfloat16 and x.is_contiguous() and d % 256 == 0
            and v.dtype == torch.bfloat16 and v.is_contiguous()):
        # every line (DeepSeek-V4 keeps hc_mult copies per row) is reflected along the hidden dim with the same v
        out = torch.empty_like(x)
        _householder_row_kernel[(lines,)](x.view(-1, d), v, out.view(-1, d), N=d, num_warps=1, enable_fp_fusion=False)
        return out
    dot = (x * v).sum(dim=-1, keepdim=True)
    out = torch.empty_like(x)
    _householder_tail_kernel[(lines, triton.cdiv(d, 1024))](
        x, v, dot, out, d, BLOCK=1024, enable_fp_fusion=False)
    return out


def embed_synth(out, base, step, prev_k, embeds, mask, hidden: int, salt: int,
                mix_a: int, mix_b: int):
    """The synth branch of PoCEmbeddingWrapper.forward: decode rows get the
    seeded normal input of their step, prefill rows the pre-filled embed, chat
    rows keep ``out``."""
    o = torch.empty_like(out)
    _embed_kernel[(out.shape[0], triton.cdiv(hidden, 512))](
        out, o, base, step, prev_k, embeds, mask, hidden, (hidden + 1) // 2,
        mix_a, mix_b, salt, BLOCK=512, enable_fp_fusion=False)
    return o


def _f32(x) -> float:
    return float(np.float32(x))


def _minimax_forward_qk(q_norm, k_norm, q, k):
    """MiniMaxText01RMSNormTP.forward_qk (TP 1) in one kernel, bit for bit."""
    if not (q.is_cuda and q.dtype in (torch.bfloat16, torch.float16) and k.dtype == q.dtype
            and q.stride(-1) == 1 and k.stride(-1) == 1 and q.dim() == 2 and k.dim() == 2
            and q.shape[1] % 128 == 0 and k.shape[1] % 128 == 0):
        return _MINIMAX_FORWARD_QK(q_norm, k_norm, q, k)
    m, qn, kn = q.shape[0], q.shape[1], k.shape[1]
    q_out = torch.empty((m, qn), dtype=q.dtype, device=q.device)
    k_out = torch.empty((m, kn), dtype=k.dtype, device=k.device)
    if m:
        # torch's mean factor: float(outputs) / numel, one fp32 division
        _qk_norm_kernel[(m,)](
            q, k, q.stride(0), k.stride(0), q_norm.weight, k_norm.weight, q_out, k_out,
            Q=qn, K=kn, q_factor=_f32(np.float32(m) / np.float32(m * qn)),
            k_factor=_f32(np.float32(m) / np.float32(m * kn)),
            q_eps=_f32(q_norm.variance_epsilon), k_eps=_f32(k_norm.variance_epsilon),
            num_warps=1, enable_fp_fusion=False)
    return q_out, k_out


_MINIMAX_NORM = None
_MINIMAX_FORWARD_QK = None


@contextmanager
def model_ops():
    """Fused stand-ins for model ops the eager PoC forward runs in plain torch:
    MiniMax-M2's q/k RMSNorm at TP 1."""
    global _MINIMAX_NORM, _MINIMAX_FORWARD_QK
    if not enabled():
        yield
        return
    if _MINIMAX_NORM is None:
        try:
            from vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp import (
                MiniMaxText01RMSNormTP as _MINIMAX_NORM)
        except ImportError:
            _MINIMAX_NORM = False
        else:
            _MINIMAX_FORWARD_QK = _MINIMAX_NORM.forward_qk
    if not _MINIMAX_NORM:
        yield
        return
    _MINIMAX_NORM.forward_qk = staticmethod(_minimax_forward_qk)
    try:
        yield
    finally:
        _MINIMAX_NORM.forward_qk = staticmethod(_MINIMAX_FORWARD_QK)


def warmup(state, dtype) -> None:
    """Compile every kernel the wrappers will launch before any cudagraph is
    captured: router per (n_experts, top_k) and logits dtype, reflection on
    [rows, hidden] and [rows, copies, hidden], embedding synthesis."""
    from gonka_poc.poc import decode_random as dr
    dev, h = state.device, state.hidden_size
    mask = torch.ones(2, dtype=torch.bool, device=dev)
    ints = torch.zeros(2, dtype=torch.int64, device=dev)
    for n_exp, top_k in sorted(set(state.router_meta)):
        for ldt in (dtype, torch.float32):
            router_override(torch.zeros(2, n_exp, dtype=ldt, device=dev), ints,
                            ints, mask, n_exp, top_k, dr._ladder_base)
    table = state.table[0] if len(state.table) else torch.zeros(1, h, dtype=dtype, device=dev)
    for shape in ((2, h), (2, 4, h)):
        reflect_rows(torch.zeros(*shape, dtype=dtype, device=dev), table, ints, mask)
    embed_synth(torch.zeros(2, h, dtype=dtype, device=dev), ints, ints, ints,
                torch.zeros(2, h, dtype=state.embeds.dtype, device=dev), mask, h,
                dr._SALT_DECODE_EMBED, dr._MIX_A, dr._MIX_B)
    torch.cuda.synchronize(dev)
