"""Fused Triton kernels for the PoC transforms: the decode wrappers (router
override, per-layer reflection, embedding synthesis) and the prefill
Householder hook.

Each kernel computes the plugin's torch expression bit for bit: the same
elementwise ops in the same order with the same rounding (no FMA contraction,
IEEE division and sqrt, the libdevice log/sin/cos that PyTorch itself calls),
and the one reduction, the reflection dot product, stays ``torch.sum``. They
only remove the launches and memory round trips between those ops.
``GONKA_POC_FUSED=0`` turns them off.
"""

import os

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
