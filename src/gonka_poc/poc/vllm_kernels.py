"""Bandwidth-tiled Triton replacements for four vLLM kernels on the prefill PoC
path, each bit-identical to the kernel it replaces (same arithmetic, same
rounding points; verified against vLLM's ops on the PoC shapes):

* the FlashInfer fp8 KV-cache write (``reshape_and_cache_flash``, unit scales);
* the per-token-group fp8 quant of the DeepGEMM path, both the fp32-scale
  (MoE input) and the packed-UE8M0 (dense GEMM input) variants;
* the DeepGEMM MoE glue: the expert token count and ``ep_scatter`` (an index
  pass plus a per-token copy; the order of tokens inside an expert group is
  arbitrary in vLLM's kernel too and the grouped GEMM is row-independent);
* the neox rotary embedding.

They are installed once per process by :mod:`fused` when the fused kernels are
enabled; ``GONKA_POC_VLLM_KERNELS=0`` leaves vLLM's own kernels in place.
Every call with an unsupported argument combination falls through to vLLM.
"""

import logging
import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None

logger = logging.getLogger(__name__)

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


if triton is not None:

    # --- fp8 KV-cache write ------------------------------------------------

    @triton.jit
    def _kv_write_kernel(k_ptr, v_ptr, kc_ptr, vc_ptr, slot_ptr, k_stride, v_stride,
                         c_block_stride, c_page_stride, c_head_stride,
                         BLOCK_SIZE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                         BLOCK: tl.constexpr):
        tok = tl.program_id(0).to(tl.int64)
        slot = tl.load(slot_ptr + tok).to(tl.int64)
        if slot >= 0:
            offs = tl.arange(0, BLOCK)
            m = offs < H * D
            h = offs // D
            d = offs % D
            k = tl.load(k_ptr + tok * k_stride + offs, mask=m).to(tl.float32)
            v = tl.load(v_ptr + tok * v_stride + offs, mask=m).to(tl.float32)
            k = tl.minimum(tl.maximum(k, -448.0), 448.0)
            v = tl.minimum(tl.maximum(v, -448.0), 448.0)
            base = (slot // BLOCK_SIZE) * c_block_stride + (slot % BLOCK_SIZE) * c_page_stride
            coff = base + h * c_head_stride + d
            tl.store(kc_ptr + coff, k.to(tl.float8e4nv).to(tl.int8, bitcast=True), mask=m)
            tl.store(vc_ptr + coff, v.to(tl.float8e4nv).to(tl.int8, bitcast=True), mask=m)

    # --- per-token-group fp8 quant -----------------------------------------

    @triton.jit
    def _group_quant_kernel(x_ptr, q_ptr, s_ptr, M, TMA_M, K: tl.constexpr, NGR: tl.constexpr,
                            NG: tl.constexpr, BM: tl.constexpr, PACKED: tl.constexpr, eps,
                            fp8_max, fp8_min):
        G: tl.constexpr = 128
        pid_m = tl.program_id(0).to(tl.int64)
        pid_g = tl.program_id(1)
        rows = pid_m * BM + tl.arange(0, BM)
        cols = pid_g * NG * G + tl.arange(0, NG * G)
        rmask = rows < M
        x = tl.load(x_ptr + rows[:, None] * K + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        x3 = tl.reshape(x, (BM, NG, G))
        amax = tl.maximum(tl.max(tl.abs(x3), axis=2), eps)
        y_s = tl.maximum(amax / fp8_max, 1e-10)
        # UE8M0: exponent rounded up, exactly as vLLM's exp2(ceil(log2)) and bit trick
        bits = y_s.to(tl.int32, bitcast=True)
        exp_byte = ((bits >> 23) & 0xFF) + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
        inv = ((254 - exp_byte) << 23).to(tl.float32, bitcast=True)
        q = tl.minimum(tl.maximum(x3 * inv[:, :, None], fp8_min), fp8_max)
        tl.store(q_ptr + rows[:, None] * K + cols[None, :],
                 tl.reshape(q, (BM, NG * G)).to(tl.float8e4nv), mask=rmask[:, None])
        g = pid_g * NG + tl.arange(0, NG)
        if PACKED:
            sidx = ((g[None, :] // 4) * TMA_M + rows[:, None]) * 4 + (g[None, :] % 4)
            tl.store(s_ptr + sidx, tl.where(rmask[:, None], exp_byte, 0).to(tl.int8),
                     mask=rows[:, None] < TMA_M)
        else:
            tl.store(s_ptr + rows[:, None] * NGR + g[None, :],
                     (exp_byte << 23).to(tl.float32, bitcast=True), mask=rmask[:, None])

    # --- MoE glue ----------------------------------------------------------

    @triton.jit
    def _count_kernel(topk_ptr, out_ptr, numel, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        e = tl.load(topk_ptr + offs, mask=offs < numel, other=-1)
        tl.atomic_add(out_ptr + e, 1, mask=e >= 0, sem="relaxed")

    @triton.jit
    def _scatter_idx_kernel(topk_ptr, cursor_ptr, idx_ptr, numel, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < numel
        e = tl.load(topk_ptr + offs, mask=m, other=-1)
        d = tl.atomic_add(cursor_ptr + e, 1, mask=e >= 0, sem="relaxed")
        tl.store(idx_ptr + offs, d, mask=m)

    @triton.jit
    def _scatter_copy_kernel(x_ptr, xs_ptr, topk_ptr, idx_ptr, out_ptr, outs_ptr, TOPK: tl.constexpr,
                             H: tl.constexpr, SK: tl.constexpr, SKP: tl.constexpr, CH: tl.constexpr):
        tok = tl.program_id(0).to(tl.int64)
        so = tl.arange(0, SKP)
        sm = so < SK
        s = tl.load(xs_ptr + tok * SK + so, mask=sm)
        for k in tl.static_range(TOPK):
            e = tl.load(topk_ptr + tok * TOPK + k)
            if e >= 0:
                d = tl.load(idx_ptr + tok * TOPK + k).to(tl.int64)
                tl.store(outs_ptr + d * SK + so, s, mask=sm)
        for c in tl.static_range(0, H, CH):
            offs = c + tl.arange(0, CH)
            row = tl.load(x_ptr + tok * H + offs)
            for k in tl.static_range(TOPK):
                e = tl.load(topk_ptr + tok * TOPK + k)
                if e >= 0:
                    d = tl.load(idx_ptr + tok * TOPK + k).to(tl.int64)
                    tl.store(out_ptr + d * H + offs, row)

    # --- rotary ------------------------------------------------------------

    @triton.jit
    def _rotary_kernel(x_ptr, pos_ptr, cache_ptr, tok_stride, HEADS: tl.constexpr, D: tl.constexpr,
                       HALF: tl.constexpr, HB: tl.constexpr):
        tok = tl.program_id(0).to(tl.int64)
        hb = tl.program_id(1)
        pos = tl.load(pos_ptr + tok).to(tl.int64)
        i = tl.arange(0, HALF)
        c = tl.load(cache_ptr + pos * (2 * HALF) + i).to(tl.float32)
        s = tl.load(cache_ptr + pos * (2 * HALF) + HALF + i).to(tl.float32)
        h = hb * HB + tl.arange(0, HB)
        m = (h < HEADS)[:, None] & (i < HALF)[None, :]
        base = x_ptr + tok * tok_stride + h[:, None] * D + i[None, :]
        x = tl.load(base, mask=m).to(tl.float32)
        y = tl.load(base + HALF, mask=m).to(tl.float32)
        ox = x * c[None, :] - y * s[None, :]
        oy = y * c[None, :] + x * s[None, :]
        tl.store(base, ox.to(x_ptr.dtype.element_ty), mask=m)
        tl.store(base + HALF, oy.to(x_ptr.dtype.element_ty), mask=m)


def kv_write(key, value, key_cache, value_cache, slot_mapping):
    _, H, D = key.shape
    _kv_write_kernel[(slot_mapping.shape[0],)](
        key, value, key_cache.view(torch.int8), value_cache.view(torch.int8), slot_mapping,
        key.stride(0), value.stride(0), key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        BLOCK_SIZE=key_cache.shape[1], H=H, D=D, BLOCK=triton.next_power_of_2(H * D), num_warps=4)


def _quant_ok(x, group_size, eps, out_q):
    return (x.dim() == 2 and x.dtype == torch.bfloat16 and x.is_contiguous() and group_size == 128
            and eps == 1e-10 and x.shape[0] > 0 and x.shape[1] % 1024 == 0
            and (out_q is None or (out_q.shape == x.shape and out_q.is_contiguous() and out_q.dtype == FP8)))


def quant_plain(x, out_q=None):
    M, K = x.shape
    q = out_q if out_q is not None else torch.empty((M, K), dtype=FP8, device=x.device)
    s = torch.empty((M, K // 128), dtype=torch.float32, device=x.device)
    _group_quant_kernel[(triton.cdiv(M, 8), K // 1024)](
        x, q, s, M, M, K=K, NGR=K // 128, NG=8, BM=8, PACKED=False, eps=1e-10, fp8_max=FP8_MAX,
        fp8_min=-FP8_MAX, num_warps=4)
    return q, s


def quant_packed(x, out_q=None):
    M, K = x.shape
    ngr = K // 128
    packs = (ngr + 3) // 4
    tma_m = (M + 3) // 4 * 4
    q = out_q if out_q is not None else torch.empty((M, K), dtype=FP8, device=x.device)
    buf = torch.empty(packs * tma_m, dtype=torch.int32, device=x.device)
    _group_quant_kernel[(triton.cdiv(tma_m, 8), ngr // 8)](
        x, q, buf.view(torch.int8), M, tma_m, K=K, NGR=ngr, NG=8, BM=8, PACKED=True, eps=1e-10,
        fp8_max=FP8_MAX, fp8_min=-FP8_MAX, num_warps=8)
    return q, buf.as_strided((M, packs), (1, tma_m))


def count_expert_tokens(topk_ids, num_experts):
    out = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    n = topk_ids.numel()
    _count_kernel[(triton.cdiv(n, 4096),)](topk_ids.reshape(-1), out, n, BLOCK=4096, num_warps=8)
    return out


def rotary(x, positions, cache, head_size, half):
    heads = x.shape[1] // head_size
    hb = 16 if heads >= 16 else 8
    _rotary_kernel[(x.shape[0], triton.cdiv(heads, hb))](
        x, positions, cache, x.stride(0), HEADS=heads, D=head_size, HALF=half, HB=hb, num_warps=2)


def _rotary_ok(x, positions, cache, head_size):
    return x is None or (x.dim() == 2 and x.stride(1) == 1 and x.dtype == cache.dtype
                         and x.shape[0] == positions.shape[0] and x.shape[1] % head_size == 0)


def install_kv_write():
    from vllm.v1.attention.backends import flashinfer as fi

    impl_cls = fi.FlashInferImpl
    orig = impl_cls.do_kv_cache_update

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if (self.kv_sharing_target_layer_name is None and not getattr(self, "is_kvcache_nvfp4", False)
                and self.cache_dtype in ("fp8", "fp8_e4m3") and kv_cache.dtype in (FP8, torch.uint8)
                and key.dtype in (torch.bfloat16, torch.float16) and value.dtype == key.dtype
                and key.stride(2) == 1 and key.stride(1) == key.shape[2] and value.stride(2) == 1
                and value.stride(1) == value.shape[2] and slot_mapping.dtype == torch.int64
                and getattr(layer, "_k_scale_float", None) == 1.0
                and getattr(layer, "_v_scale_float", None) == 1.0):
            k_cache, v_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
            if k_cache.stride(3) == 1:
                kv_write(key, value, k_cache, v_cache, slot_mapping)
                return
        orig(self, layer, key, value, kv_cache, slot_mapping)

    impl_cls.do_kv_cache_update = do_kv_cache_update


def install_group_quant():
    from vllm.model_executor.layers.fused_moe import utils as moe_utils
    from vllm.model_executor.layers.fused_moe.experts import deep_gemm_moe
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    o_plain = fp8_utils.per_token_group_quant_fp8
    o_packed = fp8_utils.per_token_group_quant_fp8_packed_for_deepgemm

    def plain(x, group_size, eps=1e-10, dtype=None, column_major_scales=False, tma_aligned_scales=False,
              out_q=None, use_ue8m0=None):
        if (_quant_ok(x, group_size, eps, out_q) and not column_major_scales and dtype in (None, FP8)
                and (use_ue8m0 if use_ue8m0 is not None else is_deep_gemm_e8m0_used())):
            return quant_plain(x, out_q)
        return o_plain(x, group_size, eps, dtype, column_major_scales, tma_aligned_scales, out_q, use_ue8m0)

    def packed(x, group_size, eps=1e-10, use_ue8m0=None, out_q=None):
        if _quant_ok(x, group_size, eps, out_q) and (use_ue8m0 if use_ue8m0 is not None else is_deep_gemm_e8m0_used()):
            return quant_packed(x, out_q)
        return o_packed(x, group_size, eps, use_ue8m0, out_q)

    fp8_utils.per_token_group_quant_fp8 = plain
    fp8_utils.per_token_group_quant_fp8_packed_for_deepgemm = packed
    moe_utils.per_token_group_quant_fp8 = plain
    deep_gemm_moe.per_token_group_quant_fp8 = plain
    deep_gemm_moe.per_token_group_quant_fp8_packed_for_deepgemm = packed


def install_moe_glue():
    from vllm.model_executor.layers.fused_moe import deep_gemm_utils as dgu
    from vllm.model_executor.layers.fused_moe import utils as moe_utils

    o_count, o_scatter = moe_utils.count_expert_num_tokens, dgu.ep_scatter

    def count_expert_num_tokens(topk_ids, num_local_experts, expert_map):
        if expert_map is None and topk_ids.dtype == torch.int32 and topk_ids.is_contiguous():
            return count_expert_tokens(topk_ids, num_local_experts)
        return o_count(topk_ids, num_local_experts, expert_map)

    def ep_scatter(recv_x, recv_x_scale, recv_topk, num_recv_tokens_per_expert, expert_map, expert_start_loc,
                   output_tensor, output_tensor_scale, m_indices, output_index, align_m=128, block_size=128,
                   pack_ue8m0=False):
        H = recv_x.shape[1]
        if (expert_map is None and not pack_ue8m0 and recv_x.dtype == FP8 and recv_x.is_contiguous()
                and recv_x_scale.dtype == torch.float32 and recv_x_scale.is_contiguous()
                and recv_topk.dtype == torch.int32 and recv_topk.is_contiguous() and output_index.is_contiguous()
                and output_tensor.is_contiguous() and output_tensor_scale.is_contiguous() and H % 1024 == 0
                and recv_x_scale.shape[1] == H // block_size
                and output_tensor_scale.shape[1] == recv_x_scale.shape[1]):
            num_experts = num_recv_tokens_per_expert.shape[0]
            dgu._fwd_kernel_ep_scatter_1[(num_experts,)](
                num_recv_tokens_per_expert, expert_start_loc, m_indices, num_experts=num_experts, num_warps=8,
                BLOCK_E=128, BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts), ALIGN_M=align_m)
            n = recv_topk.numel()
            _scatter_idx_kernel[(triton.cdiv(n, 1024),)](
                recv_topk.view(-1), expert_start_loc, output_index.view(-1), n, BLOCK=1024, num_warps=4)
            SK = recv_x_scale.shape[1]
            _scatter_copy_kernel[(recv_topk.shape[0],)](
                recv_x.view(torch.int8), recv_x_scale, recv_topk, output_index, output_tensor.view(torch.int8),
                output_tensor_scale, TOPK=recv_topk.shape[1], H=H, SK=SK, SKP=triton.next_power_of_2(SK),
                CH=1024, num_warps=4)
            return
        return o_scatter(recv_x, recv_x_scale, recv_topk, num_recv_tokens_per_expert, expert_map,
                         expert_start_loc, output_tensor, output_tensor_scale, m_indices, output_index,
                         align_m, block_size, pack_ue8m0)

    moe_utils.count_expert_num_tokens = count_expert_num_tokens
    dgu.count_expert_num_tokens = count_expert_num_tokens
    dgu.ep_scatter = ep_scatter


def install_rotary():
    from vllm import _custom_ops as ops

    orig = ops.rotary_embedding

    def rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox, rope_dim_offset=0,
                         inverse=False):
        half = cos_sin_cache.shape[-1] // 2
        if (is_neox and rope_dim_offset == 0 and not inverse and cos_sin_cache.dim() == 2
                and cos_sin_cache.is_contiguous() and cos_sin_cache.dtype in (torch.bfloat16, torch.float16)
                and positions.dim() == 1 and positions.dtype == torch.int64 and half & (half - 1) == 0
                and 8 <= half <= head_size // 2 and query is not None
                and _rotary_ok(query, positions, cos_sin_cache, head_size)
                and _rotary_ok(key, positions, cos_sin_cache, head_size)):
            rotary(query, positions, cos_sin_cache, head_size, half)
            if key is not None:
                rotary(key, positions, cos_sin_cache, head_size, half)
            return
        return orig(positions, query, key, head_size, cos_sin_cache, is_neox, rope_dim_offset, inverse)

    ops.rotary_embedding = rotary_embedding


_installed = False


def install() -> None:
    """Patch vLLM once per process; a failure leaves that kernel on vLLM's own."""
    global _installed
    if _installed or triton is None or os.environ.get("GONKA_POC_VLLM_KERNELS", "1") == "0":
        return
    _installed = True
    for name, fn in (("kv_write", install_kv_write), ("group_quant", install_group_quant),
                     ("moe_glue", install_moe_glue), ("rotary", install_rotary)):
        try:
            fn()
            logger.info("PoC: Triton %s kernel installed over vLLM's", name)
        except Exception as e:  # pragma: no cover - vLLM layout changed
            logger.warning("PoC: Triton %s kernel not installed: %s", name, e)
