"""PoC model runner for the vLLM V1 architecture (supported minors per
``gonka_poc._compat``); version-specific surfaces live in the compat shim.

Full model forward pass with proper V1 attention metadata.
Uses actual KV cache blocks for attention to work correctly.
Batched forward pass — processes all nonces in a single forward call.

Private-API touchpoint policy
-----------------------------
All ``vllm.v1.*`` private surfaces are routed through the version-dispatched
compat shim. The import binds the resolver function and each consumer calls
it to obtain the actual module: ``compat = _compat_current();
compat.build_common_attention_metadata(...)``. Touchpoints:
    * ``CommonAttentionMetadata`` construction
    * per-group ``AttentionMetadata`` construction (iteration over
      ``model_runner.attn_groups``, per-group ``kv_cache_spec.block_size``
      resolution, ``builder.build``) via ``build_attn_metadata_per_group``
    * ``model_runner.kv_caches`` access

The following vLLM imports REMAIN at module scope because they are public
(re-exported via the package root):
    * ``vllm.distributed.get_pp_group`` / ``get_tp_group``
      (pinned by ``tests/contract/test_api_surface.py::test_distributed_groups_present``)
    * ``vllm.distributed.communication_op.broadcast_tensor_dict``
      (pinned by ``::test_communication_op_broadcast``)
    * ``vllm.forward_context.set_forward_context``
      (pinned by ``::test_forward_context_set``)
    * ``vllm.sequence.IntermediateTensors``
      (pinned by ``::test_intermediate_tensors``)
    * ``vllm.logger.init_logger``

If a future minor reshuffles any of these into private namespaces, move
the import into the compat shim and add a contract-test pin.
"""
import math
import torch
import torch.distributed as dist
import numpy as np
from typing import List, Optional, Dict, Any

from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.forward_context import set_forward_context
from vllm.sequence import IntermediateTensors
from vllm.logger import init_logger

from gonka_poc._compat import current as _compat_current

from .gpu_random import (
    generate_inputs,
    generate_inputs_concat_murmur,
    derive_pseudo_input_ids,
    random_pick_indices,
    apply_haar_rotation,
)
from .layer_hooks import LayerHouseholderHook, poc_forward_context

logger = init_logger(__name__)

DEFAULT_K_DIM = 12

# NOTE: attention metadata must NOT be cached across PoC calls.
# The metadata builder's internal state (workspace buffers, page-table
# references) is mutated by every inference engine step.  Reusing a
# stale metadata object causes the attention backend to write only a
# fraction of the expected KV entries, producing all-NaN hidden states.
# The cost of rebuilding is <1 ms per call (vs ~15 ms for the model
# forward), so the overhead is negligible.


def _ensure_layer_hooks(worker, block_hash, hidden_size):
    """Ensure layer hooks are installed for the given block_hash."""
    model = worker.model_runner.model
    device = worker.device
    existing_hook = getattr(worker, "_poc_layer_hooks", None)
    if existing_hook is not None:
        if existing_hook.block_hash == block_hash:
            return
        existing_hook.detach()
    worker._poc_layer_hooks = LayerHouseholderHook(
        model, block_hash, device, hidden_size)


def _borrowed_layout(
    batch_size: int,
    seq_len: int,
    g_block: int,
    m_block: int,
    borrowed_block_ids: List[int],
    stripe: int,
    device,
):
    """slot_mapping + block_table over a LEASED set of pool blocks.

    ``borrowed_block_ids`` are pool-unit (manager) block ids leased from the
    ONE engine-wide BlockPool; ``stripe`` is the per-sequence allotment
    (``max_g ceil(seq_len/manager_block_size_g)`` — computed by the engine
    core with full group knowledge). Sequence ``i`` uses the first
    ``ceil(seq_len/m_block)`` ids of its stripe
    ``borrowed_block_ids[i*stripe : (i+1)*stripe]``.

    Unit conversion: slot/table math runs in KERNEL units (``g_block`` from
    ``builder.kv_cache_spec`` — possibly a kernel split of the manager
    size). A pool block ``b`` covers kernel blocks ``b*r .. b*r+r-1``
    (``r = m_block//g_block``), i.e. the contiguous slot range
    ``[b*m_block, (b+1)*m_block)`` — so the slot for token ``t`` of
    sequence ``i`` is ``L[i][t//m_block]*m_block + t%m_block``, and the
    kernel block table entry ``k`` is ``L[i][k//r]*r + k%r``. Using a raw
    pool id as a kernel id WITHOUT the ×r expansion would address bytes
    inside pool block ``id//r`` — unleased, possibly live inference KV.

    Fail-loud guards (ValueError → RPC error → the chunk fails visibly,
    never a silent mis-write): non-divisible split, stripe too small,
    lease too small.
    """
    if m_block % g_block != 0:
        raise ValueError(
            f"PoC borrowed layout: manager block {m_block} is not a "
            f"multiple of kernel block {g_block}")
    r = m_block // g_block
    bps = math.ceil(seq_len / m_block)
    if bps > stripe:
        raise ValueError(
            f"PoC lease stripe {stripe} too small: group with manager "
            f"block {m_block} needs {bps} blocks/seq at seq_len {seq_len}")
    if batch_size * stripe > len(borrowed_block_ids):
        raise ValueError(
            f"PoC lease has {len(borrowed_block_ids)} blocks, needs "
            f"{batch_size * stripe} ({batch_size} seqs × stripe {stripe})")

    ids = torch.tensor(
        borrowed_block_ids[:batch_size * stripe],
        dtype=torch.long, device=device).view(batch_size, stripe)
    seq_blocks = ids[:, :bps]  # [batch, bps] pool-unit ids

    j = torch.arange(seq_len, dtype=torch.long, device=device) // m_block
    off = torch.arange(seq_len, dtype=torch.long, device=device) % m_block
    slot_mapping = (seq_blocks[:, j] * m_block + off).reshape(-1)

    kernel_ids = (
        seq_blocks.unsqueeze(-1) * r
        + torch.arange(r, dtype=torch.long, device=device)
    ).view(batch_size, bps * r)
    block_table = kernel_ids.to(torch.int32)
    return slot_mapping, block_table


def _inplace_layout(batch_size, seq_len, g_block, device):
    """Legacy in-place slot_mapping + block_table over blocks ``1..N``.

    Block 0 is never used: it is vLLM's null block (``BlockPool.null_block``,
    ``NULL_BLOCK_ID == 0``), the placeholder the engine hands out as padding
    and expects to stay all-zero. Mamba-style state kernels go further and
    treat state index 0 as "padding, skip this sequence" (``causal_conv1d``
    returns without computing it), so a sequence placed on block 0 comes out
    of every linear-attention layer with whatever the output buffer held.

    slot = (1 + seq_idx*blocks_per_seq + t//g_block)*g_block + t%g_block
         = g_block + seq_idx*padded_len + t   (contiguous per sequence,
    padded to a block multiple), so the mapping still vectorizes to two
    aranges plus a constant offset.
    """
    blocks_per_seq = math.ceil(seq_len / g_block)
    padded = blocks_per_seq * g_block
    base = (torch.arange(batch_size, dtype=torch.long, device=device)
            * padded + g_block).repeat_interleave(seq_len)
    slot_mapping = base + torch.arange(
        seq_len, dtype=torch.long, device=device).repeat(batch_size)
    block_table = torch.arange(
        1, batch_size * blocks_per_seq + 1, dtype=torch.int32, device=device
    ).view(batch_size, blocks_per_seq)
    return slot_mapping, block_table


def _create_v1_attn_metadata(batch_size, seq_len, device, worker, positions,
                             borrowed_block_ids=None, borrowed_stripe=None):
    """Create attention metadata, built independently for every attention group.

    Models may register KV cache groups with DIFFERENT block sizes (e.g.
    DeepSeek-V4: sparse MLA and indexer use ``cache_config.block_size``,
    the SWA compressor uses its own ``block_size`` — 8 at compress_ratio=128).
    Sharing one slot_mapping / block_table built for the main group hands
    out-of-range slot ids to the other pools: OOB writes cause an illegal
    memory access on sm_90 and silent memory corruption elsewhere. The
    layout is therefore computed per group from that group's
    ``kv_cache_spec.block_size``. For single-group models this reduces to
    exactly the previous behaviour.

    Two block sources:
      * ``borrowed_block_ids is None`` — legacy in-place layout over blocks
        ``1..N`` (mining and the abort-based fallback). Block ids only enter
        the address translation, so this is bit-identical to the old
        ``0..N`` layout for every sequence except the one that used to sit on
        the null block — see :func:`_inplace_layout`.
      * lease (``borrowed_block_ids`` + ``borrowed_stripe`` from
        ``gonka_poc_borrow_blocks``) — validation runs on pool blocks that
        are provably disjoint from live inference; see
        :func:`_borrowed_layout`. Physical block ids enter ONLY the address
        translation (scatter targets / gather tables), never the attention
        math, so artifacts are invariant to block choice.

    ``positions`` is the shared per-token position tensor (also passed to the
    model forward); DeepSeek-V4's C128A metadata builder requires it, every
    other v0.23 backend ignores ``cm.positions``.
    """
    compat = _compat_current()
    total_tokens = batch_size * seq_len

    query_start_loc_gpu = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seq_len)
    query_start_loc_cpu = (
        torch.arange(batch_size + 1, dtype=torch.int32, device="cpu") * seq_len)
    seq_lens_gpu = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    seq_lens_cpu = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cpu")
    num_computed_cpu = torch.zeros(batch_size, dtype=torch.int32, device="cpu")

    def _layout(g_block, m_block):
        if borrowed_block_ids is not None:
            return _borrowed_layout(
                batch_size, seq_len, g_block, m_block,
                borrowed_block_ids, int(borrowed_stripe or 0), device)
        return _inplace_layout(batch_size, seq_len, g_block, device)

    layouts = {}

    def _layout_for_block_size(g_block, m_block):
        key = (g_block, m_block)
        if key not in layouts:
            layouts[key] = _layout(g_block, m_block)
        return layouts[key]

    def _common_metadata_for_layout(slot_mapping, block_table):
        return compat.build_common_attention_metadata(
            positions=positions,
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens_gpu,
            num_reqs=batch_size,
            num_actual_tokens=total_tokens,
            max_query_len=seq_len,
            max_seq_len=seq_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=True,
            _seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            _num_computed_tokens_cpu=num_computed_cpu,
        )

    return compat.build_attn_metadata_per_group(
        worker.model_runner,
        layout_for_block_size=_layout_for_block_size,
        common_metadata_for_layout=_common_metadata_for_layout,
    )


def _generate_poc_input_ids(block_hash, public_key, nonces, seq_len, worker, device):
    """Deterministic pseudo token ids for token-id-dependent architectures.

    DeepSeek-V4's hash-MoE layers route experts via ``tid2eid[input_ids]``
    and hard-fail on ``input_ids=None``. PoC has no real tokens, so ids are
    derived via :func:`gpu_random.derive_pseudo_input_ids` (see its docstring
    for the consensus-critical derivation convention).
    Gated by model_type so every other architecture keeps ``input_ids=None``.
    """
    hf_cfg = getattr(
        getattr(worker.model_runner, "model_config", None), "hf_config", None)
    if getattr(hf_cfg, "model_type", None) != "deepseek_v4":
        return None
    return derive_pseudo_input_ids(
        block_hash, public_key, nonces, seq_len, int(hf_cfg.vocab_size), device)


def _select_poc_kv_scratch(
    kv_caches: list,
    dtype: torch.dtype,
    needed_elems: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
) -> Optional[torch.Tensor]:
    """Return a no-copy scratch view into KV cache memory, if safe.

    KV cache storage may use packed dtypes (e.g. ``uint8`` for FP8) or
    backend-specific non-contiguous layouts. Only reuse memory that already
    matches model embedding dtype and is contiguous so ``view(-1)`` does not
    allocate a copy.

    CONSENSUS-CRITICAL: on configs where this selects a tensor (bf16-KV
    models) the fleet's artifacts depend on the resulting deterministic
    layer-0 K/V-over-residual overwrite — the selection criteria must not
    change (ADR-0015, Decision 5). Used ONLY on the lease-None path; the
    borrow probe (``execute_poc_borrow_compat``) disables leasing wherever
    this could select.
    """
    for kv in kv_caches:
        if kv.dtype != dtype:
            continue
        if not kv.is_contiguous():
            continue
        if kv.numel() < needed_elems:
            continue
        return kv.view(-1)[:needed_elems].view(batch_size, seq_len, hidden_size)
    return None


@torch.inference_mode()
def execute_poc_forward(
    worker,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    hidden_size: int,
    k_dim: int = DEFAULT_K_DIM,
    poc_stronger_rng: bool = False,
    borrowed_block_ids: Optional[List[int]] = None,
    borrowed_stripe: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Execute batched PoC forward pass on a V1 worker.

    Processes all nonces in a single forward call for maximum throughput.
    """
    device = worker.device
    dtype = worker.model_config.dtype
    model = worker.model_runner.model
    vllm_config = worker.vllm_config
    batch_size = len(nonces)

    tp_group = get_tp_group()
    is_tp_driver = tp_group.rank_in_group == 0

    # TP SYNC
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
        if is_tp_driver:
            broadcast_tensor_dict({
                "poc_go": True,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "nonces": nonces,
                "k_dim": k_dim,
                "poc_stronger_rng": poc_stronger_rng,
                # Lease travels in the broadcast so every TP rank builds the
                # SAME layout even if collective_rpc arg delivery ever skews
                # (belt-and-braces, mirrors the upstream port).
                "has_borrowed": borrowed_block_ids is not None,
                "borrowed_block_ids": borrowed_block_ids or [],
                "borrowed_stripe": int(borrowed_stripe or 0),
            }, src=0)
        else:
            broadcast_data = broadcast_tensor_dict(src=0)
            seq_len = int(broadcast_data["seq_len"])
            hidden_size = int(broadcast_data["hidden_size"])
            nonces = list(broadcast_data["nonces"])
            k_dim = int(broadcast_data["k_dim"])
            batch_size = len(nonces)
            poc_stronger_rng = bool(broadcast_data["poc_stronger_rng"])
            if broadcast_data.get("has_borrowed"):
                borrowed_block_ids = list(broadcast_data["borrowed_block_ids"])
                borrowed_stripe = int(broadcast_data["borrowed_stripe"])
            else:
                borrowed_block_ids = None
                borrowed_stripe = None

    pp_group = get_pp_group()

    # Pre-forward sync
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
    torch.cuda.synchronize()

    _ensure_layer_hooks(worker, block_hash, hidden_size)

    # Per-token positions: shared by the model forward and (for architectures
    # that need it, e.g. DeepSeek-V4 C128A) the attention metadata.
    positions = torch.arange(
        seq_len, dtype=torch.int64, device=device).repeat(batch_size)

    attn_metadata, slot_mapping_dict = _create_v1_attn_metadata(
        batch_size, seq_len, device, worker, positions,
        borrowed_block_ids=borrowed_block_ids,
        borrowed_stripe=borrowed_stripe,
    )

    poc_input_ids = _generate_poc_input_ids(
        block_hash, public_key, nonces, seq_len, worker, device)

    # Generate inputs for all nonces at once
    intermediate_tensors = None
    inputs_embeds = None

    if pp_group.is_first_rank:
        # CONSENSUS-CRITICAL fork in the embeds source (ADR-0015, Decision 5):
        #
        # * lease is None (mining + legacy validation): reproduce the DEPLOYED
        #   derivation path exactly, INCLUDING the KV-scratch reuse. On configs where
        #   the scratch is selected (KV dtype == model dtype, contiguous —
        #   i.e. bf16-KV models), layer-0 K/V writes overlap the scratch view
        #   that also backs the residual, and the fleet's artifacts already
        #   depend on that deterministic self-overwrite. Removing it here
        #   would change bits under every deployed validator.
        # * lease present (borrowed validation): ALWAYS a fresh buffer. The
        #   scratch would write outside the lease; borrowing is only enabled
        #   on configs where the scratch is never selected (see
        #   execute_poc_borrow_compat + the reservation-side probe), so the
        #   fresh path is bit-identical to what those configs already run.
        # NOTE: the deployed scratch path also fires with poc_stronger_rng=True
        # and then fills with the LEGACY per-nonce RNG (a known quirk that
        # diverges from the fresh path's concat-murmur). Agreement with the
        # fleet requires reproducing the quirk, not fixing it: a node deriving
        # inputs differently lands beyond the validation threshold on every
        # nonce -- not noise the statistical test would absorb.
        kv_scratch = None
        if borrowed_block_ids is None:
            compat = _compat_current()
            try:
                kv_caches = compat.get_kv_cache_pool(worker.model_runner)
            except RuntimeError:
                kv_caches = []
            kv_scratch = _select_poc_kv_scratch(
                kv_caches, dtype, batch_size * seq_len * hidden_size,
                batch_size, seq_len, hidden_size,
            )
        if kv_scratch is not None:
            from .gpu_random import _seed_from_string, _normal
            for i, nonce in enumerate(nonces):
                seed = _seed_from_string(
                    f"{block_hash}_{public_key}_nonce{nonce}")
                vals = _normal(seed, seq_len * hidden_size, device)
                kv_scratch[i].copy_(vals.view(seq_len, hidden_size).to(dtype))
                del vals
            inputs_embeds = kv_scratch
        else:
            _gen_fn = generate_inputs_concat_murmur if poc_stronger_rng else generate_inputs
            inputs_embeds = _gen_fn(
                block_hash, public_key, nonces,
                dim=hidden_size, seq_len=seq_len,
                device=device, dtype=dtype,
            )
    else:
        intermediate_tensors = IntermediateTensors(
            pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
        )

    with set_forward_context(
        attn_metadata, vllm_config,
        num_tokens=batch_size * seq_len,
        slot_mapping=slot_mapping_dict,
        skip_compiled=True,
    ):
        with poc_forward_context():
            hidden_states = model(
                input_ids=poc_input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds.view(-1, hidden_size) if inputs_embeds is not None else None,
            )

    # PP: send to next rank if not last
    if not pp_group.is_last_rank:
        if isinstance(hidden_states, IntermediateTensors):
            pp_group.send_tensor_dict(
                hidden_states.tensors, all_gather_group=get_tp_group()
            )
        return None

    # Handle tuple return
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]

    # Extract last hidden per sequence
    hidden_states = hidden_states.view(batch_size, seq_len, -1)
    last_hidden = hidden_states[:, -1, :].float()  # [batch_size, hidden_size]

    # NaN detection
    nan_mask = torch.isnan(last_hidden).any(dim=-1)  # [batch_size]
    if nan_mask.any():
        clean_idx = (~nan_mask).nonzero(as_tuple=True)[0]
        nan_count = nan_mask.sum().item()
        logger.warning("NaN in %d/%d hidden states (GPU fault?)", nan_count, batch_size)

        if clean_idx.numel() == 0:
            logger.error("All %d nonces produced NaN — batch rejected", batch_size)
            return {"nonces": [], "vectors": np.empty((0, k_dim), dtype=np.float16)}

        last_hidden = last_hidden[clean_idx]
        nonces = [nonces[i] for i in clean_idx.tolist()]
        batch_size = len(nonces)

    # Normalize to unit sphere
    last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)

    # Batched k-dim pick + Haar rotation
    indices = random_pick_indices(block_hash, public_key, nonces, hidden_size, k_dim, device)
    xk = torch.gather(last_hidden, 1, indices)
    yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)

    # Normalize output vectors
    yk = yk / (yk.norm(dim=-1, keepdim=True) + 1e-8)

    # Convert to FP16
    vectors_f16 = yk.half().cpu().numpy()  # [batch_size, k_dim]

    # Late NaN check after FP16 conversion
    nan_out = np.isnan(vectors_f16).any(axis=1)
    if nan_out.any():
        clean = ~nan_out
        vectors_f16 = vectors_f16[clean]
        nonces = [n for n, c in zip(nonces, clean) if c]
        logger.warning("NaN in FP16 output — %d nonces filtered", nan_out.sum())

    return {
        "nonces": nonces,
        "vectors": vectors_f16,
    }
