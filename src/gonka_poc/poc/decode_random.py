"""Decode-scheme RNG: everything the sphere_k trajectory needs.

Split out of ``gpu_random`` on purpose. That module carries the PREFILL
derivation the deployed fleet validates, and it is kept byte-identical to
v0.1.x -- so the decode scheme may not add a parameter, a branch or an
allocation to it. Shared primitives are imported, never copied: both schemes
must hash and draw through the exact same code.
"""
import logging
import math
from typing import List, Optional

import torch

from gonka_poc.poc.gpu_random import (
    _batched_murmur3_32,
    _normal,
    _seed_from_string,
)

logger = logging.getLogger(__name__)

_SALT_DECODE_EMBED = 0x0D
_SALT_DECODE_PICK = 0x91
_SALT_DECODE_TOKEN_ID = 0x57  # pseudo token ids, see decode_pseudo_token_ids
_MIX_A = 0x9E3779B1  # golden-ratio odd constant
_MIX_B = 0x85EBCA77


def pinned_to_device(vals, dtype, device):
    """Build a small [N] device tensor from a host sequence WITHOUT stalling the
    async pipeline.

    `torch.tensor(list, device='cuda')` — and indexing a CUDA tensor with a Python
    list — construct on-device via a BLOCKING host->device copy that synchronizes
    on the compute stream. Under async scheduling that stalls every decode step on
    the model forward (measured ~17 ms/step; see the decode-PoC tail). Building a
    pinned host tensor and copying non_blocking avoids the sync. The values are
    identical to the direct construction, so PoC artifacts stay bit-for-bit the
    same — do NOT "simplify" this back to torch.tensor(..., device=cuda).
    """
    cuda = torch.device(device).type == "cuda"
    return torch.tensor(vals, dtype=dtype, pin_memory=cuda).to(device, non_blocking=cuda)


def generate_decode_inputs(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    prev_k: List[int],
    step: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate deterministic decode-step embedding chained to the previous sphere_k.

    The seed incorporates prev_k so each decode step is deterministically
    linked to its predecessor.

    Returns:
        Tensor of shape [batch_size, 1, dim]
    """
    batch_size = len(nonces)
    result = torch.empty(batch_size, 1, dim, device=device, dtype=dtype)
    for i, (nonce, k) in enumerate(zip(nonces, prev_k)):
        seed_str = f"{block_hash}_{public_key}_nonce{nonce}_decode{step}_k{k}"
        seed = _seed_from_string(seed_str)
        normal = _normal(seed, dim, device)
        result[i, 0] = normal.to(dtype)
    return result


def decode_base_seeds(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Per-nonce base seed (constant for the whole request) -> [B] int64 on device.
    Computed once; carries no per-step dependency, so the host SHA256 here is fine."""
    seeds = [_seed_from_string(f"{block_hash}_{public_key}_nonce{n}") for n in nonces]
    return torch.tensor(seeds, dtype=torch.int64, device=device)


def _step_seeds(
    base_seeds: torch.Tensor, step: int, prev_k: torch.Tensor, salt: int
) -> torch.Tensor:
    """Per-step seed = on-GPU murmur3 mixing base (per nonce) with step + prev_k.

    base_seeds [B] int64 (constant), prev_k [B] int64 (chained on device, NEVER
    .item()'d). step is a host int (same for the whole batch) OR a [B] int64 tensor
    (per-row step, so a whole nonce-batch can be chained in ONE call). Returns [B]
    int64 fully on device, so the decode chain has no GPU->CPU sync. Avalanche from
    murmur3 makes consecutive steps / prev_k values uncorrelated (same property the
    SHA256 path gave). The per-row result is identical to calling this once per row."""
    if torch.is_tensor(step):
        step_term = step.to(torch.int64).view(-1) * _MIX_B + salt
    else:
        step_term = int(step) * _MIX_B + salt
    key = ((prev_k.to(torch.int64).view(-1) & 0xFFFFFFFF) * _MIX_A
           + step_term) & 0xFFFFFFFF
    return _batched_murmur3_32(key.view(-1, 1), base_seeds.view(-1, 1)).view(-1)


def _batched_normal_t(seeds: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    """Like _batched_normal but `seeds` is already an int64 tensor [B] (no host
    list). Returns [B, n] standard normals, fully on device."""
    batch_size = seeds.shape[0]
    n_pairs = (n + 1) // 2
    total = n_pairs * 2
    indices = torch.arange(total, device=device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)
    h = _batched_murmur3_32(indices, seeds.view(-1, 1))
    u = h.to(torch.float32) / 4294967296.0
    u1 = torch.clamp(u[:, :n_pairs], min=1e-10)
    u2 = u[:, n_pairs:]
    z0 = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    z1 = torch.sqrt(-2.0 * torch.log(u1)) * torch.sin(2.0 * math.pi * u2)
    return torch.cat([z0, z1], dim=1)[:, :n]


def generate_decode_inputs_gpu(
    base_seeds: torch.Tensor,
    prev_k: torch.Tensor,
    step: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """GPU-native counterpart of generate_decode_inputs: next decode-step input
    embedding chained to prev_k (tensor). Returns [B, 1, dim]."""
    seeds = _step_seeds(base_seeds, step, prev_k, _SALT_DECODE_EMBED)
    return _batched_normal_t(seeds, dim, device).to(dtype).unsqueeze(1)


def random_pick_indices_gpu(
    base_seeds: torch.Tensor,
    prev_k: torch.Tensor,
    step: int,
    dim: int,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    """GPU-native counterpart of random_pick_indices (decode): k dims per row,
    seed chained to prev_k (tensor). Returns [B, k] int64."""
    if k <= 0 or k > dim:
        raise ValueError(f"k must be in [1, dim], got k={k}, dim={dim}")
    seeds = _step_seeds(base_seeds, step, prev_k, _SALT_DECODE_PICK)
    all_idx = torch.arange(dim, device=device, dtype=torch.int32).unsqueeze(0).expand(seeds.shape[0], -1)
    scores = _batched_murmur3_32(all_idx, seeds.view(-1, 1))
    _, chosen = torch.topk(-scores, k=k, largest=True, sorted=False, dim=1)
    return chosen.to(torch.int64)


# Ladder base for seeded routing: a consensus constant of the decode scheme.
# One value for every model since 2026-09-07: 100. The base lifts the forced
# ladder above any router bias term (DeepSeek-V4 sqrtsoftplus + bias) so the
# seeded experts win regardless of architecture; on MiniMax-M2.7 a fresh
# B300<->H200 campaign showed the thresholds do not depend on the base (honest
# cross 7.5-7.6 % at either base, fraud 12.2-12.4 %, gap 4.8 pp), so the
# per-model map was dropped. Reference corpora frozen at base 0 (campaign R,
# August) no longer apply to MiniMax; the base-100 goldens replace them.
# Pending sign-off by the scheme owner (Ilya) before release.
LADDER_BASE = 100
_ladder_base = LADDER_BASE


def set_ladder_base_for_model(model_type) -> int:
    """Return the ladder base (one for all models; ``model_type`` is logged by
    the caller and kept in the signature for the attach-time log line)."""
    global _ladder_base
    _ladder_base = LADDER_BASE
    return _ladder_base


def ladder_base() -> int:
    return _ladder_base


def _forced_logits(seed: torch.Tensor, n_experts: int, top_k: int,
                   device: torch.device) -> torch.Tensor:
    """THE seeded expert selection — single source of truth for seeded routing.

    Contiguous seeded run: ``start = seed % n_experts``, experts
    ``start .. start+top_k-1`` (mod n). Distinct by construction for ANY
    n_experts, one arithmetic op, trivial to reimplement bit-exactly in the
    chain-side validator. Uniform per-expert coverage across seeds keeps the
    prover holding EVERY expert; unpredictability of the set is not a goal
    (seeds are public) — consensus security lives in the seeded embeds,
    per-layer reflections and the chained snap. Returns [B, n_experts]
    forced logits: the chosen run holds descending ladder values
    ``base+top_k .. base+1`` (``base`` from ``set_ladder_base_for_model``),
    the rest a low floor."""
    b = seed.shape[0]
    start = torch.remainder(seed.view(-1, 1), n_experts)             # [B,1]
    offs = torch.arange(top_k, device=device, dtype=torch.int64)     # [k]
    chosen = torch.remainder(start + offs.unsqueeze(0), n_experts)   # [B,k]
    logits = torch.full((b, n_experts), -1.0e4, device=device,
                        dtype=torch.float32)
    logits.scatter_(1, chosen,
                    (torch.arange(top_k, 0, -1, device=device,
                                  dtype=torch.float32)
                     + _ladder_base).unsqueeze(0).expand(b, -1))
    return logits


def decode_pseudo_token_ids(base_seeds: torch.Tensor, step: int | torch.Tensor,
                           prev_k: torch.Tensor, vocab: int) -> torch.Tensor:
    """Per-step pseudo token ids for architectures that route by token id
    (DeepSeek-V4 hash-MoE, ``tid2eid[input_ids]``): a constant id there would
    pin those layers to one expert set for every nonce and step.

    Same (base, step, prev_k) chain as the decode embeds under its own salt,
    integer-only on device. Decode counterpart of :func:`derive_pseudo_input_ids`.
    """
    seeds = _step_seeds(base_seeds, step, prev_k, _SALT_DECODE_TOKEN_ID)
    keys = torch.zeros_like(seeds, dtype=torch.int32).view(-1, 1)
    return (_batched_murmur3_32(keys, seeds.view(-1, 1)) % vocab).to(
        torch.int32).flatten()


def route_base_seed(block_hash: str, nonce: int, layer: int) -> str:
    """STABLE part of the routing seed: (block_hash, nonce, layer) — everything except
    the decode step. sha256'd ONCE per (block_hash,nonce,layer) and cached across all
    decode steps; ``step`` is folded in on-GPU per step (see expert_logits_from_base).
    This is the efficiency contract: NO per-step string hashing (the K-calc lesson)."""
    return f"{block_hash}_n{nonce}_route_layer_{layer}"


def expert_logits_from_base(base_ints: torch.Tensor, steps: torch.Tensor,
                            n_experts: int, top_k: int,
                            device: torch.device) -> torch.Tensor:
    """Per-row forced router logits: fold the decode ``step`` into the cached base seed
    ON GPU, then the shared _forced_logits pick. ``base_ints``/``steps`` are [B]
    int64; returns [B, n_experts]. All integer (bit-exact cross-HW), no host loop, no
    device->host sync. Equivalent per (row, layer) to seeded_experts()."""
    seed = _batched_murmur3_32(steps.view(-1, 1).to(torch.int32),
                               base_ints.view(-1, 1))               # [B,1] = fold step into base
    return _forced_logits(seed, n_experts, top_k, device)


def seeded_experts(block_hash: str, nonce: int, step: int, layer: int,
                   n_experts: int, top_k: int, device: torch.device) -> torch.Tensor:
    """Reference (single-row) seeded experts = the EXACT live derivation: cached
    sha256 base (block_hash+nonce+layer) then on-GPU ``step`` fold. Returns the chosen
    expert indices. For tests / offline validators (the live runner uses the batched,
    cached expert_logits_from_base, which is identical per row)."""
    base = torch.tensor([_seed_from_string(route_base_seed(block_hash, nonce, layer))],
                        dtype=torch.int64, device=device)
    steps = torch.tensor([step], dtype=torch.int64, device=device)
    logits = expert_logits_from_base(base, steps, n_experts, top_k, device)[0]
    return torch.topk(logits, top_k).indices


def random_pick_indices_decode(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    k: int,
    device: torch.device,
    prev_point_ids: Optional[List[int]] = None,
    step: int = 0,
) -> torch.Tensor:
    """Decode-scheme counterpart of :func:`gpu_random.random_pick_indices`.

    Same derivation, plus the decode salt: the step, and the previous sphere
    index once the chain has one. Kept HERE, not as a flag on the prefill
    function, so the prefill scheme's file stays byte-identical to v0.1.x.
    """
    if k <= 0 or k > dim:
        raise ValueError(f"k must be in [1, dim], got k={k}, dim={dim}")
    seeds = []
    for i, nonce in enumerate(nonces):
        if prev_point_ids is None:
            seeds.append(_seed_from_string(
                f"{block_hash}_{public_key}_nonce_{nonce}_pick_{k}_decode{step}"))
        else:
            seeds.append(_seed_from_string(
                f"{block_hash}_{public_key}_nonce_{nonce}_pick_{k}_decode{step}"
                f"_k_{prev_point_ids[i]}"))
    all_idx = torch.arange(dim, device=device, dtype=torch.int32).unsqueeze(0).expand(
        len(nonces), -1)
    seed_tensor = torch.tensor(seeds, dtype=torch.int64, device=device).unsqueeze(1)
    scores = _batched_murmur3_32(all_idx, seed_tensor)
    _, chosen = torch.topk(-scores, k=k, largest=True, sorted=False, dim=1)
    return chosen.to(torch.int64)


def random_pick_indices_decode_steps(
    block_hash: str,
    public_key: str,
    nonce: int,
    dim: int,
    k: int,
    device: torch.device,
    steps: List[int],
) -> torch.Tensor:
    """Same indices as random_pick_indices_decode(..., [nonce], step=s) for each
    s in steps, as one batch [len(steps), k] (one murmur, one topk; used for
    trajectory emission). prev_point_ids is not supported."""
    if k <= 0 or k > dim:
        raise ValueError(f"k must be in [1, dim], got k={k}, dim={dim}")
    seeds = [_seed_from_string(
        f"{block_hash}_{public_key}_nonce_{nonce}_pick_{k}_decode{step}")
        for step in steps]
    all_idx = torch.arange(dim, device=device, dtype=torch.int32).unsqueeze(0).expand(
        len(steps), -1)
    seed_tensor = torch.tensor(seeds, dtype=torch.int64, device=device).unsqueeze(1)
    scores = _batched_murmur3_32(all_idx, seed_tensor)
    _, chosen = torch.topk(-scores, k=k, largest=True, sorted=False, dim=1)
    return chosen.to(torch.int64)
