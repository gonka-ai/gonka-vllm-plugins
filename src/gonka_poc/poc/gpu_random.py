"""Deterministic seeded RNG primitives for the PoC forward.

Reproducible random tensors seeded by (block_hash, public_key, nonce).

CONSENSUS-CRITICAL, in this precise sense: prover and validator derive the
model's input vectors INDEPENDENTLY, each running these functions on its own
hardware. Validation then compares the resulting model outputs statistically
(per-nonce L2 distance against a threshold, then a binomial test over the
mismatch count) -- numeric noise within the tolerance is expected and passes.
What must therefore stay fixed is the DERIVATION ALGORITHM: a node running a
different derivation produces input vectors unrelated to the fleet's, its
outputs land beyond the threshold on every nonce, and an honest node fails
validation. Bitwise equality of outputs is neither required nor checked.
"""
import hashlib
import math
from typing import List, Optional

import logging

import torch

logger = logging.getLogger(__name__)


def _seed_from_string(seed_string: str) -> int:
    h = hashlib.sha256(seed_string.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


# _murmur3_32/_batched_murmur3_32 are kept separate deliberately: the fleet
# derives inputs with these exact code paths. Do not unify without a
# cross-validator harness proving the derivation is unchanged.
def _murmur3_32(keys: torch.Tensor, seed: int) -> torch.Tensor:
    """Murmur3 hash for int32 keys. Returns int64 to preserve full uint32 range."""
    c1, c2 = 0xCC9E2D51, 0x1B873593

    h = torch.full_like(keys, seed & 0xFFFFFFFF, dtype=torch.int64)
    k = keys.to(torch.int64) & 0xFFFFFFFF

    k = (k * c1) & 0xFFFFFFFF
    k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
    k = (k * c2) & 0xFFFFFFFF

    h = h ^ k
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF

    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h


def _batched_murmur3_32(keys: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    """Batched Murmur3 hash with per-row seeds.

    Args:
        keys: [batch_size, n] int32 tensor
        seeds: [batch_size, 1] int64 tensor
    Returns:
        [batch_size, n] int64 tensor
    """
    c1, c2 = 0xCC9E2D51, 0x1B873593

    h = (seeds & 0xFFFFFFFF).expand_as(keys.to(torch.int64))
    k = keys.to(torch.int64) & 0xFFFFFFFF

    k = (k * c1) & 0xFFFFFFFF
    k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
    k = (k * c2) & 0xFFFFFFFF

    h = h ^ k
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF

    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h


# _normal/_batched_normal are kept separate deliberately: the fleet derives
# inputs with these exact code paths. Do not unify without a cross-validator
# harness proving the derivation is unchanged.
def _batched_normal(seeds: list, n: int, device: torch.device) -> torch.Tensor:
    """Generate batched normal random numbers for multiple seeds.

    Args:
        seeds: List of integer seeds
        n: Number of random numbers per seed
        device: Target device

    Returns:
        Tensor of shape [len(seeds), n]
    """
    batch_size = len(seeds)
    n_pairs = (n + 1) // 2
    total = n_pairs * 2

    indices = torch.arange(total, device=device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)
    seed_tensor = torch.tensor(seeds, dtype=torch.int64, device=device).unsqueeze(1)

    h = _batched_murmur3_32(indices, seed_tensor)
    u = h.to(torch.float32) / 4294967296.0

    u1 = u[:, :n_pairs]
    u2 = u[:, n_pairs:]
    u1 = torch.clamp(u1, min=1e-10)

    z0 = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    z1 = torch.sqrt(-2.0 * torch.log(u1)) * torch.sin(2.0 * math.pi * u2)
    return torch.cat([z0, z1], dim=1)[:, :n]


def _uniform(seed: int, n: int, device: torch.device) -> torch.Tensor:
    indices = torch.arange(n, device=device, dtype=torch.int32)
    hashes = _murmur3_32(indices, seed)
    return hashes.to(torch.float32) / 4294967296.0


def _normal(seed: int, n: int, device: torch.device) -> torch.Tensor:
    n_pairs = (n + 1) // 2
    u = _uniform(seed, n_pairs * 2, device)
    u1, u2 = u[:n_pairs], u[n_pairs:]
    u1 = torch.clamp(u1, min=1e-10)
    z0 = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    z1 = torch.sqrt(-2.0 * torch.log(u1)) * torch.sin(2.0 * math.pi * u2)
    return torch.cat([z0, z1])[:n]


def generate_inputs(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate deterministic input embeddings for PoC.

    Batched: ONE _batched_normal over all nonces instead of a serial per-nonce
    Python loop (which was ~B sequential big RNGs on the prefill critical path).
    Per-row identical to the old loop (same seed -> same murmur -> same normals)."""
    seeds = [_seed_from_string(f"{block_hash}_{public_key}_nonce{n}") for n in nonces]
    normals = _batched_normal(seeds, seq_len * dim, device)  # [B, seq_len*dim]
    return normals.view(len(nonces), seq_len, dim).to(dtype)


def generate_inputs_concat_murmur(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate deterministic input embeddings using concat-murmur (stronger RNG).

    Uses all 256 bits of SHA256 by splitting into 8 × 32-bit sub-seeds.
    Each sub-seed generates one segment of length ceil(n/8) via the existing
    murmur3 pipeline; segments are concatenated.
    """
    batch_size = len(nonces)
    result = torch.empty(batch_size, seq_len, dim, device=device, dtype=dtype)
    n = seq_len * dim
    seg_len = (n + 7) // 8  # ceil(n/8); last segment may be shorter

    for i, nonce in enumerate(nonces):
        h = hashlib.sha256(
            f"{block_hash}_{public_key}_nonce{nonce}".encode()
        ).digest()
        sub_seeds = [int.from_bytes(h[j:j + 4], 'big') for j in range(0, 32, 4)]

        segments = [
            _normal(s, min(seg_len, n - k * seg_len), device)
            for k, s in enumerate(sub_seeds)
            if k * seg_len < n
        ]
        flat = torch.cat(segments)[:n]
        result[i] = flat.view(seq_len, dim).to(dtype)

    return result


def derive_pseudo_input_ids(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    vocab: int,
    device: torch.device,
) -> torch.Tensor:
    """Deterministic pseudo token ids for token-id-dependent architectures.

    Ids are derived from the same ``(block_hash, public_key, nonce)`` seed
    scheme as the input embeddings (``_input_ids`` suffix), through the same
    framework-independent murmur3 pipeline — pure integer arithmetic, stable
    across torch versions (a consensus requirement).
    """
    batch_size = len(nonces)
    keys = torch.arange(seq_len, dtype=torch.int32, device=device)
    keys = keys.unsqueeze(0).expand(batch_size, -1)
    seeds = torch.tensor(
        [[_seed_from_string(f"{block_hash}_{public_key}_nonce{n}_input_ids")]
         for n in nonces],
        dtype=torch.int64, device=device)
    # murmur3 yields uniform uint32; modulo bias at vocab << 2^32 is
    # negligible for routing purposes.
    return (_batched_murmur3_32(keys, seeds) % vocab).to(torch.int32).flatten()


def generate_householder_vector(
    seed_str: str,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a single unit vector for Householder reflection."""
    seed = _seed_from_string(seed_str)
    v = _normal(seed, dim, device)
    return v / v.norm()


def apply_householder(
    x: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Apply Householder reflection: H @ x = x - 2*(v.x)*v"""
    dot = (x * v).sum(dim=-1, keepdim=True)
    return x - 2 * dot * v


def random_pick_indices(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    k: int,
    device: torch.device,
    prev_point_ids: Optional[List[int]] = None,
    step: int = 0,
) -> torch.Tensor:
    """Pick k dimensions per nonce deterministically (vectorized).

    When prev_point_ids is provided the seed is mixed with the previous
    sphere index so decode steps pick a different subset than prefill.
    """
    if k <= 0 or k > dim:
        raise ValueError(f"k must be in [1, dim], got k={k}, dim={dim}")

    batch_size = len(nonces)

    seeds = []
    for i, nonce in enumerate(nonces):
        if prev_point_ids is None:
            seeds.append(_seed_from_string(
                f"{block_hash}_{public_key}_nonce_{nonce}_pick_{k}_decode{step}"
            ))
        else:
            seeds.append(_seed_from_string(
                f"{block_hash}_{public_key}_nonce_{nonce}_pick_{k}_decode{step}_k_{prev_point_ids[i]}"
            ))

    all_idx = torch.arange(dim, device=device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)
    seed_tensor = torch.tensor(seeds, dtype=torch.int64, device=device).unsqueeze(1)
    scores = _batched_murmur3_32(all_idx, seed_tensor)

    _, chosen = torch.topk(-scores, k=k, largest=True, sorted=False, dim=1)
    return chosen.to(torch.int64)


def apply_haar_rotation(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Apply Haar-random rotation via k-1 Householder reflections (vectorized)."""
    batch_size, k = x.shape
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    y = x.clone()

    all_seeds_by_step = []
    for j in range(k - 1):
        step_seeds = []
        for nonce in nonces:
            step_seeds.append(_seed_from_string(
                f"{block_hash}_{public_key}_nonce_{nonce}_haar_hh_{k}_{j}"
            ))
        all_seeds_by_step.append(step_seeds)

    for j in range(k - 1):
        v_batch = _batched_normal(all_seeds_by_step[j], k, device)
        v_batch = v_batch / (v_batch.norm(dim=-1, keepdim=True) + 1e-30)
        v_batch = v_batch.to(y.dtype)

        dot = (y * v_batch).sum(dim=-1, keepdim=True)
        y = y - 2 * dot * v_batch

    return y


# ============================================================================
# Decode-PoC family — ported bit-for-bit from the 0.20 in-tree branch
# (poc-v0.20-decode-poc-cg @ 5c1d09f55e92).  CONSENSUS-CRITICAL: seed strings,
# salts, murmur3 pipeline and the routing window define the k-trajectories.
# Decisions applied (see migration NOTES.md):
#   * #1/#4 — the decode seed scheme is the ONLY scheme in this release:
#     random_pick_indices above is the 0.20 variant (suffix `_decode{step}`,
#     optional `_k_{prev}`); the legacy `_pick_{k}` seeding lives in tag v0.1.x.
#   * #11 — the routing window ships as 256 via the request (set_route_window).
# ============================================================================


_SALT_DECODE_EMBED = 0x0D


_SALT_DECODE_PICK = 0x91


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


def random_pick_indices(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    k: int,
    device: torch.device,
    prev_point_ids: Optional[List[int]] = None,
    step: int = 0,
) -> torch.Tensor:
    """Pick k dimensions per nonce deterministically (vectorized).

    When prev_point_ids is provided the seed is mixed with the previous
    sphere index so decode steps pick a different subset than prefill.
    """
    if k <= 0 or k > dim:
        raise ValueError(f"k must be in [1, dim], got k={k}, dim={dim}")

    batch_size = len(nonces)

    seeds = []
    for i, nonce in enumerate(nonces):
        if prev_point_ids is None:
            seeds.append(_seed_from_string(
                f"{block_hash}_{public_key}_nonce_{nonce}_pick_{k}_decode{step}"
            ))
        else:
            seeds.append(_seed_from_string(
                f"{block_hash}_{public_key}_nonce_{nonce}_pick_{k}_decode{step}_k_{prev_point_ids[i]}"
            ))

    all_idx = torch.arange(dim, device=device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)
    seed_tensor = torch.tensor(seeds, dtype=torch.int64, device=device).unsqueeze(1)
    scores = _batched_murmur3_32(all_idx, seed_tensor)

    _, chosen = torch.topk(-scores, k=k, largest=True, sorted=False, dim=1)
    return chosen.to(torch.int64)


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


_ROUTE_WINDOW = 16


def set_route_window(n: int) -> None:
    """Push the MoE seeded-routing window from the broadcast engine config into the
    module global, once per worker before graph capture. Logged so a per-worker
    mismatch (a config value that didn't reach a worker) shows: grep "PoC route window"."""
    global _ROUTE_WINDOW
    _ROUTE_WINDOW = int(n)
    logger.info("PoC route window = %d (--poc-route-window); CONSENSUS-AFFECTING, "
                "must match on all nodes/workers.", _ROUTE_WINDOW)


def _forced_logits(seed: torch.Tensor, n_experts: int, top_k: int,
                   device: torch.device) -> torch.Tensor:
    """THE seeded expert selection — single source of truth for seeded routing.

    Draws top_k DISTINCT experts straight from ``seed`` ([B,1] int64) by a partial
    Fisher-Yates shuffle: pure integer (no topk, no scores, no ties) -> identical on
    any hardware / eager / graph, and robust for ANY n_experts. Returns [B,n_experts]
    forced logits (chosen experts hold descending values, the rest a low floor); the
    natural MoE top-k over these picks exactly the seeded experts + gate weights."""
    b = seed.shape[0]
    perm = torch.arange(n_experts, device=device, dtype=torch.int64).unsqueeze(0).repeat(b, 1)
    for i in range(top_k):                                # k swaps -> k distinct experts in perm[:, :k]
        j = i + _batched_murmur3_32(torch.full((b, 1), i, dtype=torch.int32, device=device),
                                    seed) % (n_experts - i)          # [B,1] swap target in [i, n)
        gi = perm[:, i:i + 1].clone()
        perm[:, i:i + 1] = perm.gather(1, j)
        perm.scatter_(1, j, gi)
    logits = torch.full((b, n_experts), -1.0e4, device=device, dtype=torch.float32)
    logits.scatter_(1, perm[:, :top_k],
                    torch.arange(top_k, 0, -1, device=device, dtype=torch.float32).unsqueeze(0).expand(b, -1))
    return logits


# Grouped top-k forcing (DeepSeek family; Kimi shares the architecture).
# CONSENSUS constants and formula — phase-0 spec, approved 15.08.2026:
# stage G picks topk_group groups, stage E picks top_k experts INSIDE the
# picked groups with a coverage guarantee (>=1 expert per picked group —
# an empty picked group would tie at the -1e4 floor with losing groups and
# make the engine's group selection nondeterministic).
_SALT_ROUTE_GROUP = 0x3A
_SALT_ROUTE_EXPERT = 0x5C


def _forced_logits_grouped(seed: torch.Tensor, n_experts: int, top_k: int,
                           n_group: int, topk_group: int,
                           device: torch.device) -> torch.Tensor:
    """Seeded expert selection compatible with grouped top-k routers.

    Plain _forced_logits spreads the picks over ALL experts; a grouped router
    first keeps only ``topk_group`` groups, so off-group picks could never be
    selected and the -1e4 ties would leave the outcome to kernel order. Two
    seeded stages fix that:

      G: partial Fisher-Yates over ``n_group`` from seed_g -> picked groups;
      E: one guaranteed expert per picked group (coverage), then the
         remaining ``top_k - topk_group`` picks over the rest of the picked
         groups' experts, all from seed_e.

    seed_g/seed_e derive from the (base ^ step) seed with dedicated salts, so
    neither stage correlates with the other or with the flat formula. Forced
    values are ``top_k..1`` on chosen experts and -1e4 elsewhere: gaps are
    orders of magnitude beyond |e_score_correction_bias|, so the bias can
    shift scores but never the selected set; expert WEIGHTS derive from these
    constants (deterministic), and intra-top-k order does not affect the
    weighted sum. Integer-only selection — bit-identical on any hardware,
    eager or graph.
    """
    b = seed.shape[0]
    group_size = n_experts // n_group
    if n_group * group_size != n_experts:
        raise ValueError(
            f"grouped forcing: n_experts {n_experts} not divisible by "
            f"n_group {n_group}")
    if top_k < topk_group:
        raise ValueError(
            f"grouped forcing: top_k {top_k} < topk_group {topk_group} — "
            f"coverage guarantee impossible; needs a protocol decision")
    seed_g = _batched_murmur3_32(
        torch.full((b, 1), _SALT_ROUTE_GROUP, dtype=torch.int32, device=device),
        seed)
    seed_e = _batched_murmur3_32(
        torch.full((b, 1), _SALT_ROUTE_EXPERT, dtype=torch.int32, device=device),
        seed)

    # -- stage G: pick topk_group distinct groups (partial Fisher-Yates) ----
    gperm = torch.arange(n_group, device=device, dtype=torch.int64
                         ).unsqueeze(0).repeat(b, 1)
    for i in range(topk_group):
        j = i + _batched_murmur3_32(
            torch.full((b, 1), i, dtype=torch.int32, device=device),
            seed_g) % (n_group - i)
        gi = gperm[:, i:i + 1].clone()
        gperm[:, i:i + 1] = gperm.gather(1, j)
        gperm.scatter_(1, j, gi)
    groups = gperm[:, :topk_group]                       # [B, topk_group]

    # -- stage E over the picked groups' expert slots ------------------------
    # Virtual layout: slot (g, r) = expert groups[g]*group_size + r. Coverage
    # first: for picked group g (g < topk_group) pick one in-group offset;
    # then the remaining top_k - topk_group picks run a partial Fisher-Yates
    # over the remaining topk_group*(group_size-1) virtual slots.
    off = _batched_murmur3_32(
        torch.arange(topk_group, dtype=torch.int32, device=device
                     ).unsqueeze(0).repeat(b, 1),
        seed_e) % group_size                              # [B, topk_group]
    cover = groups * group_size + off                     # [B, topk_group]

    rest = top_k - topk_group
    picks = [cover]
    if rest > 0:
        # Remaining slots per picked group: group_size-1 offsets, skipping
        # the covered one. Virtual index v in [0, topk_group*(group_size-1)):
        #   g = v // (group_size-1); r0 = v % (group_size-1)
        #   r  = r0 + (r0 >= off[g])           (skip the covered offset)
        m = topk_group * (group_size - 1)
        vperm = torch.arange(m, device=device, dtype=torch.int64
                             ).unsqueeze(0).repeat(b, 1)
        for i in range(rest):
            j = i + _batched_murmur3_32(
                torch.full((b, 1), 0x100 + i, dtype=torch.int32, device=device),
                seed_e) % (m - i)
            vi = vperm[:, i:i + 1].clone()
            vperm[:, i:i + 1] = vperm.gather(1, j)
            vperm.scatter_(1, j, vi)
        v = vperm[:, :rest]                               # [B, rest]
        g = v // (group_size - 1)
        r0 = v % (group_size - 1)
        goff = off.gather(1, g)
        r = r0 + (r0 >= goff).to(torch.int64)
        picks.append(groups.gather(1, g) * group_size + r)
    chosen = torch.cat(picks, dim=1)                      # [B, top_k]

    logits = torch.full((b, n_experts), -1.0e4, device=device,
                        dtype=torch.float32)
    logits.scatter_(1, chosen,
                    torch.arange(top_k, 0, -1, device=device,
                                 dtype=torch.float32).unsqueeze(0).expand(b, -1))
    return logits


def _forced_logits_windowed(seed: torch.Tensor, steps: torch.Tensor, n_experts: int,
                            top_k: int, window: int, device: torch.device) -> torch.Tensor:
    """Seeded expert selection restricted to a step-rotating WINDOW of ``window`` experts.

    The window offset depends on the decode STEP (shared across the batch, NOT per-nonce),
    so a decode batch activates only ~window distinct experts per step -> the MoE grouped
    GEMM batches them (measured ~7x faster than the full-scatter pick). The offset slides by
    ``stride`` each step so over the trajectory the window sweeps every expert (coverage /
    fraud-bound preserved). Each token still seed-picks top_k WITHIN the window (trajectories
    stay distinct). Pure integer -> eager==graph, bit-identical cross-HW. seed/steps [B,1]."""
    b = seed.shape[0]
    W = max(window, top_k)
    stride = max(1, n_experts // 256)                    # sweep ~all experts over a 256-step trajectory
    offset = (steps * stride) % n_experts                # [B,1] window start (batch-shared at a given step)
    perm = torch.arange(W, device=device, dtype=torch.int64).unsqueeze(0).repeat(b, 1)
    for i in range(top_k):                               # partial Fisher-Yates over [0, W)
        j = i + _batched_murmur3_32(torch.full((b, 1), i, dtype=torch.int32, device=device),
                                    seed) % (W - i)
        gi = perm[:, i:i + 1].clone()
        perm[:, i:i + 1] = perm.gather(1, j)
        perm.scatter_(1, j, gi)
    chosen = (offset + perm[:, :top_k]) % n_experts      # window-local -> global expert ids (distinct)
    logits = torch.full((b, n_experts), -1.0e4, device=device, dtype=torch.float32)
    logits.scatter_(1, chosen,
                    torch.arange(top_k, 0, -1, device=device, dtype=torch.float32).unsqueeze(0).expand(b, -1))
    return logits


def route_base_seed(block_hash: str, nonce: int, layer: int) -> str:
    """STABLE part of the routing seed: (block_hash, nonce, layer) — everything except
    the decode step. sha256'd ONCE per (block_hash,nonce,layer) and cached across all
    decode steps; ``step`` is folded in on-GPU per step (see expert_logits_from_base).
    This is the efficiency contract: NO per-step string hashing (the K-calc lesson)."""
    return f"{block_hash}_n{nonce}_route_layer_{layer}"


def expert_logits_from_base(base_ints: torch.Tensor, steps: torch.Tensor,
                            n_experts: int, top_k: int,
                            device: torch.device,
                            n_group: int = 1,
                            topk_group: int = 1) -> torch.Tensor:
    """Per-row forced router logits: fold the decode ``step`` into the cached base seed
    ON GPU, then the shared _forced_logits selection. ``base_ints``/``steps`` are [B]
    int64; returns [B, n_experts]. All integer (bit-exact cross-HW), no host loop, no
    device->host sync. Equivalent per (row, layer) to seeded_experts()."""
    seed = _batched_murmur3_32(steps.view(-1, 1).to(torch.int32),
                               base_ints.view(-1, 1))               # [B,1] = fold step into base
    if n_group > 1:
        # Grouped router (DeepSeek family): the two-stage formula; the flat
        # window path below stays byte-identical for n_group <= 1 (MiniMax).
        return _forced_logits_grouped(seed, n_experts, top_k, n_group,
                                      topk_group, device)
    if _ROUTE_WINDOW and _ROUTE_WINDOW < n_experts:
        return _forced_logits_windowed(seed, steps.view(-1, 1), n_experts, top_k,
                                       _ROUTE_WINDOW, device)
    return _forced_logits(seed, n_experts, top_k, device)


def seeded_expert_logits(seed_str: str, n_experts: int, top_k: int,
                         device: torch.device) -> torch.Tensor:
    """Single-row forced logits from a seed string (tests / offline validators).
    Same selection as the live path — both go through the shared _forced_logits."""
    seed = torch.tensor([[_seed_from_string(seed_str)]], dtype=torch.int64, device=device)
    return _forced_logits(seed, n_experts, top_k, device)[0]


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
