"""Public sampling tape.

A request carrying a tape seed samples each token as the Gumbel-max winner of
its own sampling distribution under noise addressed by (seed, output position,
token id). Gumbel-max at temperature T is ordinary sampling at T, but for a
fixed seed the honest answer is fixed, so a validator that replays the
committed tokens on its own model can score every position against its own
winner. The noise is integer murmur3, the same on every GPU and torch version.
"""
import torch

from gonka_poc.poc.gpu_random import _batched_murmur3_32, _murmur3_32

_SALT = 0x67A9E5


def tape_key(seed: int) -> int:
    """32-bit tape key of a 64-bit request seed."""
    seed &= 0xFFFFFFFFFFFFFFFF
    hi = torch.tensor([seed >> 32], dtype=torch.int64)
    return int(_murmur3_32(hi, (seed & 0xFFFFFFFF) ^ _SALT)[0])


def gumbel(keys: torch.Tensor, positions: torch.Tensor, vocab: int) -> torch.Tensor:
    """[R, vocab] float64 Gumbel noise for rows of (tape key, output position)."""
    row_seeds = _batched_murmur3_32(
        positions.to(torch.int32).view(-1, 1), keys.view(-1, 1))
    ids = torch.arange(vocab, dtype=torch.int32, device=keys.device)
    h = _batched_murmur3_32(ids.expand(keys.numel(), -1), row_seeds)
    u = (h.to(torch.float64) + 0.5) * 2.0**-32
    return -torch.log(-torch.log(u))


def truncate(logits: torch.Tensor, top_k: torch.Tensor,
             top_p: torch.Tensor) -> torch.Tensor:
    """-inf outside each row's top-k / top-p set (k <= 0, p >= 1: no cut).

    Top-p keeps the smallest most-probable prefix whose mass reaches p,
    measured after the top-k cut; the top token always stays.
    """
    srt, idx = logits.sort(dim=-1, descending=True)
    n = logits.shape[-1]
    rank = torch.arange(n, device=logits.device)
    k = torch.where(top_k > 0, top_k, n).unsqueeze(1)
    srt = srt.masked_fill(rank >= k, float("-inf"))
    probs = srt.softmax(dim=-1)
    above = probs.cumsum(dim=-1) - probs
    cut = (above >= top_p.unsqueeze(1)) & (top_p < 1.0).unsqueeze(1)
    srt = srt.masked_fill(cut, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter_(1, idx, srt)
