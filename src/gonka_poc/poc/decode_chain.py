"""Pure per-step decode-chain rules — the consensus arithmetic of a decode
nonce, extracted from the 0.20 in-tree branch (mixed_decode.py) WITHOUT the
engine orchestration around it.

Everything here is a pure tensor function: no engine state, no host branches,
address-stable — safe inside a captured CUDA graph (design rule: eager is
not an execution mode).

Ported bit-for-bit; sources cited per function.
"""
from typing import Optional

import torch


def count_mismatch(
    k: torch.Tensor, ref: torch.Tensor, margin: torch.Tensor, tau: float
) -> torch.Tensor:
    """Per-row mismatch indicator for one step  →  int64 [batch].

    0.20: mixed_decode.py:543 / :583 —
        (k != ref) & (k >= 0) & (margin >= tau)
    A non-finite snap (k == -1, the compute-fault sentinel) never counts as a
    mismatch; a low-margin disagreement (boundary jitter) is gated by ``tau``
    (VLLM_POC_MARGIN_TAU semantics; tau == 0 disables the gate).
    """
    return ((k != ref) & (k >= 0) & (margin >= tau)).to(torch.int64)


def next_prev_k(
    k: torch.Tensor, ref: Optional[torch.Tensor]
) -> torch.Tensor:
    """Chain seed for the NEXT step  →  int64 [batch].

    Teacher forcing (0.20: mixed_decode.py:544-546): during validation the
    ENFORCED (reference) k seeds the next step, so both sides walk one
    trajectory and every step stays an independent check; during generation
    the model's own snap chains.
    """
    return ref if ref is not None else k


def keep_q_step(step: int, debug: bool, va_on: bool, va_steps: int) -> bool:
    """Which decode steps retain their pre-snap slice for emission.

    0.20: mixed_decode.py:55-58 — all under debug, the leading ``va_steps``
    window under poc_vector_artifacts. Pure.
    """
    return debug or (va_on and step <= va_steps)
