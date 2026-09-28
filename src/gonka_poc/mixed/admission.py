# SPDX-License-Identifier: Apache-2.0
"""The one thing the scheduler needs to know about a PoC row.

PoC nonces are ordinary engine requests: vLLM's scheduler admits them, budgets
them, allocates their KV and preempts them exactly as it does chat; their
concurrency is the engine's ``max_num_seqs``. No per-step PoC policy lives in
the engine any more —
the row cap, the token share, the KV headroom gate, the stall hand-off, the
decode-only isolation and the one-step hold of a nonce's first decode row were
all removed on 2026-09-05 after they measured as the cause of the behaviour they
were patching (see ADR-0017).

What remains is structural, not policy: a PoC row's prompt is a synthetic
``seq_len``-token sequence whose embeddings the plugin generates itself, and
the decode chain starts from the sphere snap of the LAST prompt token. The
input builder generates the whole prompt in one go and the snap reads the
last row, so a PoC prefill must land in ONE step — never chunked. After the
prefill, a decode row schedules one token per step (PoC rows produce no
sampled tokens, so vLLM's own arithmetic would give 0).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.request import Request


def poc_step_tokens(request: "Request", num_new_tokens: int,
                    token_budget: int) -> int:
    """Tokens to schedule for ``request`` this step.

    Chat rows pass through unchanged. A PoC row at prefill takes its whole
    ``seq_len`` if that fits the remaining step budget and 0 otherwise (the
    scheduler then leaves it for a later step); a decoding PoC row takes 1.
    Pure (unit-testable)."""
    pp = getattr(request, "poc_params", None)
    if pp is None:
        return num_new_tokens
    if request.num_computed_tokens < pp.seq_len:
        return pp.seq_len if pp.seq_len <= token_budget else 0
    return 1
