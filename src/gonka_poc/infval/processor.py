"""Logits processor that samples on, and scores replays against, the tape.

A request opts in with ``vllm_xargs.gonka_tape`` and its ordinary ``seed`` is
the tape seed. The processor is declared argmax-invariant so that it runs in
the random-sampling slot, where a row has had penalties, temperature and min_p
applied; it applies top-k / top-p itself, so the tape replaces only the final
random draw. Greedy rows need no tape and never reach this slot.

A generation row is forced to its tape winner: its processed logprobs become
one-hot, raw logprobs are untouched. A replay row (enforced tokens, tagged by
the infval route) keeps the enforced sample; each committed token is scored
against this model's winner and kept per position until ``take``.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from gonka_poc._compat import current as _compat_current
from gonka_poc.infval.tape import gumbel, tape_key, truncate
from gonka_poc.poc.decode_random import pinned_to_device

LogitsProcessor, process_dict_updates = _compat_current().logits_processor_api()

TAPE_ARG = "gonka_tape"
TAG_ARG = "gonka_tape_tag"
GREEDY_ARG = "gonka_tape_greedy"

_CHUNK_ROWS = 16
_MAX_TAGS = 512
# tag -> {output position: (mismatch, gap)}
_records: Dict[str, Dict[int, Tuple[bool, float]]] = {}


@dataclass
class _Row:
    key: int
    top_k: int
    top_p: float
    output: List[int]
    committed: Optional[List[int]] = None
    tag: Optional[str] = None
    greedy: bool = False


def _new_row(params, prompt_ids, output_ids) -> Optional[_Row]:
    args = params.extra_args or {}
    if not args.get(TAPE_ARG):
        return None
    row = _Row(tape_key(params.seed), params.top_k, params.top_p, output_ids)
    committed = getattr(params, "enforced_token_ids", None)
    if args.get(TAG_ARG) and committed:
        row.committed, row.tag = committed, str(args[TAG_ARG])
        row.greedy = bool(args.get(GREEDY_ARG))
    elif params.temperature < 1e-5:
        return None
    return row


class TapeLogitsProcessor(LogitsProcessor):

    @classmethod
    def validate_params(cls, params):
        if (params.extra_args or {}).get(TAPE_ARG) and params.seed is None:
            raise ValueError(f"{TAPE_ARG} needs a seed")

    def __init__(self, vllm_config, device, is_pin_memory):
        self.rows: Dict[int, _Row] = {}

    def is_argmax_invariant(self) -> bool:
        return True

    def update_state(self, batch_update) -> None:
        process_dict_updates(self.rows, batch_update, _new_row)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        gen = [i for i, r in self.rows.items() if r.tag is None]
        rep = [i for i, r in self.rows.items()
               if r.tag is not None and len(r.output) < len(r.committed)]
        for lo in range(0, len(gen), _CHUNK_ROWS):
            self._force(logits, gen[lo:lo + _CHUNK_ROWS])
        for lo in range(0, len(rep), _CHUNK_ROWS):
            self._score(logits, rep[lo:lo + _CHUNK_ROWS])
        return logits

    def _tape(self, logits, idx):
        """Rows, their logits and noise, and their tape scores over the cut."""
        rows = [self.rows[i] for i in idx]
        dev = logits.device
        lg = logits[pinned_to_device(idx, torch.long, dev)]
        noise = gumbel(pinned_to_device([r.key for r in rows], torch.int64, dev),
                       pinned_to_device([len(r.output) for r in rows],
                                        torch.int32, dev),
                       lg.shape[-1])
        cut = truncate(lg,
                       pinned_to_device([r.top_k for r in rows], torch.int64, dev),
                       pinned_to_device([r.top_p for r in rows], torch.float32, dev))
        return rows, lg.double(), noise, cut.double() + noise

    def _force(self, logits, idx):
        *_, s = self._tape(logits, idx)
        rows_t = pinned_to_device(idx, torch.long, logits.device)
        x = s.argmax(dim=-1)
        logits[rows_t] = float("-inf")
        logits[rows_t, x] = 0.0

    def _score(self, logits, idx):
        rows, lg, noise, s = self._tape(logits, idx)
        greedy = pinned_to_device([r.greedy for r in rows], torch.bool,
                                  logits.device).unsqueeze(1)
        s = torch.where(greedy, lg, s)
        full = torch.where(greedy, lg, lg + noise)
        x = pinned_to_device([r.committed[len(r.output)] for r in rows],
                             torch.long, logits.device).unsqueeze(1)
        z = s.argmax(dim=-1, keepdim=True)
        # A committed token outside this model's top-k/top-p cut is scored by
        # its uncut tape score: boundary jitter gives no gap, a masked token 1e9.
        sx = torch.where(torch.isinf(s.gather(1, x)), full.gather(1, x),
                         s.gather(1, x))
        gap = torch.nan_to_num(s.gather(1, z) - sx, nan=0.0, posinf=1e9)
        stats = torch.cat([(x != z).double(), gap.clamp(min=0.0)], 1).tolist()
        for r, (mismatch, g) in zip(rows, stats):
            _records.setdefault(r.tag, {})[len(r.output)] = (bool(mismatch), g)
        while len(_records) > _MAX_TAGS:
            _records.pop(next(iter(_records)))


def take(tags: List[str]) -> Dict[str, Dict[str, Any]]:
    """Pop the per-response signals of finished replays."""
    out = {}
    for tag in tags:
        rec = _records.pop(tag, None)
        if rec is None:
            continue
        gaps = [g for mismatch, g in rec.values() if mismatch]
        out[tag] = {"steps": len(rec), "count": len(gaps),
                    "max_gap": max(gaps, default=0.0), "sum_gap": sum(gaps)}
    return out
