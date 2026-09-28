"""Scope vLLM's compile cache by the plugin knobs that change the traced forward.

vLLM keys its torch.compile cache (compiled graphs and the AOT artifacts) by its own
environment variables, the engine config and the *source* of the traced files. A
plugin knob that changes what the forward does at trace time (which PoC
interventions are ablated, debug prints) is none of those, so a host that
already holds a compiled graph loads it regardless of the knob: the process logs
one behaviour and runs another (03-04.09.2026, MiniMax on 1xB300: the ladder base
override rode a cached graph for a whole evening).

The fix at the right layer: when any such knob is set away from its default, the
plugin points ``VLLM_CACHE_ROOT`` at a sub-directory named by a hash of the knob
values, in every process (vLLM loads general plugins in each one), before anything
is compiled. Defaults keep the unscoped root, so production caches stay valid.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

_MARK = "gonka-poc-knobs-"

# knob -> default value (after normalisation). Extend when a new knob changes
# the traced forward; consensus parameters are NOT knobs (source constants).
_TRACED_KNOBS: dict[str, str] = {
    "POC_ABLATE": "",           # diagnostic ablation of PoC interventions
    "VLLM_POC_DEBUG_TP": "",    # debug prints inside the forward
}


def _normalise(name: str, raw: str | None) -> str:
    v = (raw or "").strip().lower()
    if name == "POC_ABLATE":
        return ",".join(sorted(x.strip() for x in v.split(",") if x.strip()))
    if name == "VLLM_POC_DEBUG_TP":
        return "1" if v == "1" else ""
    return v


def knob_signature(environ=None) -> tuple[dict[str, str], str | None]:
    """Return (normalised non-default knobs, signature) — signature None at defaults."""
    env = os.environ if environ is None else environ
    active = {k: _normalise(k, env.get(k)) for k in _TRACED_KNOBS}
    active = {k: v for k, v in active.items() if v != _TRACED_KNOBS[k]}
    if not active:
        return {}, None
    sig = hashlib.sha256(json.dumps(active, sort_keys=True).encode()).hexdigest()[:10]
    return active, sig


def scope_compile_cache(environ=None) -> str | None:
    """Point ``VLLM_CACHE_ROOT`` at a knob-specific sub-directory; idempotent.

    Returns the scoped root, or None when every knob is at its default (the
    root is then left untouched — including one scoped by a parent process)."""
    env = os.environ if environ is None else environ
    active, sig = knob_signature(env)
    root = env.get("VLLM_CACHE_ROOT") or os.path.expanduser("~/.cache/vllm")
    root = root.rstrip("/")
    base, last = os.path.split(root)
    if last.startswith(_MARK):
        if sig is None:
            return None
        if last == _MARK + sig:
            return root
        root = base                       # a parent scoped for different knobs
    if sig is None:
        return None
    scoped = os.path.join(root, _MARK + sig)
    env["VLLM_CACHE_ROOT"] = scoped
    logger.info("PoC: compile cache scoped to %s for knobs %s", scoped, active)
    return scoped
