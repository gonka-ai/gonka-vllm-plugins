"""In-model PoC transforms — module WRAPPERS, never forward hooks.

Ported from the 0.20 in-tree branch (``vllm/poc/native.py`` @ 5c1d09f55e92)
into the plugin, trimmed to what the plugin-side decode loop actually needs:

  * decoder-layer forward MONKEYPATCH — per-layer Householder reflection of
    hidden+residual on PoC rows (identity for non-PoC rows via the mask);
  * MoE gate forward MONKEYPATCH — REPLACES router logits with deterministic
    seeded logits on PoC rows (consensus: routing must not read the
    noise-prone hidden state);
  * ``PoCNativeState``   — address-stable buffers (reflection vectors, row
    mask, per-layer route bases, shared step buffer) updated IN PLACE, so a
    captured CUDA graph reads live values (design rule: eager is not an
    execution mode; the step function must be capture-ready from day one).

Dropped relative to 0.20 (their roles are covered elsewhere):
``PoCEmbeddingWrapper`` (the loop feeds ``inputs_embeds`` directly),
``PoCSnapWrapper`` (the loop snaps the returned hidden inside its own step
function), TP-rank divergence assertion (single-driver RPC path).

CONSENSUS: the arithmetic here (reflection formula, seeded-logit selection,
seed strings via ``gpu_random``) defines k-trajectories. Any change needs a
coordinated re-collection.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
from torch import nn

from .gpu_random import (
    expert_logits_from_base,
    generate_householder_vector,
    _seed_from_string,
    route_base_seed,
)

logger = logging.getLogger(__name__)


def _reflect(x: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked Householder: rows where mask is True -> x - 2*(x·v)*v; else x.

    0.20: native.py:60-66. Per-row independent, static-shape (no data-dependent
    control flow) — capture-safe.
    """
    dot = (x * v).sum(-1, keepdim=True)
    transformed = x - 2.0 * dot * v
    return torch.where(mask, transformed, x)


class PoCNativeState:
    """Address-stable per-model transform state (0.20: native.py:229-…).

    max_rows sizes every buffer: the largest token-row count a PoC forward can
    carry (prefill chunk: nonces*seq_len; decode step: nonces). Buffers are
    updated in place; a captured graph reads live values.
    """

    def __init__(self, num_layers: int, hidden_size: int, max_rows: int,
                 device: torch.device, dtype: torch.dtype):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.max_rows = max_rows
        self.device = device
        # Reflection vectors are BROADCAST [1, hidden]: one chunk carries one
        # block_hash, so a per-token-row [rows, hidden] buffer (12+ GiB at the
        # 128-nonce prefill chunk) is pure waste. per_nonce_reflection would
        # need per-row vectors — deferred (not in the consensus configuration), guarded
        # in set_rows.
        self.vectors: List[torch.Tensor] = [
            torch.zeros(1, hidden_size, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.mask = torch.zeros(max_rows, 1, device=device, dtype=torch.bool)
        # Synthetic-embedding buffer: the compiled path is entered with the
        # ENGINE signature (input_ids tensor, inputs_embeds None) and the
        # embed_tokens patch swaps in these rows by mask — the 0.20
        # PoCEmbeddingWrapper role (dropping the wrapper outright proved
        # wrong: feeding inputs_embeds from outside forces the eager path).
        self.embeds = torch.zeros(max_rows, hidden_size, device=device,
                                  dtype=dtype)
        self.route_step = torch.zeros(max_rows, dtype=torch.int64, device=device)
        self._route_base: List[torch.Tensor] = []
        self.router_meta: List[tuple] = []
        self._hh_cache: dict = {}
        self._rows_key = None
        self._route_key = None

    # -- reflection vectors ------------------------------------------------
    def _hh_vectors(self, block_hash: str, nonce: Optional[int], dtype):
        key = (block_hash, nonce)
        vs = self._hh_cache.get(key)
        if vs is None:
            if len(self._hh_cache) > 64:
                self._hh_cache.clear()
            suffix = "" if nonce is None else f"_nonce{nonce}"
            vs = [
                generate_householder_vector(
                    f"{block_hash}{suffix}_layer_{i}_householder",
                    self.hidden_size, self.device).to(dtype)
                for i in range(self.num_layers)
            ]
            self._hh_cache[key] = vs
        return vs

    def set_rows(self, block_hash: Optional[str], n_rows: int,
                 refl_nonce: Optional[int] = None,
                 per_nonce: bool = False) -> None:
        """Broadcast reflection vectors for ONE block_hash + mask first n_rows.

        block_hash None -> full identity (mask off). per_nonce reflection needs
        per-row vectors — not implemented in this revision (the consensus
        configuration has it
        off); fail loud rather than silently mis-derive.
        """
        if per_nonce:
            raise NotImplementedError(
                "per_nonce_reflection needs per-row reflection buffers; "
                "off in the consensus configuration")
        key = (block_hash, refl_nonce, n_rows)
        if key == self._rows_key:
            return
        if block_hash is None:
            self.mask.zero_()
        else:
            self.mask[:n_rows].fill_(True)
            self.mask[n_rows:].zero_()
            dtype = self.vectors[0].dtype
            vs = self._hh_vectors(block_hash, refl_nonce, dtype)
            for li in range(self.num_layers):
                self.vectors[li].copy_(vs[li].unsqueeze(0))
        self._rows_key = key

    # -- seeded routing ----------------------------------------------------
    def set_routing(self, block_hash: str, nonces: List[int],
                    tokens_per_nonce: int, step: int) -> None:
        """Refresh seeded-router state for one chunk.

        Row layout: nonce-major, ``tokens_per_nonce`` consecutive rows per
        nonce (prefill: seq_len rows/nonce, decode step: 1 row/nonce). Base
        seed sha256 is hashed once per (hash, nonce, layer) and expanded on
        device; the step folds in on-GPU inside the wrapper.
        """
        if not self._route_base:
            return
        n = len(nonces) * tokens_per_nonce
        base_key = (block_hash, tuple(nonces), tokens_per_nonce)
        if base_key != self._route_key:
            # ``li`` is the GLOBAL decoder-layer index (deliberately the
            # same numbering as the Householder seeds). For all-MoE stacks
            # (MiniMax) it equals the discovery order, so bits are unchanged;
            # for dense-prefixed stacks (DeepSeek family) it pins the seed to
            # the physical layer, not to "which MoE in order".
            for li, buf in self._route_base:
                vals = torch.tensor(
                    [_seed_from_string(route_base_seed(block_hash, nz, li))
                     for nz in nonces],
                    dtype=torch.int64, device=self.device)
                buf[:n].copy_(vals.repeat_interleave(tokens_per_nonce))
            self._route_key = base_key
        self.route_step[:n].fill_(int(step))

    def set_embeds(self, x: torch.Tensor) -> None:
        """Stage synthetic embedding rows for the next PoC forward (in place;
        address-stable — a captured graph reads live values)."""
        self.embeds[:x.shape[0]].copy_(x)

    def clear(self) -> None:
        """Identity for everything (defensive; engine paths never see us)."""
        self.mask.zero_()
        self._rows_key = None


def _find_decoder_layers(model: nn.Module) -> nn.Module:
    """Locate the decoder layer owner generically: the module whose ``.layers``
    ModuleList is the LONGEST one in the tree (the transformer stack). The
    previous "deepest" heuristic could latch onto a nested short list (seen on
    MiniMax-M2.7: 28 inner modules wrapped instead of the 62 decoder layers,
    which also left every MoE gate undiscovered)."""
    best, best_name = None, None
    for name, m in model.named_modules():
        layers = getattr(m, "layers", None)
        if isinstance(layers, nn.ModuleList) and (
                best is None or len(layers) > len(best.layers)):
            best, best_name = m, name
    if best is None:
        raise RuntimeError("PoC native: no decoder .layers ModuleList found")
    logger.info("PoC native: decoder stack at '%s' (%d layers)",
                best_name or "<root>", len(best.layers))
    return best


def attach_native_poc(model: nn.Module, hidden_size: int, max_rows: int,
                      device, dtype,
                      route_window: Optional[int] = None) -> "PoCNativeState":
    """Bake PoC transforms by MONKEYPATCHING the bound ``forward`` of each
    decoder layer and each MoE gate — the modules, parameter names and the
    ``@support_torch_compile`` signature inspection stay intact (wrapping in a
    new nn.Module renamed params to ``.inner.*`` and broke both weight loading
    and compile). The patched forward reads live state buffers, so it captures
    into the compiled graph exactly like the 0.20 attach-before-compile path.
    Idempotent; chat rows (mask False) are exact identity.
    """
    if getattr(model, "_poc_native_state", None) is not None:
        return model._poc_native_state

    owner = _find_decoder_layers(model)
    layers = list(owner.layers)
    state = PoCNativeState(len(layers), hidden_size, max_rows, device, dtype)

    emb = getattr(owner, "embed_tokens", None)
    if emb is not None:
        _patch_embed_forward(emb, state)
        state.has_embed_patch = True
    else:
        state.has_embed_patch = False
        logger.warning("PoC native: no embed_tokens on decoder owner — "
                       "compiled entry unavailable, eager fallback only")

    for li, layer in enumerate(layers):
        _patch_layer_forward(layer, state, li)
        moe = _find_layer_moe(layer)
        if moe is None:
            continue
        dims = _moe_dims(moe, getattr(model, "config", None))
        if dims is None:
            continue
        n_exp, top_k, n_group, topk_group = dims
        route_base = torch.zeros(max_rows, dtype=torch.int64, device=device)
        state._route_base.append((li, route_base))
        state.router_meta.append((n_exp, top_k, n_group, topk_group))
        _patch_gate_forward(moe, state, route_base, n_exp, top_k,
                            n_group, topk_group)

    logger.info("PoC native attached: %d layers patched, %d MoE gates seeded, "
                "embed patch %s, route window %d", len(layers),
                len(state.router_meta), state.has_embed_patch, route_window)
    # Model-agnostic window default (call agreement 2026-08-19): explicit
    # value (env POC_ROUTE_WINDOW) wins; otherwise FULL SCATTER — window =
    # n_experts of THIS model (window >= n_experts selects the legacy
    # full-scatter consensus path; on MiniMax's 256 experts this is exactly
    # the shipped-golden value 256, bit-identical behaviour). Dense models
    # (no MoE) get 0 — the window is routing-only and never read.
    if route_window is None:
        route_window = state.router_meta[0][0] if state.router_meta else 0
    state.route_window = int(route_window)
    from .gpu_random import set_route_window
    set_route_window(state.route_window)

    model._poc_native_state = state
    return state


def _patch_embed_forward(emb: nn.Module, state: "PoCNativeState"):
    """CLASS-level forward patch: dynamo inlines ``type(mod).forward`` and
    IGNORES instance monkeypatches (verified: instance-patched graphs carried
    none of the PoC transforms), so every patch here goes on the class and
    reads per-instance state attributes (dynamo specializes on them)."""
    cls = type(emb)
    if not getattr(cls, "_poc_class_patched", False):
        orig = cls.forward

        def forward(self, input_ids):
            out = orig(self, input_ids)
            st = getattr(self, "_poc_state", None)
            if st is None:
                return out
            n = out.shape[0]
            mask = st.mask
            if n > mask.shape[0]:
                return out  # chat batch beyond PoC buffers: untouched
            return torch.where(mask[:n], st.embeds[:n].to(out.dtype), out)

        cls.forward = forward
        cls._poc_class_patched = True
    emb._poc_state = state


def _find_layer_moe(layer: nn.Module):
    return next(
        (m for m in layer.modules()
         if hasattr(m, "gate") and hasattr(m, "experts")
         and not hasattr(getattr(m, "gate"), "_poc_state")),
        None)


def _moe_dims(moe: nn.Module, hf_config) -> Optional[tuple]:
    """(n_experts, top_k) across vLLM minors: FusedMoE attrs moved between
    releases, so resolve through a chain and fail LOUD, never silently."""
    exp = moe.experts
    n_exp = getattr(exp, "global_num_experts", None)
    top_k = getattr(exp, "top_k", None)
    mc = getattr(exp, "moe_config", None) or getattr(exp, "moe", None)
    if n_exp is None and mc is not None:
        n_exp = getattr(mc, "global_num_experts", None) or getattr(
            mc, "num_experts", None)
    if top_k is None and mc is not None:
        top_k = getattr(mc, "experts_per_token", None) or getattr(
            mc, "top_k", None)
    if n_exp is None:
        w = getattr(moe.gate, "weight", None)
        if w is not None and getattr(w, "ndim", 0) == 2:
            n_exp = w.shape[0]
    if top_k is None and hf_config is not None:
        top_k = getattr(hf_config, "num_experts_per_tok", None)
    if n_exp is None or top_k is None:
        have = sorted(k for k in vars(exp).keys() if not k.startswith("_"))
        logger.error(
            "PoC native: MoE found but dims unresolved (n_exp=%s top_k=%s); "
            "experts type %s attrs: %s", n_exp, top_k,
            type(exp).__name__, have[:40])
        return None
    # Grouped routers (DeepSeek family): n_group/topk_group from the experts
    # module, falling back to the HF config; 1/1 = flat router (MiniMax).
    n_group = getattr(exp, "num_expert_group", None)
    topk_group = getattr(exp, "topk_group", None)
    if n_group is None and mc is not None:
        n_group = getattr(mc, "num_expert_group", None) or getattr(
            mc, "n_group", None)
    if topk_group is None and mc is not None:
        topk_group = getattr(mc, "topk_group", None)
    if n_group is None and hf_config is not None:
        n_group = getattr(hf_config, "n_group", None)
    if topk_group is None and hf_config is not None:
        topk_group = getattr(hf_config, "topk_group", None)
    return int(n_exp), int(top_k), int(n_group or 1), int(topk_group or 1)


def _patch_layer_forward(layer: nn.Module, state: "PoCNativeState", li: int):
    """Reflect the layer's (hidden, residual) output on PoC rows.

    Class-level patch (see _patch_embed_forward); per-layer index and state
    live on the instance.
    """
    cls = type(layer)
    if not getattr(cls, "_poc_class_patched", False):
        orig = cls.forward

        def forward(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            st = getattr(self, "_poc_state", None)
            if st is None:
                return out
            vbuf, mask = st.vectors[self._poc_li], st.mask
            first = out[0] if isinstance(out, tuple) else out
            if first.shape[0] > mask.shape[0]:
                return out  # chat batch beyond PoC buffers: untouched
            if isinstance(out, tuple) and len(out) == 2:
                hidden, residual = out
                n = hidden.shape[0]
                hidden = _reflect(hidden, vbuf, mask[:n])
                if residual is not None:
                    residual = _reflect(residual, vbuf, mask[:n])
                return hidden, residual
            hidden = first
            n = hidden.shape[0]
            hidden = _reflect(hidden, vbuf, mask[:n])
            return (hidden, *out[1:]) if isinstance(out, tuple) else hidden

        cls.forward = forward
        cls._poc_class_patched = True
    layer._poc_state = state
    layer._poc_li = li


def _patch_gate_forward(moe: nn.Module, state: "PoCNativeState",
                        route_base: torch.Tensor, n_experts: int, top_k: int,
                        n_group: int = 1, topk_group: int = 1):
    """Override the MoE gate's logits with seeded logits on PoC rows.

    Class-level patch (see _patch_embed_forward); the per-gate route-base
    buffer and dims live on the gate instance.
    """
    gate = moe.gate
    cls = type(gate)
    if not getattr(cls, "_poc_class_patched", False):
        orig = cls.forward

        def forward(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            st = getattr(self, "_poc_state", None)
            if st is None:
                return out
            logits = out[0] if isinstance(out, tuple) else out
            n = logits.shape[0]
            if n > st.mask.shape[0]:
                return out  # chat batch beyond PoC buffers: untouched
            ne, tk, ng, tkg = self._poc_dims
            forced = expert_logits_from_base(
                self._poc_base[:n], st.route_step[:n], ne, tk,
                logits.device, n_group=ng, topk_group=tkg).to(logits.dtype)
            logits = torch.where(st.mask[:n], forced, logits)
            return (logits, *out[1:]) if isinstance(out, tuple) else logits

        cls.forward = forward
        cls._poc_class_patched = True
    gate._poc_state = state
    gate._poc_base = route_base
    gate._poc_dims = (int(n_experts), int(top_k), int(n_group),
                      int(topk_group))
