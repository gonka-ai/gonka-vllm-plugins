# SPDX-License-Identifier: Apache-2.0
"""Model-runner side of mixed chat+PoC execution (counterpart of admission.py).

One bridge object per GPUModelRunner. Every hook is a no-op when the step holds
no PoC rows, so the pure-chat path is untouched. Embeds/routing/reflection are
applied IN-MODEL via native.py state (never through ``inputs_embeds`` — on 0.25
external embeds force the engine off the compiled path).

Runner anchors (5):
  load()        - load_model: attach native PoC state to the model
  pre_step()    - execute_model entry: decode-state slots for PoC rows
  pre_forward() - after _preprocess: row mask + per-row routing into native
  filter_rows() - logits: PoC rows excluded from sampling
  extract()     - post-forward: k-snap/chain from hidden states -> poc_outputs
"""

from typing import TYPE_CHECKING, Any

import torch



from vllm.logger import init_logger
from gonka_poc.poc.decode_random import pinned_to_device
from gonka_poc.mixed import runtime as mixed_decode

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import PoCOutput

logger = init_logger(__name__)


class _PoCRequestView:
    """Runner-agnostic per-request view, built from SchedulerOutput."""

    __slots__ = ("req_id", "poc_params", "num_computed_tokens")

    def __init__(self, req_id, poc_params, num_computed_tokens) -> None:
        self.req_id = req_id
        self.poc_params = poc_params
        self.num_computed_tokens = num_computed_tokens


def _poc_diag() -> bool:
    import os
    return os.environ.get("POC_DIAG", "") == "1"


class PoCRunnerBridge:
    def __init__(self, runner) -> None:
        self.runner = runner
        self.native = None
        self._step: dict[str, Any] | None = None  # per-step mixed-batch info
        self._reqs: dict[str, _PoCRequestView] = {}

    # ------------------------------------------------------------- load_model
    def load(self, model: torch.nn.Module) -> None:
        """Attach the in-model PoC transforms (0.20: gpu_model_runner load_model).

        Fails LOUD: without these the forward still emits hidden states and the
        pipeline still produces plausible trajectories — they are simply not the
        consensus computation. A silent fallback here is unacceptable.
        """
        # Attached unconditionally: the wrappers are mask-gated and their
        # mask-False branch is exact identity, so a prefill-only request --
        # which runs over collective_rpc and never sets the mask -- still sees
        # an untouched model. That is what lets one process serve both schemes
        # and lets the chain pick per request instead of per launch.
        from gonka_poc.mixed.native import attach_native_poc

        runner = self.runner
        inner = getattr(model, "model", model)
        layers = getattr(inner, "layers", None)
        if layers is None:
            raise RuntimeError(
                f"PoC: cannot attach native transforms, {type(model).__name__} "
                "exposes no decoder layer list")
        self.native = attach_native_poc(
            model, layers, inner, runner.max_num_tokens,
            runner.model_config.get_hidden_size(), runner.device, runner.dtype,
            hf_config=getattr(runner.model_config, "hf_config", None),
            # distinct (block_hash, nonce) draws in one batch <= rows in flight
            max_groups=int(getattr(getattr(runner, "scheduler_config", None),
                                   "max_num_seqs", 0) or 0) or None,
        )
        # mixed_decode reads the state off the runner (0.20 contract).
        runner._poc_native = self.native

    # --------------------------------------------------------- per-step hooks
    def pre_step(self, scheduler_output: "SchedulerOutput") -> None:
        # The bridge keeps its own registry of PoC rows: a row is registered
        # the step it first appears (NewRequestData.poc_params) and forgotten
        # when the scheduler reports it finished. The rows of THIS step are
        # the registered ones among what the scheduler scheduled (new or
        # cached) — no PoC-specific field on SchedulerOutput is needed.
        for r in scheduler_output.scheduled_new_reqs:
            if r.poc_params is not None:
                self._reqs[r.req_id] = _PoCRequestView(
                    r.req_id, r.poc_params, r.num_computed_tokens)
        for rid in scheduler_output.finished_req_ids:
            self._reqs.pop(rid, None)
        if not self._reqs:
            self._step = None
            return

        poc_req_ids: set[str] = set()
        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            view = self._reqs.get(rid)
            if view is not None:
                view.num_computed_tokens = cached.num_computed_tokens[i]
                poc_req_ids.add(rid)
        for r in scheduler_output.scheduled_new_reqs:
            if r.req_id in self._reqs:
                poc_req_ids.add(r.req_id)
        if not poc_req_ids:
            self._step = None
            return
        # Deterministic order: nonce, not set-iteration (PYTHONHASHSEED).
        poc_requests = sorted(
            (self._reqs[rid] for rid in poc_req_ids if rid in self._reqs),
            key=lambda req: req.poc_params.nonce,
        )
        if _poc_diag():
            import time
            torch.cuda.synchronize(); _t0 = time.monotonic()
            mixed_decode.setup_decode_poc(self.runner, poc_requests)
            torch.cuda.synchronize(); _dt = (time.monotonic() - _t0) * 1000
            if _dt > 30:
                logger.info("poc: pre_step setup_decode_poc %.0f ms (rows=%d, new=%d)",
                            _dt, len(poc_requests), len(scheduler_output.scheduled_new_reqs))
        else:
            mixed_decode.setup_decode_poc(self.runner, poc_requests)
        self._step = {
            "poc_req_ids": poc_req_ids,
            "poc_requests": poc_requests,
            "poc_metadata": None,
            "poc_position_mask": None,
        }

    def pre_forward(self, scheduler_output: "SchedulerOutput",
                    positions: torch.Tensor, num_total_tokens: int,
                    batch_view: tuple | None = None) -> None:
        if self._step is None:
            if self.native is not None:
                self.native.set_mask(None)
            return
        if self.native is None:
            raise RuntimeError(
                "PoC step scheduled without native transforms attached — "
                "artifacts would not be the consensus computation")
        _diag = _poc_diag()
        if _diag:
            import time
            torch.cuda.synchronize()
            _t0 = time.monotonic()
        embeds, _positions, mask, metadata = (
            mixed_decode.build_unified_mixed_batch_inputs(
                self.runner, scheduler_output, None, None, positions,
                self._step["poc_req_ids"], num_total_tokens,
                batch_view if batch_view is not None else self._batch_view(),
                self._reqs,
            )
        )
        self._step["poc_metadata"] = metadata
        self._step["poc_position_mask"] = mask
        if self.native is not None and embeds is not None:
            self.native.set_embeds(embeds)
            self.native.set_mask(mask)
            if metadata:
                # Per-row block_hash seeds the reflection vectors; per-row
                # (hash, nonce, step) drives mandatory seeded routing (0.20:
                # gpu_model_runner post-build glue, verbatim). Without this the
                # reflection vectors stay zero (identity) and trajectories
                # diverge from 0.20 -- caught by cross-version parity.
                n_rows = embeds.shape[0]
                row_hashes = [None] * n_rows
                row_nonces = [0] * n_rows
                row_steps = [0] * n_rows
                row_refl_nonces = [None] * n_rows
                for meta in metadata:
                    pp = meta["poc_params"]
                    stp = meta.get("decode_step", 0)
                    for r in range(meta["start_idx"],
                                   meta["start_idx"] + meta["length"]):
                        if r < n_rows:
                            row_hashes[r] = pp.block_hash
                            row_nonces[r] = pp.nonce
                            row_steps[r] = stp
                            if getattr(pp, "per_nonce_reflection", False):
                                row_refl_nonces[r] = pp.nonce
                if _diag:
                    torch.cuda.synchronize(); _t1 = time.monotonic()
                self.native.set_row_block_hashes(row_hashes, row_refl_nonces)
                if _diag:
                    torch.cuda.synchronize(); _t2 = time.monotonic()
                self.native.set_routing(row_hashes, row_nonces, row_steps)
                if _diag:
                    torch.cuda.synchronize(); _t3 = time.monotonic()
                    self._diag_ev0 = torch.cuda.Event(enable_timing=True)
                    self._diag_ev0.record()
                    self._diag_host0 = time.monotonic()
                    self._diag_tokens = int(num_total_tokens)
                    tot = (_t3 - _t0) * 1000
                    if tot > 50:
                        n_pre = sum(1 for m in metadata if m.get("length", 1) > 1)
                        logger.info("poc: pre_forward %.0f ms (build=%.0f, hashes=%.0f,"
                                    " routing=%.0f; rows=%d, prefill_rows=%d)",
                                    tot, (_t1 - _t0) * 1000, (_t2 - _t1) * 1000,
                                    (_t3 - _t2) * 1000, n_rows, n_pre)

    def _batch_view(self) -> tuple:
        ib = self.runner.input_batch
        return (ib.num_reqs, ib.req_ids)

    def extract(self, hidden_states: torch.Tensor
                ) -> "dict[str, PoCOutput] | None":
        if self._step is None or self._step.get("poc_metadata") is None:
            return None
        if _poc_diag() and getattr(self, "_diag_ev0", None) is not None:
            import time
            ev1 = torch.cuda.Event(enable_timing=True); ev1.record(); ev1.synchronize()
            gpu_ms = self._diag_ev0.elapsed_time(ev1)
            host_ms = (time.monotonic() - self._diag_host0) * 1000
            if gpu_ms > 150 or host_ms > 150:
                logger.info("poc: fwd gpu=%.0f ms host=%.0f ms, tokens=%d, poc_rows=%d",
                            gpu_ms, host_ms, self._diag_tokens,
                            len(self._step.get("poc_metadata") or ()))
            self._diag_ev0 = None
        out = mixed_decode.process_poc_outputs_from_hidden(
            self.runner, hidden_states, self._step["poc_metadata"])
        if out:
            mgr = mixed_decode.get_decode_manager(self.runner)
            for rid in out:
                mgr.free(rid)
        # Leave the row mask cleared. The prefill scheme runs its own forward
        # over collective_rpc and never passes through this bridge, so a mask
        # left set here would still be set when that forward runs -- the
        # wrappers would transform its hidden state and its artifacts would
        # stop matching the shipped image. The decode side clears it, because
        # the decode side is what sets it; the prefill files stay untouched.
        if self.native is not None:
            self.native.set_mask(None)
        return out

    # ------------------------------------------------------ sampling exclusion
    def mixed_active(self) -> bool:
        return self._step is not None

    def _chat_rows(self) -> list[int]:
        rows = self._step.get("chat_rows")
        if rows is None:
            poc_ids = self._step["poc_req_ids"]
            rows = [
                i for i, rid in enumerate(
                    self.runner.input_batch.req_ids[
                        : self.runner.input_batch.num_reqs])
                if rid not in poc_ids
            ]
            self._step["chat_rows"] = rows
            self._step["num_reqs_snapshot"] = self.runner.input_batch.num_reqs
        return rows

    def compute_logits(self, sample_hidden_states: torch.Tensor
                       ) -> torch.Tensor | None:
        # PoC scores hidden states, not logits — never run the LM head for
        # PoC rows (0.20: _compute_logits_with_poc_filter).
        chat_rows = self._chat_rows()
        if not chat_rows:
            return None
        idx = pinned_to_device(chat_rows, torch.long, sample_hidden_states.device)
        return self.runner.model.compute_logits(sample_hidden_states[idx])

    def sample_chat_rows(self, logits, sampling_metadata):
        # Sample CHAT ROWS ONLY, scatter back into a full natural-order tensor
        # (PoC slots = 0, never read). 0.20: _sample mixed branch, verbatim.
        from vllm.v1.outputs import SamplerOutput

        runner = self.runner
        chat_rows = self._chat_rows()
        n_full = self._step["num_reqs_snapshot"]
        if not chat_rows:
            return SamplerOutput(
                sampled_token_ids=torch.zeros(
                    (n_full, 1), dtype=torch.int32, device=runner.device),
                logprobs_tensors=None,
            )
        chat_sm = mixed_decode.slice_sampling_metadata(
            sampling_metadata, chat_rows, runner.device)
        chat_out = runner.sampler(logits=logits, sampling_metadata=chat_sm)
        idx = pinned_to_device(chat_rows, torch.long, runner.device)
        full = torch.zeros(
            (n_full, chat_out.sampled_token_ids.shape[1]),
            dtype=chat_out.sampled_token_ids.dtype, device=runner.device)
        full[idx] = chat_out.sampled_token_ids
        return SamplerOutput(
            sampled_token_ids=full,
            logprobs_tensors=self._scatter_logprobs(
                chat_out.logprobs_tensors, idx, n_full),
        )

    @staticmethod
    def _scatter_logprobs(lp, idx, n_full: int):
        """Lift chat-only logprob rows into the same natural order as the tokens.

        ``sampled_token_ids`` is scattered to full batch width, but the sampler
        saw chat rows only, so its logprobs come back compacted 0..len(chat)-1.
        The runner reads the two in parallel by request index
        (``output.sampled_token_ids`` / ``output.logprobs``), so leaving them in
        different row spaces hands a chat request the logprobs of whichever row
        follows a PoC row. Same disease as the renumbering incident the natural
        -order rule exists to prevent, just on the other tensor.

        PoC rows get zero rows that nothing reads: PoC carries no sampling
        params and never asks for logprobs.
        """
        from vllm.v1.outputs import LogprobsTensors

        if lp is None:
            return None
        n_rows = lp.logprobs.shape[0]
        if n_rows != idx.shape[0]:
            # >1 row per request (spec decode) needs a per-request row map,
            # which this path does not carry. Refuse rather than mis-index.
            raise RuntimeError(
                f"mixed PoC: cannot place {n_rows} logprob rows over "
                f"{idx.shape[0]} chat rows; per-request row counts differ")

        def _lift(t):
            if t is None or t.numel() == 0:
                return t
            out = torch.zeros((n_full, *t.shape[1:]),
                              dtype=t.dtype, device=t.device)
            out[idx] = t
            return out

        return LogprobsTensors(
            _lift(lp.logprob_token_ids),
            _lift(lp.logprobs),
            _lift(lp.selected_token_ranks),
        )
