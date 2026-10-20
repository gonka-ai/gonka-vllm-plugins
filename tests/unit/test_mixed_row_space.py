from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("vllm")
import torch  # noqa: E402

from vllm.poc.poc_params import PoCParams  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.core.sched.output import GrammarOutput  # noqa: E402
from vllm.v1.sample.logits_processor import LogitsProcessors  # noqa: E402
from vllm.v1.sample.logits_processor.builtin import (  # noqa: E402
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.sample.sampler import Sampler  # noqa: E402
from vllm.v1.structured_output.utils import apply_grammar_bitmask  # noqa: E402
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch  # noqa: E402

from gonka_poc.mixed.bridge import PoCRunnerBridge  # noqa: E402

VOCAB = 64
HIDDEN = 16
MAX_REQS = 8
THINK_START = 1
THINK_END = 2
ALLOWED = 42
CPU = torch.device("cpu")


class _Model:
    def __init__(self):
        gen = torch.Generator().manual_seed(0)
        self.weight = torch.randn(VOCAB, HIDDEN, generator=gen)

    def compute_logits(self, hidden):
        return hidden @ self.weight.T


def _input_batch():
    cfg = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=MAX_REQS))
    logitsprocs = LogitsProcessors([
        MinPLogitsProcessor(cfg, CPU, False),
        LogitBiasLogitsProcessor(cfg, CPU, False),
        MinTokensLogitsProcessor(cfg, CPU, False),
    ])
    return InputBatch(
        max_num_reqs=MAX_REQS,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=CPU,
        vocab_size=VOCAB,
        block_sizes=[16],
        kernel_block_sizes=[16],
        max_num_blocks_per_req=[4],
        logitsprocs=logitsprocs,
        reasoning_config=SimpleNamespace(
            reasoning_start_token_ids=[THINK_START],
            reasoning_end_token_ids=[THINK_END],
        ),
    )


def _request(req_id, spec):
    if spec is None:
        return CachedRequestState(
            req_id=req_id,
            prompt_token_ids=[0] * 8,
            mm_features=[],
            sampling_params=None,
            block_ids=([],),
            generator=None,
            num_computed_tokens=0,
            output_token_ids=[],
            poc_params=PoCParams(block_hash="h", public_key="k", block_height=1,
                                 nonce=hash(req_id) % 1000, seq_len=8,
                                 poc_decode=True, max_tokens=4),
        )
    params, seed = spec
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[5, THINK_START],
        mm_features=[],
        sampling_params=params,
        block_ids=([],),
        generator=None if seed is None else torch.Generator().manual_seed(seed),
        num_computed_tokens=2,
        output_token_ids=[],
    )


def _bitmask(n):
    mask = np.zeros((n, (VOCAB + 31) // 32), dtype=np.int32)
    mask[:, ALLOWED // 32] = 1 << (ALLOWED % 32)
    return mask


def _apply_grammar(input_batch, logits, structured):
    if structured and logits is not None:
        apply_grammar_bitmask(
            SimpleNamespace(scheduled_spec_decode_tokens={}),
            GrammarOutput(structured_output_request_ids=list(structured),
                          grammar_bitmask=_bitmask(len(structured))),
            input_batch,
            logits,
        )


def _bridge(input_batch, poc_ids):
    runner = SimpleNamespace(input_batch=input_batch, model=_Model(), sampler=Sampler(),
                             device=CPU, speculative_config=None)
    bridge = PoCRunnerBridge(runner)
    bridge._step = {"poc_req_ids": set(poc_ids), "poc_requests": [],
                    "poc_metadata": None, "poc_position_mask": None}
    return bridge


def _fill(input_batch, rows):
    for req_id, spec in rows:
        input_batch.add_request(_request(req_id, spec))
    input_batch.refresh_metadata()


def _hidden(rows):
    gen = torch.Generator().manual_seed(1)
    table = {req_id: torch.randn(HIDDEN, generator=gen) for req_id, _ in rows}
    return table


def _run_mixed(rows, structured=(), input_batch=None):
    if input_batch is None:
        input_batch = _input_batch()
        _fill(input_batch, rows)
    table = _hidden(rows)
    order = input_batch.req_ids[: input_batch.num_reqs]
    hidden = torch.stack([table[r] for r in order])
    poc_ids = [r for r, s in rows if s is None]
    bridge = _bridge(input_batch, poc_ids)
    with torch.inference_mode():
        logits = bridge.compute_logits(hidden)
        _apply_grammar(input_batch, logits, structured)
        out = bridge.sample_chat_rows(logits, input_batch.sampling_metadata)
    return out, input_batch


def _run_reference(rows, structured=()):
    chat_rows = [(r, s) for r, s in rows if s is not None]
    input_batch = _input_batch()
    _fill(input_batch, chat_rows)
    table = _hidden(rows)
    hidden = torch.stack([table[r] for r, _ in chat_rows])
    with torch.inference_mode():
        logits = _Model().compute_logits(hidden)
        _apply_grammar(input_batch, logits, structured)
        out = Sampler()(logits=logits, sampling_metadata=input_batch.sampling_metadata)
    return out, input_batch


def _token(out, input_batch, req_id):
    return int(out.sampled_token_ids[input_batch.req_id_to_index[req_id], 0])


def _assert_matches_reference(rows, structured=(), logprobs=False):
    mixed, mixed_batch = _run_mixed(rows, structured)
    ref, ref_batch = _run_reference(rows, structured)
    for req_id, spec in rows:
        if spec is None:
            assert _token(mixed, mixed_batch, req_id) == 0
            continue
        assert _token(mixed, mixed_batch, req_id) == _token(ref, ref_batch, req_id), req_id
        if logprobs:
            mi = mixed_batch.req_id_to_index[req_id]
            ri = ref_batch.req_id_to_index[req_id]
            for field in ("logprob_token_ids", "logprobs", "selected_token_ranks"):
                torch.testing.assert_close(
                    getattr(mixed.logprobs_tensors, field)[mi],
                    getattr(ref.logprobs_tensors, field)[ri],
                )
    return mixed, mixed_batch


def _greedy(**kw):
    return (SamplingParams(temperature=0.0, **kw), None)


def _seeded(seed, **kw):
    return (SamplingParams(temperature=kw.pop("temperature", 0.8), seed=seed, **kw), seed)


def test_structured_row_after_poc_gets_its_mask():
    rows = [("poc0", None), ("so", _greedy())]
    mixed, batch = _assert_matches_reference(rows, structured=["so"])
    assert _token(mixed, batch, "so") == ALLOWED


def test_mask_does_not_move_to_following_plain_row():
    rows = [("poc0", None), ("so", _greedy()), ("plain", _greedy())]
    mixed, batch = _assert_matches_reference(rows, structured=["so"])
    assert _token(mixed, batch, "so") == ALLOWED
    assert _token(mixed, batch, "plain") != ALLOWED


def test_min_p_applies_to_chat_row():
    _assert_matches_reference([("poc0", None), ("chat", _seeded(3, min_p=0.1))])


def test_logit_bias_applies_to_chat_row():
    rows = [("poc0", None), ("plain", _greedy()), ("chat", _greedy(logit_bias={7: 100.0}))]
    mixed, batch = _assert_matches_reference(rows)
    assert _token(mixed, batch, "chat") == 7
    assert _token(mixed, batch, "plain") != 7


def test_thinking_budget_forces_only_its_row():
    rows = [("poc0", None), ("a", _greedy(thinking_token_budget=0)), ("b", _greedy())]
    mixed, batch = _assert_matches_reference(rows)
    assert _token(mixed, batch, "a") == THINK_END
    assert _token(mixed, batch, "b") != THINK_END


def test_mixed_logprobs_modes():
    rows = [
        ("poc0", None),
        ("a", _greedy(logprobs=3, logprobs_mode="processed_logprobs",
                      logit_bias={9: 5.0})),
        ("b", _greedy(logprobs=3, logprobs_mode="raw_logprobs")),
    ]
    _, batch = _run_mixed(rows)
    assert batch.batch_logprobs_mode == "mixed"
    _assert_matches_reference(rows, logprobs=True)


@pytest.mark.parametrize("rows", [
    [("chat", _greedy())],
    [("chat", _greedy()), ("poc0", None)],
])
def test_chat_first_batches_unchanged(rows):
    _assert_matches_reference(rows, structured=["chat"])


def test_poc_only_step_skips_sampler():
    input_batch = _input_batch()
    _fill(input_batch, [("poc0", None), ("poc1", None)])
    bridge = _bridge(input_batch, ["poc0", "poc1"])
    bridge.runner.sampler = None
    hidden = torch.zeros(2, HIDDEN)
    assert bridge.compute_logits(hidden) is None
    out = bridge.sample_chat_rows(None, input_batch.sampling_metadata)
    assert out.sampled_token_ids.tolist() == [[0], [0]]
    assert out.logprobs_tensors is None


def test_interleaved_rows_match_chat_only_batch():
    rows = [
        ("p0", None),
        ("so", _greedy(logprobs=2)),
        ("p1", None),
        ("p2", None),
        ("minp", _seeded(11, min_p=0.1, min_tokens=3, stop_token_ids=[ALLOWED],
                         logprobs=2)),
        ("bias", _seeded(12, logit_bias={7: 3.0}, logprobs=2,
                         logprobs_mode="processed_logprobs")),
        ("p3", None),
        ("think", _greedy(thinking_token_budget=0, logprobs=2)),
    ]
    mixed, batch = _assert_matches_reference(rows, structured=["so"], logprobs=True)
    assert _token(mixed, batch, "so") == ALLOWED
    assert _token(mixed, batch, "think") == THINK_END


def test_poc_in_reused_chat_slot_does_not_change_chat_output():
    stale = _greedy(top_k=5, min_p=0.2, logit_bias={7: 100.0},
                    thinking_token_budget=0, min_tokens=5, stop_token_ids=[3])
    input_batch = _input_batch()
    _fill(input_batch, [("old", stale), ("chat", _greedy(logprobs=2))])
    input_batch.remove_request("old")
    input_batch.add_request(_request("poc0", None))
    input_batch.condense()
    input_batch.refresh_metadata()
    assert input_batch.req_id_to_index["poc0"] == 0
    rows = [("poc0", None), ("chat", _greedy(logprobs=2))]
    mixed, _ = _run_mixed(rows, input_batch=input_batch)
    ref, ref_batch = _run_reference(rows)
    assert _token(mixed, input_batch, "chat") == _token(ref, ref_batch, "chat")
    assert _token(mixed, input_batch, "poc0") == 0
    assert not input_batch.thinking_budget_state_holder.has_tracked_requests()


def test_all_random_batch_with_poc_has_finite_outputs():
    rows = [("poc0", None), ("a", _seeded(21, logprobs=2)), ("b", _seeded(22, logprobs=2)),
            ("poc1", None)]
    mixed, batch = _assert_matches_reference(rows, logprobs=True)
    assert batch.all_random
    for req_id in ("a", "b"):
        assert torch.isfinite(mixed.logprobs_tensors.logprobs[batch.req_id_to_index[req_id]]).all()


def test_logits_rows_must_match_requests():
    input_batch = _input_batch()
    _fill(input_batch, [("poc0", None), ("chat", _greedy())])
    bridge = _bridge(input_batch, ["poc0"])
    with pytest.raises(RuntimeError, match="logits rows"):
        bridge.compute_logits(torch.zeros(3, HIDDEN))


def test_load_rejects_speculative_decoding():
    bridge = PoCRunnerBridge(SimpleNamespace(speculative_config=object()))
    with pytest.raises(RuntimeError, match="speculative"):
        bridge.load(None)


def test_pp_receiver_samples_without_local_compute_logits():
    rows = [("poc0", None), ("so", _greedy()), ("chat", _greedy(logprobs=2))]
    sender = _input_batch()
    _fill(sender, rows)
    table = _hidden(rows)
    hidden = torch.stack([table[r] for r in sender.req_ids[: sender.num_reqs]])
    with torch.inference_mode():
        broadcast = _bridge(sender, ["poc0"]).compute_logits(hidden)
    receiver = _input_batch()
    _fill(receiver, rows)
    bridge = _bridge(receiver, ["poc0"])
    with torch.inference_mode():
        logits = broadcast.clone()
        _apply_grammar(receiver, logits, ["so"])
        out = bridge.sample_chat_rows(logits, receiver.sampling_metadata)
    ref, ref_batch = _run_reference(rows, ["so"])
    assert _token(out, receiver, "poc0") == 0
    for req_id in ("so", "chat"):
        assert _token(out, receiver, req_id) == _token(ref, ref_batch, req_id)
    assert _token(out, receiver, "so") == ALLOWED


def test_lora_runs_lm_head_on_all_rows_and_zeroes_poc_rows():
    rows = [("poc0", None), ("chat", _greedy(logprobs=2)), ("poc1", None)]
    input_batch = _input_batch()
    _fill(input_batch, rows)
    table = _hidden(rows)
    table["poc0"] = torch.full((HIDDEN,), float("nan"))
    table["poc1"] = torch.full((HIDDEN,), float("inf"))
    hidden = torch.stack([table[r] for r in input_batch.req_ids[: input_batch.num_reqs]])
    bridge = _bridge(input_batch, ["poc0", "poc1"])
    bridge.runner.lora_config = object()
    seen = []
    model = bridge.runner.model
    original = model.compute_logits
    model.compute_logits = lambda h: seen.append(h.shape[0]) or original(h)
    with torch.inference_mode():
        logits = bridge.compute_logits(hidden)
        out = bridge.sample_chat_rows(logits, input_batch.sampling_metadata)
    assert seen == [3]
    assert torch.isfinite(logits).all()
    for req_id in ("poc0", "poc1"):
        assert torch.equal(logits[input_batch.req_id_to_index[req_id]],
                           torch.zeros(VOCAB))
        assert _token(out, input_batch, req_id) == 0
    ref, ref_batch = _run_reference(rows)
    assert _token(out, input_batch, "chat") == _token(ref, ref_batch, "chat")
