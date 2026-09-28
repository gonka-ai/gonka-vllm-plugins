"""The scheduler seam: a PoC prefill is atomic, a decode row takes one token."""
import types

from gonka_poc.mixed.admission import poc_step_tokens


def req(poc, computed=0, seq_len=256, max_tokens=256):
    r = types.SimpleNamespace(num_computed_tokens=computed)
    r.poc_params = types.SimpleNamespace(seq_len=seq_len, max_tokens=max_tokens) if poc else None
    return r


def test_chat_rows_pass_through():
    assert poc_step_tokens(req(False, computed=300), 17, 8192) == 17


def test_prefill_is_all_or_nothing():
    assert poc_step_tokens(req(True, computed=0), 100, 8192) == 256    # vLLM clamped to 100: take all 256
    assert poc_step_tokens(req(True, computed=0), 256, 256) == 256     # exactly fits
    assert poc_step_tokens(req(True, computed=0), 200, 200) == 0       # does not fit: wait for a later step


def test_decode_rows_take_one_token():
    assert poc_step_tokens(req(True, computed=256), 0, 8192) == 1     # vLLM's arithmetic says 0 (no sampled tokens)
    assert poc_step_tokens(req(True, computed=300), 1, 1) == 1


def test_prefill_only_scheme_is_one_atomic_step():
    assert poc_step_tokens(req(True, computed=0, max_tokens=0), 64, 8192) == 256
    assert poc_step_tokens(req(True, computed=0, max_tokens=0), 64, 64) == 0
