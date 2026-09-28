"""PoC knobs come from vLLM's public --additional-config (gonka_poc section)."""
from types import SimpleNamespace

import pytest

from gonka_poc.mixed.policy import POC_CONFIG_DEFAULTS, poc_cfg


def _vc(section=None, extra=None):
    add = {} if extra is None else dict(extra)
    if section is not None:
        add["gonka_poc"] = section
    return SimpleNamespace(additional_config=add)


def test_defaults_without_config():
    for name, default in POC_CONFIG_DEFAULTS.items():
        assert poc_cfg(None, name) == default
        assert poc_cfg(SimpleNamespace(), name) == default
        assert poc_cfg(_vc(), name) == default
        assert poc_cfg(_vc(extra={"other": {"poc_seq_len": 1}}), name) == default


def test_section_values_win_and_are_coerced():
    vc = _vc({"poc_max_batch_size": "512", "poc_seq_len": 128,
              "poc_vector_artifacts": "true"})
    assert poc_cfg(vc, "poc_max_batch_size") == 512
    assert poc_cfg(vc, "poc_seq_len") == 128
    assert poc_cfg(vc, "poc_vector_artifacts") is True
    assert poc_cfg(vc, "poc_max_tokens") == POC_CONFIG_DEFAULTS["poc_max_tokens"]


def test_bool_coercion_is_strict():
    assert poc_cfg(_vc({"poc_vector_artifacts": "0"}), "poc_vector_artifacts") is False
    assert poc_cfg(_vc({"poc_vector_artifacts": False}), "poc_vector_artifacts") is False
    assert poc_cfg(_vc({"poc_vector_artifacts": 1}), "poc_vector_artifacts") is True


def test_unknown_knob_is_an_error():
    with pytest.raises(KeyError):
        poc_cfg(_vc(), "poc_share")
