"""The compile cache is scoped by knobs that change the traced forward."""
from gonka_poc.compile_cache import knob_signature, scope_compile_cache


def test_defaults_leave_the_root_alone():
    env = {"VLLM_CACHE_ROOT": "/tmp/vllm_r"}
    assert scope_compile_cache(env) is None
    assert env["VLLM_CACHE_ROOT"] == "/tmp/vllm_r"
    env = {"VLLM_CACHE_ROOT": "/tmp/vllm_r", "POC_ABLATE": "", "VLLM_POC_DEBUG_TP": "0"}
    assert scope_compile_cache(env) is None          # explicit defaults == unset


def test_non_default_knob_gets_its_own_root_and_is_idempotent():
    env = {"VLLM_CACHE_ROOT": "/tmp/vllm_r", "VLLM_POC_DEBUG_TP": "1"}
    scoped = scope_compile_cache(env)
    assert scoped and scoped.startswith("/tmp/vllm_r/gonka-poc-knobs-")
    assert env["VLLM_CACHE_ROOT"] == scoped
    assert scope_compile_cache(env) == scoped        # child process inherits: unchanged
    env2 = {"VLLM_CACHE_ROOT": "/tmp/vllm_r", "VLLM_POC_DEBUG_TP": "1", "POC_ABLATE": "router,reflect"}
    other = scope_compile_cache(env2)
    assert other != scoped
    env3 = {"VLLM_CACHE_ROOT": "/tmp/vllm_r", "POC_ABLATE": "reflect, router"}
    assert knob_signature(env3)[0] == {"POC_ABLATE": "reflect,router"}   # order-insensitive


def test_parent_scoped_for_other_knobs_is_replaced():
    env = {"VLLM_CACHE_ROOT": "/tmp/vllm_r/gonka-poc-knobs-deadbeef00", "VLLM_POC_DEBUG_TP": "1"}
    scoped = scope_compile_cache(env)
    assert scoped.startswith("/tmp/vllm_r/gonka-poc-knobs-") and "deadbeef00" not in scoped
