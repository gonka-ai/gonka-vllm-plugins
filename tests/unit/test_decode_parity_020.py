# SPDX-License-Identifier: Apache-2.0
"""Bit-for-bit parity: ported decode math vs the 0.20 in-tree branch.

Ladder step 1 of the migration: every consensus-critical function
ported from ``poc-v0.20-decode-poc-cg @ 5c1d09f55e92`` must produce IDENTICAL
bits to the original source, executed side by side on CPU.

The 0.20 sources are not a package — they are loaded from the directory
named by ``DECODE020_PATH`` (no default). When the variable is unset or the
path is absent the module SKIPS: CI without the reference sources stays
green, a box that has them runs the real comparison.

Also proves the two DOCUMENTED equivalences the port relies on:
  * ``generate_inputs``: 0.20 batched variant == plugin v0.1.x per-nonce loop;
  * decode seed scheme: ``random_pick_indices`` now always
    salts ``_decode{step}`` — asserted against the 0.20 body, and asserted
    DIFFERENT from the legacy v0.1.x seeding (guard against silent rollback).
"""
import hashlib
import importlib.util
import os
import sys
import types

import pytest
import torch

DECODE020_PATH = os.environ.get(
    "DECODE020_PATH",
    "",
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(DECODE020_PATH),
    reason="set DECODE020_PATH to the 0.20 reference sources to run parity",
)


def _load_020(name: str):
    """Import a 0.20 module from the bundle with a stub vllm.logger."""
    if "vllm" not in sys.modules:
        vllm_pkg = types.ModuleType("vllm")
        logger_mod = types.ModuleType("vllm.logger")
        logger_mod.init_logger = lambda _: __import__("logging").getLogger(_)
        vllm_pkg.logger = logger_mod
        sys.modules["vllm"] = vllm_pkg
        sys.modules["vllm.logger"] = logger_mod
    spec = importlib.util.spec_from_file_location(
        f"branch020_{name}", os.path.join(DECODE020_PATH, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def g020():
    return _load_020("gpu_random")


@pytest.fixture(scope="module")
def s020():
    return _load_020("sphere")


@pytest.fixture(scope="module")
def gnew():
    from gonka_poc.poc import gpu_random
    return gpu_random


@pytest.fixture(scope="module")
def snew():
    from gonka_poc.poc import sphere
    return sphere


DEV = torch.device("cpu")
BH = "deadbeef" * 8
PK = "cafebabe" * 8
NONCES = [0, 1, 7, 41, 399]


# ---------------------------------------------------------------- seeds
def test_seed_from_string_parity(g020, gnew):
    strings = [
        f"{BH}_{PK}_nonce{n}" for n in NONCES
    ] + [
        f"{BH}_{PK}_nonce{n}_decode{s}_k{k}"
        for n in NONCES for s in (0, 1, 255) for k in (0, 15)
    ] + [g020.route_base_seed(BH, n, layer) for n in NONCES for layer in (0, 61)]
    for st in strings:
        assert g020._seed_from_string(st) == gnew._seed_from_string(st), st


def test_decode_base_seeds_parity(g020, gnew):
    a = g020.decode_base_seeds(BH, PK, NONCES, DEV)
    b = gnew.decode_base_seeds(BH, PK, NONCES, DEV)
    assert torch.equal(a, b)


def test_step_seeds_parity(g020, gnew):
    base = gnew.decode_base_seeds(BH, PK, NONCES, DEV)
    prev = torch.tensor([0, 3, 15, 7, 11], dtype=torch.int64)
    for step in (0, 1, 128, 256):
        a = g020._step_seeds(base, step, prev, g020._SALT_DECODE_EMBED)
        b = gnew._step_seeds(base, step, prev, gnew._SALT_DECODE_EMBED)
        assert torch.equal(a, b), f"step={step}"
    steps_t = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    a = g020._step_seeds(base, steps_t, prev, g020._SALT_DECODE_PICK)
    b = gnew._step_seeds(base, steps_t, prev, gnew._SALT_DECODE_PICK)
    assert torch.equal(a, b)


def test_salts_unchanged(g020, gnew):
    assert (g020._SALT_DECODE_EMBED, g020._SALT_DECODE_PICK) == \
           (gnew._SALT_DECODE_EMBED, gnew._SALT_DECODE_PICK) == (0x0D, 0x91)
    assert (g020._MIX_A, g020._MIX_B) == (gnew._MIX_A, gnew._MIX_B)


# ------------------------------------------------------------- inputs
def test_generate_inputs_parity_and_loop_equivalence(g020, gnew):
    dim, seq = 64, 16
    a = g020.generate_inputs(BH, PK, NONCES, dim, seq, DEV)
    b = gnew.generate_inputs(BH, PK, NONCES, dim, seq, DEV)
    assert torch.equal(a, b)
    # documented equivalence: batched == the v0.1.x per-nonce loop
    loop = torch.empty(len(NONCES), seq, dim, dtype=torch.float16)
    for i, n in enumerate(NONCES):
        seed = gnew._seed_from_string(f"{BH}_{PK}_nonce{n}")
        loop[i] = gnew._normal(seed, seq * dim, DEV).view(seq, dim).to(torch.float16)
    assert torch.equal(b, loop)


def test_concat_murmur_parity(g020, gnew):
    a = g020.generate_inputs_concat_murmur(BH, PK, NONCES, 32, 8, DEV)
    b = gnew.generate_inputs_concat_murmur(BH, PK, NONCES, 32, 8, DEV)
    assert torch.equal(a, b)


def test_decode_inputs_parity_host_and_gpu(g020, gnew):
    dim = 48
    prev = [0, 3, 15, 7, 11]
    for step in (1, 2, 200):
        a = g020.generate_decode_inputs(BH, PK, NONCES, prev, step, dim, DEV)
        b = gnew.generate_decode_inputs(BH, PK, NONCES, prev, step, dim, DEV)
        assert torch.equal(a, b), f"host step={step}"
    base = gnew.decode_base_seeds(BH, PK, NONCES, DEV)
    prev_t = torch.tensor(prev, dtype=torch.int64)
    for step in (1, 2, 200):
        st = torch.full((len(NONCES),), step, dtype=torch.int64)
        a = g020.generate_decode_inputs_gpu(base, prev_t, st, dim, DEV)
        b = gnew.generate_decode_inputs_gpu(base, prev_t, st, dim, DEV)
        assert torch.equal(a, b), f"gpu step={step}"


# ---------------------------------------------------------------- pick
def test_pick_parity_including_chain(g020, gnew):
    H, K = 3072, 256
    a = g020.random_pick_indices(BH, PK, NONCES, H, K, DEV)
    b = gnew.random_pick_indices(BH, PK, NONCES, H, K, DEV)
    assert torch.equal(a, b)
    prev = [1, 2, 3, 4, 5]
    for step in (0, 5):
        a = g020.random_pick_indices(BH, PK, NONCES, H, K, DEV,
                                     prev_point_ids=prev, step=step)
        b = gnew.random_pick_indices(BH, PK, NONCES, H, K, DEV,
                                     prev_point_ids=prev, step=step)
        assert torch.equal(a, b), f"step={step}"


def test_pick_differs_from_legacy_v01x_seeding(gnew):
    """Decision #1/#4 guard: the new scheme must NOT silently reproduce the
    legacy `_pick_{k}` seeding of the live prefill fleet."""
    H, K = 512, 12
    new = gnew.random_pick_indices(BH, PK, [0], H, K, DEV)
    legacy_seed = gnew._seed_from_string(f"{BH}_{PK}_nonce_0_pick_{K}")
    idx = torch.arange(H, dtype=torch.int32).unsqueeze(0)
    seed_t = torch.tensor([legacy_seed], dtype=torch.int64).unsqueeze(1)
    scores = gnew._batched_murmur3_32(idx, seed_t)
    legacy = torch.topk(-scores, k=K, largest=True, sorted=False, dim=1).indices.to(torch.int64)
    assert not torch.equal(torch.sort(new[0]).values, torch.sort(legacy[0]).values)


def test_pick_gpu_parity(g020, gnew):
    base = gnew.decode_base_seeds(BH, PK, NONCES, DEV)
    prev = torch.tensor([0, 3, 15, 7, 11], dtype=torch.int64)
    steps = torch.tensor([1, 1, 2, 2, 3], dtype=torch.int64)
    a = g020.random_pick_indices_gpu(base, prev, steps, 3072, 256, DEV)
    b = gnew.random_pick_indices_gpu(base, prev, steps, 3072, 256, DEV)
    assert torch.equal(a, b)


# ------------------------------------------------------------- routing
def test_routing_parity_full_and_windowed(g020, gnew):
    n_experts, top_k = 256, 8
    for window in (16, 64, 256):
        g020.set_route_window(window)
        gnew.set_route_window(window)
        for step in (0, 1, 200):
            for layer in (0, 30):
                a = g020.seeded_experts(BH, 7, step, layer, n_experts, top_k, DEV)
                b = gnew.seeded_experts(BH, 7, step, layer, n_experts, top_k, DEV)
                assert torch.equal(a, b), f"w={window} step={step} layer={layer}"
    # window >= n_experts must be the legacy full scatter (docs/04 §3)
    g020.set_route_window(256); gnew.set_route_window(256)
    base = torch.tensor([gnew._seed_from_string(gnew.route_base_seed(BH, 7, 0))],
                        dtype=torch.int64)
    steps = torch.tensor([3], dtype=torch.int64)
    a = g020.expert_logits_from_base(base, steps, n_experts, top_k, DEV)
    b = gnew.expert_logits_from_base(base, steps, n_experts, top_k, DEV)
    assert torch.equal(a, b)


def test_householder_and_haar_parity(g020, gnew):
    v020 = g020.generate_householder_vector(f"{BH}_layer_5_householder", 128, DEV)
    vnew = gnew.generate_householder_vector(f"{BH}_layer_5_householder", 128, DEV)
    assert torch.equal(v020, vnew)
    x = gnew._normal(12345, 5 * 12, DEV).view(5, 12)
    a = g020.apply_haar_rotation(BH, PK, NONCES, x.clone(), DEV)
    b = gnew.apply_haar_rotation(BH, PK, NONCES, x.clone(), DEV)
    assert torch.equal(a, b)


# -------------------------------------------------------------- sphere
def test_codebook_bytes_and_sha(s020, snew):
    frozen = open(os.path.join(DECODE020_PATH, "sphere_codebook.pt"), "rb").read()
    import gonka_poc.poc as pkg
    ours = open(os.path.join(os.path.dirname(pkg.__file__),
                             "sphere_codebook.pt"), "rb").read()
    assert hashlib.sha256(frozen).hexdigest() == hashlib.sha256(ours).hexdigest()
    cb020 = s020.get_sphere_codebook()
    cbnew = snew.get_sphere_codebook()
    assert torch.equal(cb020, cbnew)
    assert (snew.SPHERE_DIM, snew.SPHERE_POINTS) == (256, 16)


def test_snap_parity(s020, snew):
    cb = snew.get_sphere_codebook()
    q = snew.project_to_sphere(torch.randn(64, snew.SPHERE_DIM,
                                           generator=torch.Generator().manual_seed(7)))
    q[5] = float("nan")
    k0, bad0, m0 = s020.snap_with_margin(q, cb)
    k1, bad1, m1 = snew.snap_with_margin(q, cb)
    assert torch.equal(k0, k1) and torch.equal(bad0, bad1) and torch.equal(m0, m1)
    assert k1[5] == -1 and bad1[5]


# -------------------------------------------------------- chain rules
def test_chain_rules_against_020_formula():
    from gonka_poc.poc.decode_chain import count_mismatch, next_prev_k, keep_q_step
    k = torch.tensor([3, 3, -1, 7, 7], dtype=torch.int64)
    ref = torch.tensor([3, 4, 4, 7, 2], dtype=torch.int64)
    margin = torch.tensor([0.5, 0.5, 0.5, 0.01, 0.001])
    # 0.20 formula: (k != ref) & (k >= 0) & (margin >= tau)
    for tau in (0.0, 0.05):
        expect = ((k != ref) & (k >= 0) & (margin >= tau)).to(torch.int64)
        assert torch.equal(count_mismatch(k, ref, margin, tau), expect)
    assert torch.equal(next_prev_k(k, ref), ref)          # teacher forcing
    assert torch.equal(next_prev_k(k, None), k)           # generation
    assert keep_q_step(3, True, False, 0)
    assert keep_q_step(64, False, True, 64)
    assert not keep_q_step(65, False, True, 64)
