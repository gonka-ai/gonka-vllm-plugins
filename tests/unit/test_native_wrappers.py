# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the ported native wrappers (layer B plumbing).

A tiny fake model (decoder .layers with a .gate/.experts MoE inside) exercises
attach discovery, mask identity, the Householder reflection and the seeded
router override — no vLLM, no GPU.
"""
import torch
from torch import nn

from gonka_poc.poc import gpu_random
from gonka_poc.poc.native import attach_native_poc

H = 32


class FakeExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_num_experts = 16
        self.top_k = 2


class FakeGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(H, 16, bias=False)

    def forward(self, x):
        return self.lin(x), None      # GateLinear returns (logits, bias)


class FakeMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = FakeGate()
        self.experts = FakeExperts()

    def forward(self, x):
        logits, _ = self.gate(x)
        return logits


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = FakeMoE()

    def forward(self, hidden, residual=None):
        return hidden + 1.0, residual


class FakeCore(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(64, H)
        self.layers = nn.ModuleList(FakeLayer() for _ in range(n_layers))


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeCore()


def test_attach_discovery_and_idempotence():
    m = FakeModel()
    st = attach_native_poc(m, H, max_rows=8, device=torch.device("cpu"),
                           dtype=torch.float32, route_window=256)
    assert len(st.router_meta) == 2 and st.router_meta[0] == (16, 2, 1, 1)
    assert all(hasattr(l.moe.gate, "_poc_state")
               for l in m.model.layers)
    again = attach_native_poc(m, H, 8, torch.device("cpu"),
                              torch.float32, 256)
    assert again is st  # idempotent


def test_layer_wrapper_identity_when_masked_off():
    m = FakeModel()
    st = attach_native_poc(m, H, 4, torch.device("cpu"), torch.float32, 256)
    st.clear()
    x = torch.randn(4, H)
    out, _ = m.model.layers[0](x, None)
    assert torch.equal(out, x + 1.0)  # exact identity path


def test_layer_wrapper_reflects_poc_rows():
    m = FakeModel()
    st = attach_native_poc(m, H, 4, torch.device("cpu"), torch.float32, 256)
    bh = "deadbeef" * 8
    st.set_rows(bh, 2)  # first 2 rows PoC
    x = torch.randn(4, H)
    out, _ = m.model.layers[0](x, None)
    base = x + 1.0
    v = gpu_random.generate_householder_vector(
        f"{bh}_layer_0_householder", H, torch.device("cpu"))
    expect0 = base[0] - 2.0 * (base[0] @ v) * v
    assert torch.allclose(out[0], expect0, atol=1e-6)
    assert torch.equal(out[2], base[2])          # row beyond n_rows untouched


def test_router_wrapper_forces_seeded_logits():
    m = FakeModel()
    st = attach_native_poc(m, H, 4, torch.device("cpu"), torch.float32, 256)
    bh = "deadbeef" * 8
    st.set_rows(bh, 4)
    st.set_routing(bh, [0, 1, 2, 3], 1, 5)
    x = torch.randn(4, H)
    logits = m.model.layers[0].moe(x)
    # expected: seeded logits for (bh, nonce, step=5, layer=0)
    for row, nonce in enumerate([0, 1, 2, 3]):
        exp = gpu_random.seeded_experts(bh, nonce, 5, 0, 16, 2,
                                        torch.device("cpu"))
        got = torch.topk(logits[row], 2).indices
        assert torch.equal(torch.sort(got).values,
                           torch.sort(exp).values), f"row {row}"


def test_per_nonce_reflection_guarded():
    m = FakeModel()
    st = attach_native_poc(m, H, 2, torch.device("cpu"), torch.float32, 256)
    import pytest as _pt
    with _pt.raises(NotImplementedError):
        st.set_rows("deadbeef" * 8, 2, per_nonce=True)


def test_embed_patch_swaps_poc_rows_and_stays_identity_for_chat():
    m = FakeModel()
    st = attach_native_poc(m, H, 4, torch.device("cpu"), torch.float32, 256)
    assert st.has_embed_patch
    ids = torch.arange(4)
    plain = nn.Embedding(64, H)
    plain.load_state_dict(m.model.embed_tokens.state_dict())
    st.clear()
    out = m.model.embed_tokens(ids)
    assert torch.equal(out, plain(ids))          # mask off -> exact identity
    synth = torch.randn(2, H)
    st.set_rows("deadbeef" * 8, 2)
    st.set_embeds(synth)
    out = m.model.embed_tokens(ids)
    assert torch.equal(out[:2], synth)           # PoC rows swapped
    assert torch.equal(out[2:], plain(ids)[2:])  # rest untouched
    big = torch.zeros(9, dtype=torch.int64)      # beyond max_rows -> guard
    assert torch.equal(m.model.embed_tokens(big), plain(big))


class FakeGroupedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_num_experts = 64
        self.top_k = 8
        self.num_expert_group = 8
        self.topk_group = 4


class FakeSharedExpert(nn.Module):
    def forward(self, x):
        return x * 0.5


class FakeGroupedGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(H, 64, bias=False)

    def forward(self, x):
        return self.lin(x), None


class FakeGroupedMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = FakeGroupedGate()
        self.experts = FakeGroupedExperts()
        self.shared_experts = FakeSharedExpert()

    def forward(self, x):
        logits, _ = self.gate(x)
        return logits


class FakeDenseLayer(nn.Module):
    def forward(self, hidden, residual=None):
        return hidden + 2.0, residual


class FakeDSLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = FakeGroupedMoE()

    def forward(self, hidden, residual=None):
        return hidden + 1.0, residual


class FakeDSCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(64, H)
        # dense prefix (layers 0-1), then MoE layers (2-4) — DeepSeek shape
        self.layers = nn.ModuleList(
            [FakeDenseLayer(), FakeDenseLayer(),
             FakeDSLayer(), FakeDSLayer(), FakeDSLayer()])


class FakeDSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeDSCore()


def test_deepseek_shape_global_layer_index_and_groups():
    """Phase-0 #4: route seeds use the GLOBAL decoder-layer index (2,3,4 for
    the MoE layers behind a 2-layer dense prefix), and #2: group dims flow
    into router_meta / the gate patch."""
    m = FakeDSModel()
    st = attach_native_poc(m, H, 8, torch.device("cpu"), torch.float32, 256)
    assert [li for li, _ in st._route_base] == [2, 3, 4]
    assert st.router_meta == [(64, 8, 8, 4)] * 3
    bh = "deadbeef" * 8
    st.set_rows(bh, 4)
    st.set_routing(bh, [0, 1, 2, 3], 1, 5)
    # seed of the FIRST MoE layer must be derived with layer index 2
    expect = gpu_random._seed_from_string(
        gpu_random.route_base_seed(bh, 0, 2))
    assert int(st._route_base[0][1][0]) == expect
    # forced logits obey the group structure (coverage over 4 of 8 groups)
    logits = m.model.layers[2].moe(torch.randn(4, H))
    gsz = 64 // 8
    for row in range(4):
        experts = (logits[row] > 0).nonzero().flatten()
        assert len(experts) == 8
        assert len(torch.unique(experts // gsz)) == 4
    # dense prefix layers still reflect (Householder applies to ALL layers)
    st.set_rows(bh, 2)
    x = torch.randn(2, H)
    out, _ = m.model.layers[0](x, None)
    assert not torch.equal(out, x + 2.0)
