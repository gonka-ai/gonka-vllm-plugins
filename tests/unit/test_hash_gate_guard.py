import types

import pytest
import torch
from torch import nn

from gonka_poc.mixed.native import _TOKEN_ID_ROUTED_MODELS, _token_id_table


def gate_with(**tensors):
    g = nn.Linear(4, 8, bias=False)
    for name, t in tensors.items():
        if isinstance(t, nn.Parameter):
            g.register_parameter(name, t)
        else:
            g.register_buffer(name, t)
    return g


def test_int_table_on_gate_is_detected():
    g = gate_with(tid2eid=nn.Parameter(torch.zeros(100, 6, dtype=torch.int32), requires_grad=False))
    assert _token_id_table(g) is not None
    g64 = gate_with(table=torch.zeros(100, 6, dtype=torch.int64))
    assert _token_id_table(g64) is not None


def test_float_bias_and_1d_ints_are_not_tables():
    g = gate_with(e_score_correction_bias=nn.Parameter(torch.zeros(256), requires_grad=False),
                  offsets=torch.zeros(7, dtype=torch.int32))
    assert _token_id_table(g) is None


def test_allowlist_holds_deepseek_v4():
    assert "deepseek_v4" in _TOKEN_ID_ROUTED_MODELS
