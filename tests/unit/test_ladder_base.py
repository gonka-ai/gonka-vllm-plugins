"""Seeded-routing ladder base is one consensus constant for every model (100)."""
import torch

from gonka_poc.poc import decode_random as dr


def test_ladder_base_is_uniform():
    assert dr.LADDER_BASE == 100
    assert dr.set_ladder_base_for_model("deepseek_v4") == 100
    assert dr.set_ladder_base_for_model("minimax_m2") == 100
    assert dr.set_ladder_base_for_model(None) == 100
    assert dr.ladder_base() == 100


def test_forced_logits_use_the_base():
    seed = torch.tensor([7], dtype=torch.int64)
    dr.set_ladder_base_for_model("minimax_m2")
    lo = dr._forced_logits(seed, 16, 4, torch.device("cpu"))
    chosen = lo[0] > -1.0e3
    assert chosen.sum().item() == 4
    assert lo[0][chosen].max().item() == 104.0          # ladder 104..101
    assert lo[0][chosen].min().item() == 101.0
