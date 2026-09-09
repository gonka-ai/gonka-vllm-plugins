# SPDX-License-Identifier: Apache-2.0
"""Compat shim: the MiniMax entry now comes from the model-agnostic factory
(models/factory.py, call agreement 2026-08-19). Kept importable because
image build checks and older registry strings reference this module."""
from vllm.model_executor.models.minimax_m2 import MiniMaxM2ForCausalLM

from .factory import (  # noqa: F401  (re-exported process constants)
    POC_NATIVE_MAX_ROWS,
    POC_ROUTE_WINDOW,
    make_poc_subclass,
)

MiniMaxM2ForCausalLMPoC = make_poc_subclass(
    MiniMaxM2ForCausalLM, "MiniMaxM2ForCausalLM")
