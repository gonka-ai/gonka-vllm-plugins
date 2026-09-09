# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic PoC entry: ONE factory, an architecture list, zero
per-model code (call agreement 2026-08-19).

For every architecture in the list we register, over the stock name, a
subclass whose ``load_weights`` attaches the PoC wrappers at the END of
weight loading — after the checkpoint is mapped (earlier wrapping renames
parameters and breaks loading) and before the first forward (vLLM compiles
lazily, so the compiled graph contains the wrappers — the 0.20 bit path).
Everything model-shaped is DISCOVERED at attach: decoder layers by longest
ModuleList, embed_tokens by attribute, MoE by gate+experts, grouped-topk by
n_group/topk_group from the config, routing window default = n_experts of
this model (full scatter; on MiniMax == the shipped-golden 256).

Extra architectures without code changes:
    POC_ARCHITECTURES="MyArch=vllm.model_executor.models.my_mod:MyArchClass,..."
"""
import importlib
import logging
import os
from typing import Optional

import torch

from gonka_poc.poc.native import attach_native_poc

logger = logging.getLogger(__name__)

# Buffer cap: prefill rows of the largest decode chunk (nonces * seq_len).
POC_NATIVE_MAX_ROWS = int(os.environ.get("POC_NATIVE_MAX_ROWS", str(128 * 256)))
# None => derive per model at attach (full scatter = n_experts). An explicit
# env value is a PROCESS constant frozen into the compiled graph — change via
# env before start, never per request (0.20 semantics).
_env_window = os.environ.get("POC_ROUTE_WINDOW", "").strip()
POC_ROUTE_WINDOW: Optional[int] = int(_env_window) if _env_window else None

DEFAULT_ARCHITECTURES = [
    ("MiniMaxM2ForCausalLM",
     "vllm.model_executor.models.minimax_m2", "MiniMaxM2ForCausalLM"),
    ("DeepseekV3ForCausalLM",
     "vllm.model_executor.models.deepseek_v2", "DeepseekV3ForCausalLM"),
    ("DeepseekV2ForCausalLM",
     "vllm.model_executor.models.deepseek_v2", "DeepseekV2ForCausalLM"),
    ("DeepseekV4ForCausalLM",
     "vllm.model_executor.models.deepseek_v4", "DeepseekV4ForCausalLM"),
]


def _extra_architectures():
    raw = os.environ.get("POC_ARCHITECTURES", "").strip()
    if not raw:
        return
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            arch, ref = item.split("=", 1)
            mod, cls = ref.split(":", 1)
            yield arch.strip(), mod.strip(), cls.strip()
        except ValueError:
            logger.error("POC_ARCHITECTURES: bad entry %r "
                         "(want Arch=module:Class)", item)


def _attach_after_load(model) -> None:
    vllm_config = getattr(model, "vllm_config", None)
    hidden = (int(vllm_config.model_config.get_hidden_size())
              if vllm_config is not None else
              int(model.config.hidden_size))
    try:
        p = next(model.parameters())
        device, dtype = p.device, p.dtype
    except StopIteration:  # pragma: no cover
        device, dtype = torch.device("cuda"), torch.bfloat16
    max_rows = POC_NATIVE_MAX_ROWS
    if vllm_config is not None:
        try:
            max_rows = max(max_rows,
                           int(vllm_config.scheduler_config
                               .max_num_batched_tokens))
        except Exception:  # pragma: no cover — config shape drift
            pass
    attach_native_poc(model, hidden, max_rows, device, dtype,
                      POC_ROUTE_WINDOW)


def make_poc_subclass(base, cls_name: str):
    def load_weights(self, weights, _base=base):
        out = _base.load_weights(self, weights)
        _attach_after_load(self)
        return out

    return type(f"{cls_name}PoC", (base,), {
        "__doc__": f"{cls_name} + PoC wrappers attached after weight load "
                   f"(model-agnostic factory; see module docstring).",
        "load_weights": load_weights,
    })


def build_poc_subclasses():
    """Yield (architecture_name, subclass) for every architecture whose base
    class exists in this vLLM build. Import errors are per-architecture."""
    for arch, mod_name, cls_name in (*DEFAULT_ARCHITECTURES,
                                     *_extra_architectures()):
        try:
            base = getattr(importlib.import_module(mod_name), cls_name)
        except (ImportError, AttributeError):
            continue
        yield arch, make_poc_subclass(base, cls_name)
