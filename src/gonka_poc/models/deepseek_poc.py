# SPDX-License-Identifier: Apache-2.0
"""Compat shim: DeepSeek-family entries now come from the model-agnostic
factory (models/factory.py). Kimi K-series declares the DeepSeek
architecture in its HF config and resolves through the same list."""
from .factory import build_poc_subclasses  # noqa: F401
