"""Version-dispatched compat shim isolating private vLLM API touchpoints.

The plugin is pinned in ``pyproject.toml`` to the vllm minors registered in
:data:`_DISPATCH`. Across those minors some private structures
(CommonAttentionMetadata, kv_caches layout, internal model_runner attributes)
may shift; we localise every such touchpoint inside one of the ``vN_M.py``
modules in this package.

Selection happens LAZILY -- the first call to :func:`current` imports the
matching submodule. This keeps test / lint discovery from forcing a real
``import vllm`` at module load time.

Usage (the ONLY supported form -- see history below for why)::

    from gonka_poc._compat import current
    compat = current()
    md = compat.build_common_attention_metadata(...)

History: an earlier revision exposed ``current`` as an ``lru_cache``-wrapped
function and tried to dual-publish it via PEP 562 ``__getattr__`` so
``_compat.current.<symbol>`` would also work. The PEP 562 path silently lost
to real-attribute lookup for ``from gonka_poc._compat import current``
(Python binds the function object directly, bypassing ``__getattr__``), so
the first ``compat.build_common_attention_metadata(...)`` call raised
``AttributeError`` and crashed the PoC forward. The contract is now:
``current`` is a plain function, call it to get the module.

Add a new ``vN_M.py`` AND register it in :data:`_DISPATCH` when porting to a
new vLLM minor; keep contract tests (``tests/contract/test_v*_api_surface.py``)
in lock-step.
"""
from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType
from typing import Mapping, Tuple


def _parse_version(v: str) -> Tuple[int, ...]:
    parts: list[int] = []
    for chunk in v.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            parts.append(int(digits))
        if len(parts) >= 3:
            break
    return tuple(parts)


def _detect_vllm_version() -> Tuple[int, ...]:
    try:
        import vllm  # type: ignore
        return _parse_version(getattr(vllm, "__version__", "0.0.0"))
    except Exception:
        return (0, 0, 0)


# Map of (major, minor) -> module name within this package.
_DISPATCH: Mapping[Tuple[int, int], str] = {
    (0, 25): "gonka_poc._compat.v0_25",
    (0, 28): "gonka_poc._compat.v0_28",
}


@lru_cache(maxsize=1)
def _current_impl() -> ModuleType:
    """Resolve and import the compat submodule for the installed vllm minor.

    Cached so repeat callers don't re-import. Raises ``RuntimeError`` with the
    list of supported versions when the installed vllm minor is not in
    :data:`_DISPATCH` -- we explicitly do NOT silently fall back to the
    latest-known shim, because contract drift across minors can silently
    corrupt PoC outputs.
    """
    major, minor, *_ = (_detect_vllm_version() + (0, 0, 0))[:3]
    mod_name = _DISPATCH.get((major, minor))
    if mod_name is None:
        supported = ", ".join(
            f"{m}.{n}" for (m, n) in sorted(_DISPATCH.keys())
        )
        raise RuntimeError(
            f"gonka_poc._compat: vllm {major}.{minor} is not supported. "
            f"Supported vllm minors: {supported}. "
            "Add a new gonka_poc/_compat/vN_M.py and register it in _DISPATCH."
        )
    return importlib.import_module(mod_name)


def current() -> ModuleType:
    """Return the compat submodule matching the installed vllm minor.

    Consumers MUST call this: ``compat = current(); compat.<symbol>(...)``
    -- see the module docstring's History note for why no other form is
    supported.
    """
    return _current_impl()


__all__ = ["current"]
