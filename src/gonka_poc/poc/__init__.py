"""PoC sub-package.

Intentionally inert: importing ``gonka_poc.poc`` MUST NOT trigger any
side effects (no engine patching, no router registration, no model
imports). Each consumer pulls what it needs from the explicit module
path, e.g.::

    from gonka_poc.poc.routes import router as poc_router
    from gonka_poc.poc.data import encode_vector, decode_vector
    from gonka_poc.poc.decode_runner import execute_poc_decode

PoC dispatch is via ``collective_rpc("execute_poc_decode", kwargs=...)``
on :class:`gonka_poc.worker.PoCWorkerExtension`.
"""
