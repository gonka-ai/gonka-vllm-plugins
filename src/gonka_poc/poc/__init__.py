"""PoC sub-package.

Intentionally inert: importing ``gonka_poc.poc`` MUST NOT trigger any side
effects. Consumers pull what they need from the explicit module path::

    from gonka_poc.poc.routes import router as poc_router
    from gonka_poc.poc.data import encode_vector, decode_vector
    from gonka_poc.poc.poc_model_runner import execute_poc_forward

Re-exporting here drags the API layer -- and through it vLLM -- into every
process that only wanted the worker extension.
"""
