"""Drift detector: pin the private vLLM surfaces the plugin depends on.

Each test asserts a precise upstream symbol exists with the expected shape
(class name, dataclass field set, method signature, attribute presence).
Run on every CI bump of the vllm pin -- a failure here means the matching
``gonka_poc._compat`` module needs an update before the plugin can target
that vllm minor.

One file covers every supported minor. Where upstream moved a symbol
between minors, the location lives in ``PINS`` below rather than in a
forked copy of this file: a per-minor copy drifts, and the copy that
drifts is the one nobody re-reads.

Adding a minor: add its entry to ``PINS`` and a ``_compat/v0_XX.py``. If a
test needs a genuinely different assertion (not just a different path),
that is a signal the surface changed shape -- write the branch explicitly
rather than loosening the assertion.

Scope: read-only inspection of vllm modules; NO GPU, NO engine startup, NO
forward pass. Safe to run in a vanilla ``pip install vllm`` environment.
"""
from __future__ import annotations

import importlib
import inspect

import pytest

# Per-minor symbol locations. Only entries that actually moved belong here.
PINS = {
    "0.25": {
        "attention_metadata_module": "vllm.v1.attention.backend",
    },
    "0.28": {
        "attention_metadata_module": "vllm.v1.attention.backend",
    },
}

_vllm = pytest.importorskip("vllm")
_VERSION = getattr(_vllm, "__version__", "")
_MINOR = ".".join(_VERSION.split(".")[:2])

# Against an unsupported minor every pin below is EXPECTED to differ, so
# skip wholesale instead of reporting 20 failures that all mean the same
# thing: nobody has vetted this minor yet.
if _MINOR not in PINS:
    pytest.skip(
        f"vllm {_VERSION or '?'} installed; this suite pins "
        f"{', '.join(sorted(PINS))}. Add an entry to PINS (and a matching "
        f"_compat module) to support it.",
        allow_module_level=True,
    )

_PINS = PINS[_MINOR]


# ---------------------------------------------------------------------------- #
# vllm pin
# ---------------------------------------------------------------------------- #

def test_vllm_version_pin() -> None:
    """The installed wheel must be a minor this suite has pins for."""
    vllm = pytest.importorskip("vllm")
    version = getattr(vllm, "__version__", "")
    minor = ".".join(version.split(".")[:2])
    assert minor in PINS, (
        f"gonka-poc targets vllm {', '.join(sorted(PINS))}, got {version!r}. "
        f"The plugin pin (vllm>=0.25.0,<0.29) admits wheels this "
        f"suite has not vetted."
    )


def test_common_attention_metadata_fields() -> None:
    """The PoC forward constructs CommonAttentionMetadata directly.

    Per fork commit 582f087a5 ("restore seq_lens_cpu_upper_bound kwarg for
    MLA attention #9"), the field set shifted across versions. Pin the
    fields the plugin relies on so a future minor bump fails loudly.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module(_PINS["attention_metadata_module"])
    cls = getattr(mod, "CommonAttentionMetadata", None)
    assert cls is not None, (
        f"CommonAttentionMetadata not found at "
        f"{_PINS['attention_metadata_module']} (vllm {_MINOR}) "
        f"-- compat shim needs update."
    )

    # If it's a dataclass / NamedTuple, inspect __annotations__; otherwise
    # fall back to __init__ signature.
    annotations = getattr(cls, "__annotations__", None) or {}
    if not annotations:
        sig = inspect.signature(cls)
        annotations = {
            name: p.annotation
            for name, p in sig.parameters.items()
            if name != "self"
        }

    # Fields the plugin relies on, per fork PoC v2 + #9 fix.
    required_fields = {
        "seq_lens",
        # seq_lens_cpu_upper_bound was REINTRODUCED in 0.20+ MLA path; if
        # this assertion fails on a future minor, the compat shim's
        # build_common_attention_metadata kwarg list must change.
        "seq_lens_cpu_upper_bound",
        # positions: optional field the v0_25 shim passes through; read by the
        # DeepSeek-V4 C128A sparse-MLA builder + SWA compressor. A future
        # minor dropping/renaming it = fleet-wide TypeError at cm construction.
        "positions",
    }
    missing = required_fields - set(annotations)
    assert not missing, (
        f"CommonAttentionMetadata is missing fields {sorted(missing)}; "
        f"present fields = {sorted(annotations)}. "
        f"Update gonka_poc._compat.v0_25.build_common_attention_metadata."
    )


# ---------------------------------------------------------------------------- #
# GPUModelRunner.kv_caches
# ---------------------------------------------------------------------------- #

def test_kv_caches_attribute() -> None:
    """PoC reuses kv_caches[:N] as scratch. Pin the attribute name + type hint."""
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    cls = getattr(mod, "GPUModelRunner", None)
    assert cls is not None, "GPUModelRunner missing"
    annotations = getattr(cls, "__annotations__", {})
    # GPUModelRunner.kv_caches is declared as ``list[torch.Tensor]``
    # at class scope. If the attribute moves to __init__ only, this check
    # still passes as long as one of the parent class scopes carries it --
    # but the compat shim's getattr-based access still works.
    has_class_annotation = "kv_caches" in annotations
    # Fallback: instantiated class has the attr (we can't instantiate here,
    # so we just verify it's referenced in the class source).
    src = inspect.getsource(cls)
    assert has_class_annotation or "kv_caches" in src, (
        f"GPUModelRunner.kv_caches not visible in vllm {_MINOR} source -- "
        "gonka_poc._compat.v0_25.get_kv_cache_pool needs revision."
    )


# ---------------------------------------------------------------------------- #
# EngineClient.abort presence
# ---------------------------------------------------------------------------- #

def test_engine_client_has_abort() -> None:
    """PoCGatingMiddleware does NOT abort, but the PoC router does (via
    EngineClient.abort_request) right before issuing collective_rpc.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.engine.protocol")
    cls = getattr(mod, "EngineClient", None)
    assert cls is not None, "EngineClient ABC missing"
    # Either ``abort`` or ``abort_request`` -- both have appeared.
    assert any(
        hasattr(cls, name) for name in ("abort", "abort_request")
    ), "EngineClient lacks an abort method -- compat shim needs revision."


# ---------------------------------------------------------------------------- #
# EngineClient runtime data-plane (collective_rpc / get_supported_tasks /
# model_config) -- the every-PoC-round call path
# ---------------------------------------------------------------------------- #

def test_engine_client_runtime_surface() -> None:
    """Pin the EngineClient runtime data-plane the plugin hits on every PoC round.

    Three surfaces are pinned here, each backed by a real consumer in the
    plugin source tree. A future vLLM bump that drifts any of these MUST
    surface in unit-CI rather than crash the first PoC request:

        (a) ``EngineClient.collective_rpc(method, timeout, args, kwargs)``
            consumed by:
              * src/gonka_poc/poc/routes.py (compute artifacts per chunk)
              * src/gonka_poc/worker/extension.py
                (await async_llm.collective_rpc("execute_poc_forward", ...))
            -- the PoC dispatch path. If kwargs is renamed/reshaped, every
            PoC forward fails.

        (b) ``EngineClient.get_supported_tasks() -> tuple[str, ...]``
            consumed by:
              * src/gonka_poc/entrypoint/api_router.py
                (``supported_tasks = await engine_client.get_supported_tasks()``)
            -- called once at gonka-vllm-serve init to wire up routes.

        (c) ``EngineClient.model_config: ModelConfig``
            consumed by:
              * src/gonka_poc/entrypoint/api_router.py
                (``model_config = engine_client.model_config`` to read
                ``max_model_len`` etc. for route limits)
            On the EngineClient ABC this is a class-level annotation
            ``model_config: ModelConfig`` (not a property nor an
            @abstractmethod). The concrete ``vllm.v1.engine.async_llm.AsyncLLM``
            sets it as an instance attribute in ``__init__`` via
            ``self.model_config = vllm_config.model_config``. We pin BOTH:
            the annotation on the ABC declares the contract; an absence on
            AsyncLLM would mean the concrete class no longer satisfies it.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.engine.protocol")
    cls = getattr(mod, "EngineClient", None)
    assert cls is not None, "EngineClient ABC missing"

    # ---- (a) collective_rpc -------------------------------------------------
    collective_rpc = getattr(cls, "collective_rpc", None)
    assert collective_rpc is not None and callable(collective_rpc), (
        "EngineClient.collective_rpc missing -- "
        "src/gonka_poc/poc/routes.py and src/gonka_poc/worker/extension.py "
        "issue ``await engine_client.collective_rpc('execute_poc_forward', ...)`` "
        "on every PoC round; without it the PoC dispatch path is dead."
    )
    assert inspect.iscoroutinefunction(collective_rpc), (
        "EngineClient.collective_rpc is no longer a coroutine -- "
        "src/gonka_poc/poc/routes.py and src/gonka_poc/worker/extension.py "
        "await it; if it's now sync the await sites must change."
    )
    rpc_params = set(inspect.signature(collective_rpc).parameters)
    required_rpc = {"method", "timeout", "args", "kwargs"}
    missing_rpc = required_rpc - rpc_params
    assert not missing_rpc, (
        f"EngineClient.collective_rpc signature drifted: missing {sorted(missing_rpc)!r}, "
        f"present = {sorted(rpc_params)!r}. "
        f"src/gonka_poc/poc/routes.py and src/gonka_poc/worker/extension.py "
        f"pass these as keyword args; rename / removal breaks the PoC dispatch."
    )

    # ---- (b) get_supported_tasks --------------------------------------------
    gst = getattr(cls, "get_supported_tasks", None)
    assert gst is not None and callable(gst), (
        "EngineClient.get_supported_tasks missing -- "
        "src/gonka_poc/entrypoint/api_router.py calls "
        "``await engine_client.get_supported_tasks()`` at gonka-vllm-serve "
        "init to wire up routes; without it serve init crashes."
    )
    assert inspect.iscoroutinefunction(gst), (
        "EngineClient.get_supported_tasks is no longer a coroutine -- "
        "src/gonka_poc/entrypoint/api_router.py awaits it."
    )
    gst_params = inspect.signature(gst).parameters
    assert len(gst_params) == 1 and "self" in gst_params, (
        f"EngineClient.get_supported_tasks acquired non-self parameters: "
        f"{list(gst_params)!r}. "
        f"src/gonka_poc/entrypoint/api_router.py calls it with no args; "
        f"either the call site must pass the new args or the compat shim "
        f"must wrap it."
    )

    # ---- (c) model_config ----------------------------------------------------
    # The ABC declares ``model_config: ModelConfig`` as a class-level
    # annotation (not a @property, not an @abstractmethod). Pin both the
    # annotation on the ABC and the instance-attribute set-site on the
    # concrete AsyncLLM, so either kind of drift surfaces here.
    abc_annotations = getattr(cls, "__annotations__", {}) or {}
    assert "model_config" in abc_annotations, (
        f"EngineClient.model_config annotation missing from ABC; "
        f"present annotations = {sorted(abc_annotations)!r}. "
        f"src/gonka_poc/entrypoint/api_router.py does "
        f"``model_config = engine_client.model_config`` to read max_model_len "
        f"etc.; if the contract no longer requires this attribute, every "
        f"concrete impl is free to omit it."
    )
    # The annotation type SHOULD be ModelConfig (from vllm.config).
    from vllm.config import ModelConfig as _ModelConfig  # noqa: WPS433
    annotated_type = abc_annotations.get("model_config")
    assert annotated_type is _ModelConfig, (
        f"EngineClient.model_config annotation drifted from "
        f"vllm.config.ModelConfig to {annotated_type!r}; "
        f"src/gonka_poc/entrypoint/api_router.py reads ModelConfig fields "
        f"(max_model_len, etc.) and will break if the type changes."
    )

    # Concrete impl: AsyncLLM (the v1 path serve uses) MUST also expose
    # model_config -- either as a class attribute, a property, or an instance
    # attribute set in __init__. We can't instantiate the engine here, so
    # check the __init__ source for the set-site.
    async_llm_mod = importlib.import_module("vllm.v1.engine.async_llm")
    async_llm = getattr(async_llm_mod, "AsyncLLM", None)
    assert async_llm is not None, (
        "vllm.v1.engine.async_llm.AsyncLLM missing -- "
        "gonka-vllm-serve cannot wire model_config without it."
    )
    has_attr = (
        "model_config" in (getattr(async_llm, "__annotations__", {}) or {})
        or isinstance(getattr(async_llm, "model_config", None), property)
        or "self.model_config" in inspect.getsource(async_llm.__init__)
    )
    assert has_attr, (
        "AsyncLLM no longer sets ``self.model_config`` in __init__ nor "
        "exposes it as a class attribute/property -- "
        "src/gonka_poc/entrypoint/api_router.py will raise AttributeError "
        "on the very first request."
    )


# ---------------------------------------------------------------------------- #
# worker_extension_cls injection point
# ---------------------------------------------------------------------------- #

def test_worker_extension_cls_supported() -> None:
    """``--worker-extension-cls`` is the documented hook for PoCWorkerExtension."""
    pytest.importorskip("vllm")
    parallel = importlib.import_module("vllm.config.parallel")
    pc_cls = getattr(parallel, "ParallelConfig", None)
    assert pc_cls is not None, "ParallelConfig moved -- compat shim needs revision."
    assert "worker_extension_cls" in getattr(pc_cls, "__annotations__", {}) or hasattr(
        pc_cls, "worker_extension_cls"
    ), "ParallelConfig.worker_extension_cls vanished -- compat shim needs revision."


# ---------------------------------------------------------------------------- #
# api_server composition surface
# ---------------------------------------------------------------------------- #

def test_api_server_composition_symbols() -> None:
    """``gonka-vllm-serve`` calls these public helpers verbatim."""
    pytest.importorskip("vllm")
    api = importlib.import_module("vllm.entrypoints.openai.api_server")
    for name in ("build_app", "build_async_engine_client", "init_app_state", "setup_server"):
        assert hasattr(api, name), (
            f"vllm.entrypoints.openai.api_server.{name} missing -- "
            f"gonka_poc.entrypoint.api_router._run_server needs revision."
        )


# ---------------------------------------------------------------------------- #
# CLI args helpers
# ---------------------------------------------------------------------------- #

def test_cli_args_helpers() -> None:
    pytest.importorskip("vllm")
    cli = importlib.import_module("vllm.entrypoints.openai.cli_args")
    for name in ("make_arg_parser", "validate_parsed_serve_args"):
        assert hasattr(cli, name), (
            f"vllm.entrypoints.openai.cli_args.{name} missing -- "
            f"gonka_poc.entrypoint.api_router.main needs revision."
        )


def test_flexible_argument_parser_import_path() -> None:
    """``FlexibleArgumentParser`` MUST be importable from
    ``vllm.utils.argparse_utils`` since v0.22.

    Background: pre-0.22 ``vllm/utils.py`` was a flat module that exported
    ``FlexibleArgumentParser`` directly. In 0.22.0 ``vllm.utils`` became a
    package; the symbol moved to ``vllm.utils.argparse_utils`` and is NOT
    re-exported from the package ``__init__.py``. The legacy import
    ``from vllm.utils import FlexibleArgumentParser`` raises ImportError
    at runtime (the smoke-help job caught exactly this against a real
    pin). ``gonka_poc.entrypoint.api_router.main`` now imports from the
    canonical location with a fallback to the flat path; this contract
    test gives the smoke-help job a symbol-level guard so regressions
    surface in unit-CI rather than the heavier subprocess job.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.utils.argparse_utils")
    cls = getattr(mod, "FlexibleArgumentParser", None)
    assert cls is not None and inspect.isclass(cls), (
        "vllm.utils.argparse_utils.FlexibleArgumentParser missing -- "
        "gonka_poc.entrypoint.api_router.main needs revision "
        "(currently falls back to vllm.utils.FlexibleArgumentParser, "
        "which is gone since 0.22)."
    )


def test_cli_env_setup_import_path() -> None:
    """``cli_env_setup`` MUST be importable from
    ``vllm.entrypoints.serve.utils.api_utils``.

    Background: upstream ``api_server.py:__main__`` calls
    ``cli_env_setup()`` before ``uvloop.run`` to set
    ``VLLM_WORKER_MULTIPROC_METHOD=spawn``. Skipping it crashes TP>1 / PP>1
    launches with the classic CUDA-in-forked-process error.
    ``gonka_poc.entrypoint.api_router.main`` mirrors that call; if the
    import path moves in a future vLLM release, the soft fallback in main()
    kicks in (warning + unsafe default), and this contract test fires loudly
    so we can update the import.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.entrypoints.serve.utils.api_utils")
    fn = getattr(mod, "cli_env_setup", None)
    assert fn is not None and callable(fn), (
        "vllm.entrypoints.serve.utils.api_utils.cli_env_setup missing -- "
        "gonka_poc.entrypoint.api_router.main needs revision (currently "
        "soft-falls-back to a warning, leaving multiproc method unsafe)."
    )


# ---------------------------------------------------------------------------- #
# ModelRegistry surface
# ---------------------------------------------------------------------------- #

def test_model_registry_register_model() -> None:
    pytest.importorskip("vllm")
    vllm = importlib.import_module("vllm")
    registry = getattr(vllm, "ModelRegistry", None)
    assert registry is not None, "vllm.ModelRegistry not re-exported -- plugin needs revision."
    assert hasattr(registry, "register_model"), "ModelRegistry.register_model missing."


# ---------------------------------------------------------------------------- #
# Distributed group getters
# ---------------------------------------------------------------------------- #

def test_distributed_groups_present() -> None:
    """PoCWorkerExtension uses these for the inter-rank PoC dispatch flow.

    ``get_tp_group`` / ``get_pp_group`` live in
    ``vllm.distributed.parallel_state`` and are re-exported from the package
    root via ``vllm.distributed`` (``from .parallel_state import *``). The
    plugin imports the short path -- pin both surfaces so a re-shuffle that
    drops the re-export breaks here, not at runtime.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.distributed")
    for name in ("get_pp_group", "get_tp_group"):
        fn = getattr(mod, name, None)
        assert fn is not None and callable(fn), (
            f"vllm.distributed.{name} missing or non-callable -- "
            f"gonka_poc._compat.v0_25 must reroute via "
            f"vllm.distributed.parallel_state."
        )
        # Both are zero-arg getters returning a GroupCoordinator. If a
        # future minor introduces required args, the compat shim must wrap.
        sig = inspect.signature(fn)
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        assert not required, (
            f"vllm.distributed.{name} acquired required args {required!r}; "
            f"PoCWorkerExtension call site needs revision."
        )


# ---------------------------------------------------------------------------- #
# broadcast_tensor_dict fast-path
# ---------------------------------------------------------------------------- #

def test_communication_op_broadcast() -> None:
    """Optional fast-path for PoC payload broadcast across the TP group.

    Absence here means PoCWorkerExtension must fall back to
    ``get_tp_group().broadcast_tensor_dict(...)`` directly. We still pin the
    free-function form so a future rename forces a compat-shim review.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.distributed.communication_op")
    fn = getattr(mod, "broadcast_tensor_dict", None)
    assert fn is not None and callable(fn), (
        "vllm.distributed.communication_op.broadcast_tensor_dict missing -- "
        "compat shim needs update (fallback: get_tp_group().broadcast_tensor_dict)."
    )
    sig = inspect.signature(fn)
    # The signature is (tensor_dict=None, src=0). The plugin passes
    # both positionally; pin the parameter names to catch a kwarg rename.
    params = list(sig.parameters)
    assert params[:2] == ["tensor_dict", "src"], (
        f"broadcast_tensor_dict signature changed: {params!r}; "
        f"compat shim needs update."
    )


# ---------------------------------------------------------------------------- #
# set_forward_context contextmanager
# ---------------------------------------------------------------------------- #

def test_forward_context_set() -> None:
    """PoC forward needs to enter a forward context bearing AttentionMetadata.

    ``vllm.forward_context.set_forward_context`` is a ``@contextmanager``
    accepting ``(attn_metadata, vllm_config, ...)``. Pin both required
    parameters so a future split (e.g., into ``begin_forward`` /
    ``end_forward``) is caught here.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.forward_context")
    fn = getattr(mod, "set_forward_context", None)
    assert fn is not None and callable(fn), (
        "vllm.forward_context.set_forward_context missing -- "
        "compat shim needs update."
    )
    sig = inspect.signature(fn)
    params = list(sig.parameters)
    for required in ("attn_metadata", "vllm_config"):
        assert required in params, (
            f"set_forward_context lost parameter {required!r}; "
            f"present = {params!r}; compat shim needs update."
        )


# ---------------------------------------------------------------------------- #
# IntermediateTensors (PP>1 PoC)
# ---------------------------------------------------------------------------- #

def test_intermediate_tensors() -> None:
    """PP > 1 PoC may pass / receive IntermediateTensors between stages.

    It lives at ``vllm.sequence.IntermediateTensors`` as a
    ``@dataclass`` with a single field ``tensors: dict[str, torch.Tensor]``.
    If a future minor relocates it (likely candidates:
    ``vllm.v1.outputs`` or ``vllm.v1.sequence``), this test fails and the
    compat shim must add a fallback import path.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.sequence")
    cls = getattr(mod, "IntermediateTensors", None)
    assert cls is not None, (
        "vllm.sequence.IntermediateTensors missing -- "
        "try vllm.v1.outputs or vllm.v1.sequence and update the compat shim."
    )
    # Pin the dataclass field name; a rename of ``tensors`` would break the
    # PP-stage hand-off in PoCWorkerExtension.
    annotations = getattr(cls, "__annotations__", {}) or {}
    assert "tensors" in annotations, (
        f"IntermediateTensors field set changed: {sorted(annotations)!r}; "
        f"compat shim needs update."
    )


# ---------------------------------------------------------------------------- #
# launcher.serve_http
# ---------------------------------------------------------------------------- #

def test_launcher_serve_http() -> None:
    """``gonka-vllm-serve`` composition calls ``serve_http`` directly.

    Confirmed at ``vllm.entrypoints.launcher.serve_http`` -- an
    ``async def`` taking ``(app, sock, enable_ssl_refresh=False, **uvicorn_kwargs)``.
    If a future minor inlines it back into ``api_server`` or renames it to
    ``_serve_http``, this test fires.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.entrypoints.launcher")
    fn = getattr(mod, "serve_http", None)
    assert fn is not None and callable(fn), (
        "vllm.entrypoints.launcher.serve_http missing -- "
        "gonka_poc.entrypoint.api_router._run_server needs revision."
    )
    assert inspect.iscoroutinefunction(fn), (
        "serve_http is no longer a coroutine -- await-site needs revision."
    )
    sig = inspect.signature(fn)
    params = list(sig.parameters)
    # The plugin passes `app` and `sock` positionally; pin the leading pair.
    assert params[:2] == ["app", "sock"], (
        f"serve_http signature drifted: {params!r}; compat shim needs update."
    )


# ---------------------------------------------------------------------------- #
# SamplingParams residual-fork bridge (logprobs_mode + enforced_token_ids)
# ---------------------------------------------------------------------------- #

def test_sampling_params_has_fork_patches() -> None:
    """Pin the residual-fork SamplingParams fields the plugin depends on.

    Background -- production engines ship a residual build of upstream vllm
    carrying the sampler patches (poc-sampler-residual-v0.25). Two of those
    patches add per-request ``logprobs_mode`` and ``enforced_token_ids``
    fields to SamplingParams; the PoC v2 mixed-mode sampling path
    (validator replay + logits-mode selection) reads them at request
    admission.

    Why this test belongs in the PLUGIN contract suite (not just the fork):
        The plugin advertises ``vllm>=0.25.0,<0.29`` as its
        install pin. A vanilla ``pip install`` of a supported wheel (no
        ``+gonka.sampler``) satisfies that pin -- and the other contract
        tests stay GREEN against vanilla vllm, but engine startup
        crashes the moment a request with ``logprobs_mode`` arrives. This
        pin catches that misconfiguration BEFORE production.

    Pattern: same as the residual branch's
    ``tests/contract/test_sampler_surface.py::test_sampling_params_has_poc_fields``.
    We read ``__struct_fields__`` first (msgspec.Struct); the fallback
    chain to ``__annotations__`` and the ``__init__`` signature covers a
    dataclass conversion.
    """
    vllm = pytest.importorskip("vllm")
    # CI installs a vanilla upstream wheel for all the OTHER contract tests
    # in this file — those test the upstream surface and should stay green
    # there. THIS test only meaningfully runs when a residual wheel is
    # installed; on vanilla vllm it would fail loudly with no actionable
    # signal (the operator running CI is not the same as the operator
    # deploying production). Skip-with-clear-message so the test serves as
    # a runtime alert (visible in pytest -v output) without polluting the
    # CI failure surface.
    version = getattr(vllm, "__version__", "")
    if "+gonka.sampler" not in version:
        pytest.skip(
            f"vllm=={version!r} is the vanilla upstream wheel; this test "
            f"requires a residual wheel carrying the sampler stack "
            f"(local version suffix '+gonka.sampler', e.g. "
            f"vllm==0.25.1+gonka.sampler1). Production deployments MUST "
            f"install the residual wheel — see MIGRATION_FROM_FORK.md and "
            f"docker/Dockerfile.gonka-poc in the vLLM residual repository."
        )

    mod = importlib.import_module("vllm.sampling_params")
    cls = getattr(mod, "SamplingParams", None)
    assert cls is not None, "vllm.sampling_params.SamplingParams missing"

    # Primary: msgspec.Struct exposes the field tuple via __struct_fields__.
    fields: set = set(getattr(cls, "__struct_fields__", ()) or ())
    # Fallback chain: dataclass / NamedTuple / annotated class.
    if not fields:
        fields = set(getattr(cls, "__annotations__", {}) or {})
    if not fields:
        sig = inspect.signature(cls.__init__)
        fields = {
            name for name, p in sig.parameters.items()
            if name != "self"
        }

    for required in ("logprobs_mode", "enforced_token_ids"):
        assert required in fields, (
            f"SamplingParams.{required} missing despite vllm version {version!r} "
            f"claiming to be the residual wheel — patch regression. Re-check "
            f"commit 1c5368212 application on the residual branch for "
            f"vllm {_MINOR}."
        )


# ---------------------------------------------------------------------------- #
# OpenAIServingChat export (restructured in 0.23)
# ---------------------------------------------------------------------------- #

def test_openai_serving_chat_export() -> None:
    """Pin the OpenAIServingChat import path for CLI-surface stability.

    Nothing in ``src/`` imports OpenAIServingChat directly — the plugin
    reuses vLLM's own OpenAI serving stack via ``gonka-vllm-serve``. The
    pin stays because a relocation of this class signals a restructure of
    the OpenAI entrypoint package that the serve wiring builds on.

    NOTE -- 0.23 restructure: ``vllm.entrypoints.openai.serving_chat`` no
    longer exists. The class moved to
    ``vllm.entrypoints.openai.chat_completion.serving.OpenAIServingChat``.
    The ``chat_completion`` package's ``__init__.py`` does NOT re-export
    the symbol, so any deep import must use the full path.
    """
    pytest.importorskip("vllm")

    # The pre-0.23 path is gone -- assert that explicitly so a future
    # vllm bump that *restores* the alias surfaces here and lets us
    # simplify the compat shim.
    try:
        legacy = importlib.import_module("vllm.entrypoints.openai.serving_chat")
    except ImportError:
        legacy = None
    if legacy is not None and hasattr(legacy, "OpenAIServingChat"):
        # Legacy path restored -- compat shim can drop the deep-path fallback.
        return

    mod = importlib.import_module("vllm.entrypoints.openai.chat_completion.serving")
    cls = getattr(mod, "OpenAIServingChat", None)
    assert cls is not None and inspect.isclass(cls), (
        "OpenAIServingChat not found at "
        "vllm.entrypoints.openai.chat_completion.serving -- "
        "the OpenAI entrypoint package restructured again "
        "(check vllm.entrypoints.openai.chat_completion.__init__ for re-exports)."
    )
    # Pin the documented entry-point method; a rename (e.g.
    # ``handle_chat_request``) signals a CLI-surface restructure worth
    # reviewing at the next pin bump.
    assert hasattr(cls, "create_chat_completion"), (
        "OpenAIServingChat.create_chat_completion missing -- "
        "the OpenAI chat serving surface restructured."
    )


# ---------------------------------------------------------------------------- #
# _compat dispatcher contract (regression for the lru_cache function-binding bug)
# ---------------------------------------------------------------------------- #

def test_compat_current_returns_module() -> None:
    """``from gonka_poc._compat import current`` MUST be a callable resolver
    that returns the dispatched compat module.

    Regression: a previous revision exposed ``current`` as the
    ``lru_cache``-wrapped function itself. ``from gonka_poc._compat import
    current as compat`` then bound the function, so every subsequent
    ``compat.build_common_attention_metadata(...)`` raised ``AttributeError``
    (functions do not carry arbitrary attributes) and the first PoC forward
    crashed in production. The contract pinned here:

        from gonka_poc._compat import current
        mod = current()          # callable, returns a module
        mod.build_common_attention_metadata(...)  # symbols live on the module

    is the only shape that survives both ``from ... import current`` (which
    bypasses PEP 562 ``__getattr__``) and attribute-style ``_compat.current``.
    """
    pytest.importorskip("vllm")
    from gonka_poc._compat import current

    assert callable(current), (
        "gonka_poc._compat.current must be callable -- "
        "the lru_cache function-binding regression has returned. "
        "Consumers do ``from gonka_poc._compat import current`` and then "
        "``compat = current(); compat.<symbol>(...)``."
    )

    import types

    mod = current()
    assert isinstance(mod, types.ModuleType), (
        f"current() must return a module, got {type(mod)!r}. "
        "Check gonka_poc._compat.__init__._current_impl resolver."
    )

    for symbol in (
        "build_common_attention_metadata",
        "build_attn_metadata_per_group",
        "get_kv_cache_pool",
        "abort_all_requests",
        "install_engine_core_poc_methods",
        "borrow_poc_blocks",
        "return_poc_blocks",
    ):
        attr = getattr(mod, symbol, None)
        assert callable(attr), (
            f"compat module missing callable {symbol!r}; "
            f"present attrs = {sorted(a for a in dir(mod) if not a.startswith('_'))!r}. "
            f"Either gonka_poc._compat.v0_25 dropped the export or _DISPATCH "
            f"is pointing at the wrong module."
        )


# ---------------------------------------------------------------------------- #
# KV block borrowing surface (PoC validation without aborting inference)
# ---------------------------------------------------------------------------- #

def test_kv_block_pool_borrow_surface() -> None:
    """Pin every private surface the KV borrow path touches.

    The compat shim (``install_engine_core_poc_methods`` /
    ``borrow_poc_blocks`` / ``return_poc_blocks``) relies on:

      * ``EngineCore`` (vllm/v1/engine/core.py) loading general plugins in
        the engine-core process — that is what makes class-level method
        injection reach the process that owns the BlockPool;
      * ``Scheduler.kv_cache_manager`` -> ``KVCacheManager.block_pool``
        (the ONE pool shared by every kv-cache group) and
        ``KVCacheManager.kv_cache_config`` (per-group block sizes for
        lease sizing);
      * ``BlockPool.get_new_blocks`` / ``free_blocks`` /
        ``get_num_free_blocks`` / ``blocks`` and the null-block sentinel;
      * the UTILITY RPC surface: ``AsyncLLM.engine_core`` +
        ``call_utility_async`` on the MP client; and
        ``AsyncLLM.reset_prefix_cache`` (poisoned-cache reset after
        in-place PoC rounds).
    """
    import inspect

    BlockPool = importlib.import_module("vllm.v1.core.block_pool").BlockPool
    KVCacheManager = importlib.import_module(
        "vllm.v1.core.kv_cache_manager").KVCacheManager
    Scheduler = importlib.import_module(
        "vllm.v1.core.sched.scheduler").Scheduler
    core_mod = importlib.import_module("vllm.v1.engine.core")
    AsyncLLM = importlib.import_module("vllm.v1.engine.async_llm").AsyncLLM
    AsyncMPClient = importlib.import_module(
        "vllm.v1.engine.core_client").AsyncMPClient

    for name in ("get_new_blocks", "free_blocks", "get_num_free_blocks"):
        assert callable(getattr(BlockPool, name, None)), (
            f"BlockPool.{name} vanished — the borrow lease cannot be taken/"
            "returned; update the compat shim")
    pool_init = inspect.getsource(BlockPool.__init__)
    assert "self.null_block" in pool_init, (
        "BlockPool null-block sentinel moved — re-verify the is_null guard "
        "in gonka_poc_borrow_blocks")
    assert "self.blocks" in pool_init, (
        "BlockPool.blocks list moved — gonka_poc_return_blocks indexes it")

    mgr_init = inspect.getsource(KVCacheManager.__init__)
    assert "self.block_pool = self.coordinator.block_pool" in mgr_init, (
        "KVCacheManager no longer exposes the coordinator's single "
        "BlockPool — the one-lease-covers-all-groups premise must be "
        "re-verified before trusting borrowed PoC layouts")
    assert "self.kv_cache_config = kv_cache_config" in mgr_init, (
        "KVCacheManager.kv_cache_config moved — lease sizing reads "
        "kv_cache_groups[*].kv_cache_spec.block_size from it")

    assert "self.kv_cache_manager" in inspect.getsource(Scheduler.__init__), (
        "Scheduler.kv_cache_manager moved — EngineCore borrow methods "
        "traverse scheduler.kv_cache_manager.block_pool")

    assert hasattr(core_mod, "EngineCore"), "EngineCore class moved"
    assert "load_general_plugins" in inspect.getsource(core_mod), (
        "EngineCore no longer loads general plugins — the injected borrow "
        "methods would never reach the engine-core process")

    assert callable(getattr(AsyncMPClient, "call_utility_async", None)), (
        "call_utility_async gone from the MP client — the frontend borrow "
        "wrappers have no transport")
    assert "self.engine_core" in inspect.getsource(AsyncLLM.__init__), (
        "AsyncLLM.engine_core attribute moved — compat wrappers getattr it")
    assert callable(getattr(AsyncLLM, "reset_prefix_cache", None)), (
        "AsyncLLM.reset_prefix_cache gone — in-place PoC rounds would leave "
        "poisoned prefix-cache entries with no way to drop them")
    assert "reset_running_requests" in inspect.signature(
        AsyncLLM.reset_prefix_cache).parameters, (
        "reset_prefix_cache lost the reset_running_requests kwarg — the "
        "reservation layer escalates through it when the plain reset "
        "returns False (blocks still held)")

    # Semantic pins the lease design DEPENDS on (not just symbol presence):
    # (a) leasing must evict cached hashes, else a leased block can still be
    #     served from the prefix cache after PoC overwrote it;
    assert "_maybe_evict_cached_block" in inspect.getsource(
        BlockPool.get_new_blocks), (
        "get_new_blocks no longer evicts cached hashes — the 'lease is "
        "disjoint from the prefix cache' premise fails; re-verify the "
        "borrow design before trusting leased PoC forwards")
    # (b) the borrow/return methods read b.block_id / b.is_null and index
    #     pool.blocks[bid] — pin the field names.
    KVCacheBlock = importlib.import_module(
        "vllm.v1.core.kv_cache_utils").KVCacheBlock
    for field in ("block_id", "is_null"):
        assert hasattr(KVCacheBlock, field) or field in getattr(
            KVCacheBlock, "__dataclass_fields__", {}), (
            f"KVCacheBlock.{field} renamed — gonka_poc_borrow_blocks/"
            "return_blocks read it")
