"""A replay's signals stay in the worker of the engine that ran it, so the take
must reach every engine, not only the first one."""
import asyncio
from types import SimpleNamespace

from gonka_poc._compat.v0_30 import collective_rpc_each_engine


class FakeEngineCore:
    def __init__(self, engines):
        self.core_engines = engines
        self.asked = []

    async def _call_utility_async(self, method, *args, engine):
        self.asked.append(engine)
        return [{engine: "signals"}]

    async def collective_rpc_async(self, method, args):
        return [{"only": "signals"}]


def run(engines):
    core = FakeEngineCore(engines)
    client = SimpleNamespace(engine_core=core)
    results = asyncio.run(collective_rpc_each_engine(client, "take", (["tag"],)))
    return core, results


def test_every_engine_is_asked_under_data_parallelism():
    core, results = run(["e0", "e1"])
    assert core.asked == ["e0", "e1"]
    assert results == [{"e0": "signals"}, {"e1": "signals"}]


def test_one_engine_takes_the_plain_rpc():
    core, results = run([])
    assert core.asked == []
    assert results == [{"only": "signals"}]
