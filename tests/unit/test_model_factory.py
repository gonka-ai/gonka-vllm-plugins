# SPDX-License-Identifier: Apache-2.0
"""Модель-агностичная фабрика входа: сабкласс строится для любого базового
класса, attach зовётся после load_weights, env-расширение списка парсится."""
import sys
import types

import gonka_poc.models.factory as F


class _Base:
    config = types.SimpleNamespace(hidden_size=64)

    def load_weights(self, weights):
        self.loaded = list(weights)
        return "loaded"

    def parameters(self):
        return iter(())


def test_subclass_attaches_after_load(monkeypatch):
    calls = []
    monkeypatch.setattr(F, "_attach_after_load", lambda m: calls.append(m))
    Sub = F.make_poc_subclass(_Base, "FakeArch")
    assert Sub.__name__ == "FakeArchPoC" and issubclass(Sub, _Base)
    m = Sub()
    assert m.load_weights(["w"]) == "loaded"
    assert calls == [m] and m.loaded == ["w"]


def test_extra_architectures_env(monkeypatch):
    monkeypatch.setenv("POC_ARCHITECTURES",
                       "MyArch=fake_mod_xyz:MyCls, Bad, Other=m:C")
    entries = list(F._extra_architectures())
    assert ("MyArch", "fake_mod_xyz", "MyCls") in entries
    assert ("Other", "m", "C") in entries
    assert len(entries) == 2  # 'Bad' отвергнут с ошибкой в лог


def test_build_skips_missing_modules(monkeypatch):
    fake = types.ModuleType("fake_mod_ok")
    fake.GoodCls = _Base
    monkeypatch.setitem(sys.modules, "fake_mod_ok", fake)
    monkeypatch.setenv("POC_ARCHITECTURES", "Good=fake_mod_ok:GoodCls")
    monkeypatch.setattr(F, "DEFAULT_ARCHITECTURES",
                        [("Gone", "no_such_module_abc", "X")])
    got = dict(F.build_poc_subclasses())
    assert list(got) == ["Good"] and got["Good"].__name__ == "GoodClsPoC"
