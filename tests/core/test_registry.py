import pytest

from virtualmodelcontrol.core import registry


@pytest.fixture
def clean_registry(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_plugins_loaded", False)
    monkeypatch.setattr(registry, "entry_points", lambda group: [])


def test_register_and_get(clean_registry):
    @registry.register("component", "my_spring")
    class MySpring:
        pass

    assert registry.get("component", "my_spring") is MySpring
    assert registry.names("component") == ["my_spring"]


def test_duplicate_name_is_refused(clean_registry):
    registry.register("model", "arm")(object)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("model", "arm")(int)


def test_unknown_name_lists_known_ones(clean_registry):
    registry.register("model", "arm")(object)
    with pytest.raises(KeyError, match="known: \\['arm'\\]"):
        registry.get("model", "leg")


def test_plugins_load_on_first_miss(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_plugins_loaded", False)

    class FakeEntryPoint:
        def load(self):
            registry.register("model", "plugin_arm")(dict)

    monkeypatch.setattr(registry, "entry_points", lambda group: [FakeEntryPoint()])
    assert registry.get("model", "plugin_arm") is dict
