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


def test_defining_a_class_again_replaces_it_and_another_object_is_still_refused(clean_registry):
    def cell():  # a notebook cell run twice
        @registry.register("component", "mine")
        class Mine:
            pass

        return Mine

    first, second = cell(), cell()
    assert first is not second and registry.get("component", "mine") is second
    with pytest.raises(ValueError, match="already registered"):
        registry.register("component", "mine")(dict)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("component", "mine")(lambda: None)


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


def test_a_failing_plugin_reports_its_error_every_time_and_the_others_still_load(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_plugins_loaded", False)

    class Broken:
        name = "needs_extra"

        def load(self):
            raise ImportError("no module named extra")

    class Fine:
        def load(self):
            registry.register("model", "plugin_arm")(dict)

    monkeypatch.setattr(registry, "entry_points", lambda group: [Broken(), Fine()])
    for _ in range(2):  # a listing warns and still lists the others
        with pytest.warns(UserWarning, match=r"needs_extra.*no module named extra"):
            assert registry.names("model") == ["plugin_arm"]
    with pytest.raises(KeyError, match=r"no model named 'other'.*needs_extra"):
        registry.get("model", "other")  # an unknown name says so, and that a plugin is broken
    assert registry.get("model", "plugin_arm") is dict  # the plugin after the broken one loaded
