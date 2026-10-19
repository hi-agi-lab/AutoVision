# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
import pytest

from autovision.config.registry import Registry


@pytest.fixture
def registry() -> Registry[str]:
    return Registry("test")


class TestRegistry:
    def test_register_and_get(self, registry):
        @registry.register("alpha")
        def item(): ...

        assert registry.get("alpha") is item
        assert "alpha" in registry
        assert len(registry) == 1

    def test_multiple_names(self, registry):
        @registry.register("a", "b")
        def item(): ...

        assert registry.get("a") is item
        assert registry.get("b") is item

    def test_duplicate_raises(self, registry):
        registry.register("dup")(lambda: None)
        with pytest.raises(ValueError, match="Duplicate registration.*'dup'"):
            registry.register("dup")(lambda: None)

    def test_unknown_get_lists_available(self, registry):
        registry.register("known")(lambda: None)
        with pytest.raises(KeyError, match="Available:.*'known'"):
            registry.get("nope")

    def test_hidden_names(self, registry):
        @registry.register("visible")
        def item(): ...

        registry.register("secret", include_in_list=False)(item)

        assert registry.list_names() == ["visible"]
        assert registry.list_names(include_hidden=True) == ["secret", "visible"]
        assert registry.get("secret") is item  # hidden but reachable

    def test_items_sorted(self, registry):
        for name in ("c", "a", "b"):
            registry.register(name)(name)
        assert [name for name, _ in registry.items()] == ["a", "b", "c"]
