# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
import pytest
from pydantic import Field

from autovision.config.base import PydanticConfig
from autovision.config.validate import (
    assert_config_resolved,
    no_auto,
    pydantic_model_validate,
    suggest_key,
)
from autovision.errors import ConfigValidationError, UnresolvedAutoError
from autovision.types import ImageSizeTuple


class AutoConfig(PydanticConfig):
    value: int | str = "auto"


class SimpleConfig(PydanticConfig):
    batch_size: int = 8
    name: str


class NestedConfig(PydanticConfig):
    inner: SimpleConfig
    sizes: ImageSizeTuple = Field(default=(4, 4))


class TestPydanticConfigPolicy:
    def test_extra_forbidden(self):
        with pytest.raises(Exception, match="extra"):
            SimpleConfig(name="x", typo=1)

    def test_strict_rejects_string_int(self):
        with pytest.raises(Exception):
            SimpleConfig(name="x", batch_size="8")

    def test_protected_namespace_allows_model_fields(self):
        class ModelishConfig(PydanticConfig):
            model: str
            model_args: dict = {}

        config = ModelishConfig(model="a/b")
        assert config.model == "a/b"

    def test_assignment_revalidated(self):
        config = SimpleConfig(name="x")
        with pytest.raises(Exception):
            config.batch_size = "not-int"
        config.batch_size = 16
        assert config.batch_size == 16

    def test_has_auto_recursive(self):
        class AutoHolder(PydanticConfig):
            inner: AutoConfig

        assert AutoConfig().has_auto()
        assert not AutoConfig(value=3).has_auto()
        nested = AutoHolder(inner=AutoConfig())
        assert nested.has_auto()
        nested.inner.value = 1
        assert not nested.has_auto()
        assert not SimpleConfig(name="x").has_auto()

    def test_list_dict_recursion(self):
        class DictConfig(PydanticConfig):
            payload: dict = {}

        assert DictConfig(payload={"a": "auto"}).has_auto()
        assert not DictConfig(payload={"a": [1, "b"]}).has_auto()


class TestPydanticModelValidate:
    def test_success(self):
        config = pydantic_model_validate(SimpleConfig, {"name": "x", "batch_size": 4})
        assert isinstance(config, SimpleConfig)
        assert config.batch_size == 4

    def test_missing_key_message(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            pydantic_model_validate(SimpleConfig, {})
        assert "Missing key: 'name'" in str(exc_info.value)
        assert "Found 1 errors" in str(exc_info.value)

    def test_unknown_key_message_with_did_you_mean(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            pydantic_model_validate(SimpleConfig, {"name": "x", "batch_szie": 4})
        message = str(exc_info.value)
        assert "Unknown key: 'batch_szie'" in message
        assert "Did you mean 'batch_size'?" in message

    def test_unknown_key_without_close_match(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            pydantic_model_validate(SimpleConfig, {"name": "x", "zzzzzz": 4})
        assert "Did you mean" not in str(exc_info.value)

    def test_nested_error_lines(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            pydantic_model_validate(NestedConfig, {"inner": {"batch_szie": 1}})
        assert "Missing key: 'inner.name'" in str(exc_info.value)

    def test_type_error_message(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            pydantic_model_validate(SimpleConfig, {"name": "x", "batch_size": "1"})
        message = str(exc_info.value)
        assert "Invalid type for key 'batch_size'" in message
        assert "with type 'str'" in message

    def test_tuple_coercion_from_list(self):
        config = pydantic_model_validate(NestedConfig, {"inner": {"name": "x"}, "sizes": [2, 2]})
        assert config.sizes == (2, 2)


class TestNoAuto:
    def test_passthrough(self):
        assert no_auto(5) == 5
        assert no_auto("hello") == "hello"
        assert no_auto({"a": 1}) == {"a": 1}

    def test_raises_on_auto(self):
        with pytest.raises(UnresolvedAutoError, match="not a user error"):
            no_auto("auto")

    def test_raises_on_nested_auto(self):
        with pytest.raises(UnresolvedAutoError):
            no_auto({"epochs": "auto"})


class TestAssertConfigResolved:
    def test_resolved_config_passes(self):
        assert_config_resolved(SimpleConfig(name="x"))

    def test_unresolved_config_raises(self):
        with pytest.raises(UnresolvedAutoError):
            assert_config_resolved(AutoConfig())

    def test_dict_recursion(self):
        assert_config_resolved({"a": {"b": 1}})
        with pytest.raises(UnresolvedAutoError):
            assert_config_resolved({"a": {"b": "auto"}})


class TestSuggestKey:
    def test_close_match(self):
        assert suggest_key("batch_szie", ["batch_size", "batch_idx"]) == "batch_size"

    def test_no_match(self):
        assert suggest_key("xyz", ["batch_size"]) is None
